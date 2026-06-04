"""HTTP serving layer over the attribute pipeline.

A small FastAPI app: upload audio, poll the job, fetch the result as
speaker-attributed JSON or SRT, or follow it live over a WebSocket that
emits each segment as the decode produces it.

The pipeline runs diarization first (it is the fast stage), then streams
faster-whisper's lazily decoded segments through the word-speaker
alignment, so attributed segments leave the server before the transcription
finishes. A client that connects after completion gets the same event
stream replayed.

Jobs run on one worker thread. The models load once and the GPU is a serial
resource, and the measured real-time factors say a single worker is enough
for interactive use: small/float16 transcribes at 0.022 RTF on an A10, so
one worker sustains roughly 45x real time before queueing matters.

Run it:

    uv sync --extra models --extra serve
    uv run uvicorn diarlab.server:app --port 8000

Configuration is read from the environment at startup: DIARLAB_MODEL
(default small), DIARLAB_DEVICE (cpu), DIARLAB_COMPUTE (int8). The app
object is also constructable with an injected pipeline function, which is
how the tests exercise the HTTP surface without model downloads.
"""

# No `from __future__ import annotations` here: FastAPI resolves the
# UploadFile annotation at runtime, and it is imported lazily below.
import os
import tempfile
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Annotated

from .formats import segments_to_dict, segments_to_srt
from .types import Segment

AttributeFn = Callable[[Path], list[Segment]]
StreamFn = Callable[[Path], "object"]  # Path -> iterator of Segment

ALLOWED_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued -> running -> done | error
    error: str | None = None
    segments: list[Segment] = field(default_factory=list)

    def to_response(self) -> dict:
        out: dict = {"id": self.id, "filename": self.filename, "status": self.status}
        if self.status == "done":
            out["result"] = segments_to_dict(self.segments)
        if self.status == "error":
            out["error"] = self.error
        return out


def _stream_from_env() -> StreamFn:
    """The real pipeline, configured from the environment, loaded lazily.

    Diarization runs first and in full (VAD, embeddings, and clustering are
    the cheap stages), then ASR segments are attributed and yielded as
    faster-whisper decodes them.
    """
    model = os.environ.get("DIARLAB_MODEL", "small")
    device = os.environ.get("DIARLAB_DEVICE", "cpu")
    compute = os.environ.get("DIARLAB_COMPUTE", "int8")

    def run(path: Path):
        from .align import assign_words, group_segments
        from .asr import transcribe_stream
        from .audio import load_audio
        from .diarize import ClusteredConfig, diarize_clustered

        audio, rate = load_audio(path)
        turns = diarize_clustered(audio, rate, ClusteredConfig(device=device))
        for words in transcribe_stream(path, model_size=model, device=device, compute_type=compute):
            yield from group_segments(assign_words(words, turns))

    return run


def _live_session_from_env():
    """A LiveSession over the real models, configured from the environment."""
    import numpy as np

    from .asr import transcribe_array
    from .diarize import ClusteredConfig
    from .embeddings import EcapaEmbedder
    from .live import LiveSession
    from .vad import detect_speech

    model = os.environ.get("DIARLAB_MODEL", "small")
    device = os.environ.get("DIARLAB_DEVICE", "cpu")
    compute = os.environ.get("DIARLAB_COMPUTE", "int8")
    embedder = EcapaEmbedder(device=device)

    return LiveSession(
        vad_fn=lambda chunk: detect_speech(chunk, 16_000),
        embed_fn=lambda chunk, windows: embedder.embed_windows(
            np.asarray(chunk), 16_000, windows
        ),
        transcribe_fn=lambda chunk: transcribe_array(
            chunk, model_size=model, device=device, compute_type=compute
        ),
        config=ClusteredConfig(device=device),
    )


def create_app(
    attribute_fn: AttributeFn | None = None,
    stream_fn: StreamFn | None = None,
    live_session_factory=None,
):
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
        from fastapi.responses import HTMLResponse, PlainTextResponse
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "fastapi is not installed; install the serving layer with `uv sync --extra serve`"
        ) from exc
    import asyncio

    if stream_fn is None:
        if attribute_fn is not None:
            # a batch pipeline streams trivially: one burst at the end
            def stream_fn(path, _fn=attribute_fn):
                yield from _fn(path)
        else:
            stream_fn = _stream_from_env()

    app = FastAPI(title="diarlab", description="speaker-attributed transcription")
    jobs: dict[str, Job] = {}
    lock = Lock()
    executor = ThreadPoolExecutor(max_workers=1)

    def process(job_id: str, path: Path) -> None:
        with lock:
            jobs[job_id].status = "running"
        try:
            for segment in stream_fn(path):
                with lock:
                    jobs[job_id].segments.append(segment)
            with lock:
                jobs[job_id].status = "done"
        except Exception as exc:  # surface the failure on the job, not the worker
            with lock:
                jobs[job_id].status = "error"
                jobs[job_id].error = f"{type(exc).__name__}: {exc}"
        finally:
            path.unlink(missing_ok=True)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/jobs", status_code=202)
    async def submit(file: Annotated[UploadFile, File()]) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                415, f"unsupported file type {suffix or '(none)'}; send one of "
                + ", ".join(sorted(ALLOWED_SUFFIXES))
            )
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty upload")
        job_id = uuid.uuid4().hex[:12]
        tmp = Path(tempfile.gettempdir()) / f"diarlab_{job_id}{suffix}"
        tmp.write_bytes(data)
        job = Job(id=job_id, filename=file.filename or "upload")
        with lock:
            jobs[job_id] = job
        executor.submit(process, job_id, tmp)
        return {"id": job_id, "status": job.status}

    def _get(job_id: str) -> Job:
        with lock:
            job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job

    @app.get("/jobs/{job_id}")
    def status(job_id: str) -> dict:
        return _get(job_id).to_response()

    @app.get("/jobs/{job_id}/srt", response_class=PlainTextResponse)
    def srt(job_id: str) -> str:
        job = _get(job_id)
        if job.status != "done":
            raise HTTPException(409, f"job is {job.status}, not done")
        return segments_to_srt(job.segments)

    @app.websocket("/jobs/{job_id}/stream")
    async def stream(websocket: WebSocket, job_id: str) -> None:
        """Emit each attributed segment as it is produced, then the final status.

        Already-produced segments are replayed first, so connecting late
        (or after completion) yields the same stream.
        """
        await websocket.accept()
        with lock:
            known = job_id in jobs
        if not known:
            await websocket.send_json({"type": "error", "error": "no such job"})
            await websocket.close(code=1008)
            return
        sent = 0
        while True:
            with lock:
                job = jobs[job_id]
                fresh = job.segments[sent:]
                job_status, job_error = job.status, job.error
            for segment in fresh:
                payload = segments_to_dict([segment])["segments"][0]
                await websocket.send_json({"type": "segment", "segment": payload})
            sent += len(fresh)
            if job_status in ("done", "error"):
                final: dict = {"type": "status", "status": job_status}
                if job_error:
                    final["error"] = job_error
                await websocket.send_json(final)
                break
            await asyncio.sleep(0.05)
        await websocket.close()

    @app.get("/live", response_class=HTMLResponse)
    def live_page() -> str:
        return (Path(__file__).parent / "static" / "live.html").read_text(encoding="utf-8")

    @app.websocket("/live/ws")
    async def live_ws(websocket: WebSocket) -> None:
        """Near-real-time mode: int16 PCM in, attributed segments out.

        Protocol: the client sends one JSON text frame {"sample_rate": N},
        then binary int16 mono PCM frames at that rate; a text frame
        {"type": "stop"} flushes the tail. The server resamples to 16 kHz,
        processes rolling chunks, and pushes one JSON event per attributed
        segment. Inference runs on the shared single worker so live audio
        and uploaded jobs serialize on the same GPU.
        """
        import numpy as np

        from .audio import resample

        await websocket.accept()
        try:
            hello = await websocket.receive_json()
            source_rate = int(hello.get("sample_rate", 16_000))
        except Exception:
            await websocket.send_json({"type": "error", "error": "expected a JSON hello frame"})
            await websocket.close(code=1003)
            return

        loop = asyncio.get_running_loop()
        session = (live_session_factory or _live_session_from_env)()
        await websocket.send_json({"type": "ready"})

        async def emit(segments) -> None:
            for segment in segments:
                payload = segments_to_dict([segment])["segments"][0]
                await websocket.send_json({"type": "segment", "segment": payload})

        try:
            while True:
                message = await websocket.receive()
                if message.get("bytes") is not None:
                    pcm = np.frombuffer(message["bytes"], dtype=np.int16)
                    samples = resample(pcm.astype(np.float32) / 32768.0, source_rate, 16_000)
                    segments = await loop.run_in_executor(executor, session.feed, samples)
                    await emit(segments)
                elif (
                    message.get("text") is not None
                    or message.get("type") == "websocket.disconnect"
                ):
                    break  # stop request or disconnect: flush below
        except Exception:
            pass  # client went away mid-receive; still try to flush
        try:
            await emit(await loop.run_in_executor(executor, session.flush))
            await websocket.send_json({"type": "status", "status": "done"})
            await websocket.close()
        except Exception:
            pass  # nothing to flush to a closed socket

    return app


# uvicorn entry point: `uvicorn diarlab.server:app`
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)
