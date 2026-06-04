"""HTTP serving layer over the attribute pipeline.

A small FastAPI app: upload audio, poll the job, fetch the result as
speaker-attributed JSON or SRT.

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


def _attribute_from_env() -> AttributeFn:
    """The real pipeline, configured from the environment, loaded lazily."""
    model = os.environ.get("DIARLAB_MODEL", "small")
    device = os.environ.get("DIARLAB_DEVICE", "cpu")
    compute = os.environ.get("DIARLAB_COMPUTE", "int8")

    def run(path: Path) -> list[Segment]:
        from .align import assign_words, group_segments
        from .asr import transcribe
        from .audio import load_audio
        from .diarize import ClusteredConfig, diarize_clustered

        result = transcribe(path, model_size=model, device=device, compute_type=compute)
        audio, rate = load_audio(path)
        turns = diarize_clustered(audio, rate, ClusteredConfig(device=device))
        return group_segments(assign_words(result.words, turns))

    return run


def create_app(attribute_fn: AttributeFn | None = None):
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from fastapi.responses import PlainTextResponse
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "fastapi is not installed; install the serving layer with `uv sync --extra serve`"
        ) from exc

    run_pipeline = attribute_fn or _attribute_from_env()
    app = FastAPI(title="diarlab", description="speaker-attributed transcription")
    jobs: dict[str, Job] = {}
    lock = Lock()
    executor = ThreadPoolExecutor(max_workers=1)

    def process(job_id: str, path: Path) -> None:
        with lock:
            jobs[job_id].status = "running"
        try:
            segments = run_pipeline(path)
            with lock:
                jobs[job_id].segments = segments
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

    return app


# uvicorn entry point: `uvicorn diarlab.server:app`
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)
