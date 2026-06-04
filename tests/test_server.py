"""The HTTP surface, exercised with an injected pipeline (no models)."""

import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from diarlab.server import create_app  # noqa: E402
from diarlab.types import Segment, Word  # noqa: E402


def _stub_segments(path):
    return [
        Segment(
            start=0.0,
            end=1.0,
            speaker="SPEAKER_00",
            text="hello world",
            words=[Word(0.0, 0.5, "hello"), Word(0.5, 1.0, "world")],
        )
    ]


def _failing_pipeline(path):
    raise ValueError("corrupt audio")


def _wait_done(client, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


@pytest.fixture
def client():
    return TestClient(create_app(attribute_fn=_stub_segments))


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_submit_poll_result(client):
    resp = client.post("/jobs", files={"file": ("a.wav", b"RIFFfake", "audio/wav")})
    assert resp.status_code == 202
    job_id = resp.json()["id"]

    body = _wait_done(client, job_id)
    assert body["status"] == "done"
    seg = body["result"]["segments"][0]
    assert seg["speaker"] == "SPEAKER_00"
    assert seg["words"][0]["text"] == "hello"


def test_srt_after_done(client):
    job_id = client.post("/jobs", files={"file": ("a.wav", b"x", "audio/wav")}).json()["id"]
    _wait_done(client, job_id)
    srt = client.get(f"/jobs/{job_id}/srt")
    assert srt.status_code == 200
    assert "SPEAKER_00: hello world" in srt.text
    assert "00:00:00,000 --> 00:00:01,000" in srt.text


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/srt").status_code == 404


def test_unsupported_suffix_is_415(client):
    resp = client.post("/jobs", files={"file": ("a.txt", b"hello", "text/plain")})
    assert resp.status_code == 415


def test_empty_upload_is_400(client):
    resp = client.post("/jobs", files={"file": ("a.wav", b"", "audio/wav")})
    assert resp.status_code == 400


def test_pipeline_failure_lands_on_the_job():
    client = TestClient(create_app(attribute_fn=_failing_pipeline))
    job_id = client.post("/jobs", files={"file": ("a.wav", b"x", "audio/wav")}).json()["id"]
    body = _wait_done(client, job_id)
    assert body["status"] == "error"
    assert "corrupt audio" in body["error"]
    # the result endpoints refuse until there is a result
    assert client.get(f"/jobs/{job_id}/srt").status_code == 409


def _stub_stream(path):
    """Two segments produced with a gap, the streaming shape of the pipeline."""
    yield Segment(0.0, 1.0, "SPEAKER_00", "hello", words=[Word(0.0, 1.0, "hello")])
    time.sleep(0.05)
    yield Segment(1.5, 2.0, "SPEAKER_01", "world", words=[Word(1.5, 2.0, "world")])


def _collect_stream(ws):
    messages = []
    while True:
        msg = ws.receive_json()
        messages.append(msg)
        if msg["type"] in ("status", "error"):
            return messages


def test_websocket_streams_segments_then_status():
    client = TestClient(create_app(stream_fn=_stub_stream))
    job_id = client.post("/jobs", files={"file": ("a.wav", b"x", "audio/wav")}).json()["id"]
    with client.websocket_connect(f"/jobs/{job_id}/stream") as ws:
        messages = _collect_stream(ws)
    kinds = [m["type"] for m in messages]
    assert kinds == ["segment", "segment", "status"]
    assert messages[0]["segment"]["speaker"] == "SPEAKER_00"
    assert messages[1]["segment"]["text"] == "world"
    assert messages[-1]["status"] == "done"


def test_websocket_replays_after_completion():
    client = TestClient(create_app(stream_fn=_stub_stream))
    job_id = client.post("/jobs", files={"file": ("a.wav", b"x", "audio/wav")}).json()["id"]
    _wait_done(client, job_id)  # connect only after the job finished
    with client.websocket_connect(f"/jobs/{job_id}/stream") as ws:
        messages = _collect_stream(ws)
    assert [m["type"] for m in messages] == ["segment", "segment", "status"]


def test_websocket_unknown_job():
    client = TestClient(create_app(attribute_fn=_stub_segments))
    with client.websocket_connect("/jobs/nope/stream") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error"


def test_streamed_segments_also_land_in_the_rest_result():
    client = TestClient(create_app(stream_fn=_stub_stream))
    job_id = client.post("/jobs", files={"file": ("a.wav", b"x", "audio/wav")}).json()["id"]
    body = _wait_done(client, job_id)
    assert [s["text"] for s in body["result"]["segments"]] == ["hello", "world"]


class _StubLiveSession:
    """Counts fed samples; emits one segment per feed call, one on flush."""

    def __init__(self):
        self.fed = 0
        self.calls = 0

    def feed(self, samples):
        self.fed += len(samples)
        self.calls += 1
        return [
            Segment(0.0, 1.0, f"SPEAKER_{self.calls - 1:02d}", f"feed{self.calls - 1}", words=[])
        ]

    def flush(self):
        return [Segment(9.0, 9.5, "SPEAKER_00", "tail", words=[])]


def _live_client(sessions):
    def factory():
        sessions.append(_StubLiveSession())
        return sessions[-1]

    return TestClient(create_app(attribute_fn=_stub_segments, live_session_factory=factory))


def test_live_websocket_round_trip():
    import numpy as np

    sessions = []
    client = _live_client(sessions)
    with client.websocket_connect("/live/ws") as ws:
        ws.send_json({"sample_rate": 16_000})
        assert ws.receive_json() == {"type": "ready"}
        ws.send_bytes(np.zeros(16_000, dtype=np.int16).tobytes())
        msg = ws.receive_json()
        assert msg["type"] == "segment"
        assert msg["segment"]["text"] == "feed0"
        ws.send_json({"type": "stop"})
        assert ws.receive_json()["segment"]["text"] == "tail"
        assert ws.receive_json() == {"type": "status", "status": "done"}
    assert sessions[0].fed == 16_000  # 1s at 16 kHz arrives unresampled


def test_live_websocket_resamples_to_16k():
    import numpy as np

    sessions = []
    client = _live_client(sessions)
    with client.websocket_connect("/live/ws") as ws:
        ws.send_json({"sample_rate": 48_000})
        ws.receive_json()
        ws.send_bytes(np.zeros(48_000, dtype=np.int16).tobytes())
        ws.receive_json()
        ws.send_json({"type": "stop"})
    assert sessions[0].fed == 16_000  # 1s at 48 kHz lands as 1s at 16 kHz


def test_live_page_served():
    client = TestClient(create_app(attribute_fn=_stub_segments))
    resp = client.get("/live")
    assert resp.status_code == 200
    assert "diarlab live" in resp.text
