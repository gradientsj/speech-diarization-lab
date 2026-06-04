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
