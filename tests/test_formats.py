"""Output writers: timestamps, SRT blocks, RTTM lines, JSON round trip."""

import json

from diarlab.formats import (
    segments_to_json,
    segments_to_srt,
    srt_timestamp,
    turns_to_rttm,
)
from diarlab.types import Segment, Turn, Word


def test_srt_timestamp_formatting():
    assert srt_timestamp(0) == "00:00:00,000"
    assert srt_timestamp(75.5) == "00:01:15,500"
    assert srt_timestamp(3661.001) == "01:01:01,001"
    assert srt_timestamp(-1) == "00:00:00,000"


def test_srt_blocks_numbered_with_speaker_prefix():
    segments = [
        Segment(0.0, 1.0, "SPEAKER_00", "hello there"),
        Segment(1.5, 2.0, None, "unattributed"),
    ]
    srt = segments_to_srt(segments)
    assert "1\n00:00:00,000 --> 00:00:01,000\nSPEAKER_00: hello there" in srt
    assert "2\n00:00:01,500 --> 00:00:02,000\nunattributed" in srt


def test_rttm_line_format():
    line = turns_to_rttm([Turn(0.0, 1.5, "A")], file_id="demo").strip()
    assert line == "SPEAKER demo 1 0.000 1.500 <NA> <NA> A <NA> <NA>"


def test_rttm_sorted_by_start():
    out = turns_to_rttm([Turn(5, 6, "B"), Turn(0, 1, "A")])
    first, second = out.strip().splitlines()
    assert " 0.000 " in first
    assert " 5.000 " in second


def test_json_round_trip():
    segments = [Segment(0.0, 0.5, "A", "hi", words=[Word(0.0, 0.5, "hi", 0.9)])]
    data = json.loads(segments_to_json(segments))
    seg = data["segments"][0]
    assert seg["speaker"] == "A"
    assert seg["words"][0]["text"] == "hi"
    assert seg["words"][0]["probability"] == 0.9
