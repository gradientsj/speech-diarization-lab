"""Output writers: JSON, SRT subtitles, and RTTM for diarization interchange."""

from __future__ import annotations

import json
from typing import Any

from .types import Segment, Turn


def segments_to_dict(segments: list[Segment]) -> dict[str, Any]:
    return {
        "segments": [
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "speaker": s.speaker,
                "text": s.text,
                "words": [
                    {
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "text": w.text,
                        "probability": round(w.probability, 3),
                    }
                    for w in s.words
                ],
            }
            for s in segments
        ]
    }


def segments_to_json(segments: list[Segment]) -> str:
    return json.dumps(segments_to_dict(segments), indent=2, ensure_ascii=False)


def srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp, e.g. 75.5 -> '00:01:15,500'."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments: list[Segment]) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        label = f"{seg.speaker}: " if seg.speaker else ""
        blocks.append(
            f"{i}\n{srt_timestamp(seg.start)} --> {srt_timestamp(seg.end)}\n{label}{seg.text}\n"
        )
    return "\n".join(blocks)


def turns_to_rttm(turns: list[Turn], file_id: str = "audio") -> str:
    """RTTM SPEAKER lines, the interchange format diarization scorers expect."""
    lines = [
        f"SPEAKER {file_id} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
        for t in sorted(turns, key=lambda t: t.start)
    ]
    return "\n".join(lines) + ("\n" if lines else "")
