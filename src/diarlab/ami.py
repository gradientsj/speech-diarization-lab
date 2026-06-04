"""AMI headset mix: a real-conversation benchmark with published baselines.

Everything else in this repo is scored on synthetic read-speech mixtures
and says so. AMI is the step that tests the pipeline against reality:
multi-party meetings with cross-talk, hesitations, and far-from-clean
turn-taking, hand-annotated with per-speaker segment times and words.

`python -m diarlab.bench fetch-ami` downloads the Mix-Headset wav for a
small set of meetings from the AMI test partition plus the public manual
annotations (CC BY 4.0, no credentials), parses the NXT XML into the same
manifest schema the synthetic mixtures use, and writes data/ami/. After
that, `bench run --set ami` scores it like any other set.

Two honesty notes, also in the README:

- The reference turns are human-annotated segment boundaries, which carry
  their own error bars; treat small DER differences accordingly.
- The reference text interleaves all speakers in time order. Where speech
  overlaps, no single word order is canonical, so WER on this set is an
  approximation; DER is the primary metric here.
"""

from __future__ import annotations

import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import soundfile as sf

MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ANNOTATIONS_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
)

# A small, fixed subset of the AMI test partition: 4-speaker scenario
# meetings from three different sites (Edinburgh, Idiap, TNO), so the
# acoustics vary. Kept small on purpose; this is a sanity check against
# reality, not a leaderboard run.
DEFAULT_MEETINGS = ("ES2004a", "IS1009b", "TS3003a")

SPEAKER_CHANNELS = ("A", "B", "C", "D")


def parse_segments(xml_data: bytes | str, speaker: str) -> list[dict]:
    """Per-speaker reference turns from a `<meeting>.<X>.segments.xml` file.

    Each <segment> carries transcriber_start/transcriber_end in seconds.
    Bytes are preferred: the AMI files declare ISO-8859-1 and ElementTree
    honors the declaration only when given bytes.
    """
    root = ET.fromstring(xml_data)
    turns = []
    for seg in root.iter("segment"):
        start = seg.get("transcriber_start")
        end = seg.get("transcriber_end")
        if start is None or end is None:
            continue
        s, e = float(start), float(end)
        if e > s:
            turns.append({"start": s, "end": e, "speaker": speaker})
    return turns


def parse_words(xml_data: bytes | str) -> list[tuple[float, str]]:
    """(start_time, word) pairs from a `<meeting>.<X>.words.xml` file.

    Only timed <w> elements count; punctuation marks (punc="true") and
    vocal sounds are not words the ASR should be charged for.
    """
    root = ET.fromstring(xml_data)
    words = []
    for el in root.iter("w"):
        if el.get("punc") == "true":
            continue
        start, text = el.get("starttime"), (el.text or "").strip()
        if start is None or not text:
            continue
        words.append((float(start), text))
    return words


def build_manifest_entry(
    meeting: str, annotations_dir: Path, wav_path: Path
) -> dict:
    """One manifest row in the same schema the synthetic mixtures use."""
    turns: list[dict] = []
    words: list[tuple[float, str]] = []
    speakers = []
    for channel in SPEAKER_CHANNELS:
        seg_file = annotations_dir / "segments" / f"{meeting}.{channel}.segments.xml"
        word_file = annotations_dir / "words" / f"{meeting}.{channel}.words.xml"
        if not seg_file.exists():
            continue
        speakers.append(channel)
        turns.extend(parse_segments(seg_file.read_bytes(), channel))
        if word_file.exists():
            words.extend(parse_words(word_file.read_bytes()))
    if not turns:
        raise FileNotFoundError(f"no segment annotations found for {meeting}")
    turns.sort(key=lambda t: (t["start"], t["end"]))
    words.sort(key=lambda w: w[0])
    info = sf.info(str(wav_path))
    return {
        "id": meeting,
        "wav": f"audio/{wav_path.name}",  # relative to data/ami/
        "num_speakers": len(speakers),
        "duration": round(info.duration, 3),
        "turns": turns,
        "text": " ".join(w for _, w in words),
    }


def fetch(ami_dir: Path, meetings: tuple[str, ...] = DEFAULT_MEETINGS) -> int:
    """Download wavs + annotations and write data/ami/manifest.jsonl."""
    audio_dir = ami_dir / "audio"
    annotations_dir = ami_dir / "annotations"
    audio_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    archive = ami_dir / "ami_public_manual.zip"
    if not (annotations_dir / "segments").exists():
        if not archive.exists():
            print(f"downloading {ANNOTATIONS_URL} ...", file=sys.stderr)
            urllib.request.urlretrieve(ANNOTATIONS_URL, archive)  # noqa: S310 - fixed https URL
        with zipfile.ZipFile(archive) as zf:
            members = [n for n in zf.namelist() if n.startswith(("segments/", "words/"))]
            zf.extractall(annotations_dir, members=members)

    manifest = []
    for meeting in meetings:
        wav = audio_dir / f"{meeting}.Mix-Headset.wav"
        if not wav.exists():
            url = f"{MIRROR}/{meeting}/audio/{meeting}.Mix-Headset.wav"
            print(f"downloading {url} ...", file=sys.stderr)
            urllib.request.urlretrieve(url, wav)  # noqa: S310 - fixed https URL
        entry = build_manifest_entry(meeting, annotations_dir, wav)
        manifest.append(entry)
        speech = sum(t["end"] - t["start"] for t in entry["turns"])
        print(
            f"{meeting}: {entry['duration']/60:.1f} min, {entry['num_speakers']} speakers, "
            f"{len(entry['turns'])} reference turns, {speech/60:.1f} min speech",
            file=sys.stderr,
        )

    with open(ami_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    print(f"manifest written: {len(manifest)} meetings", file=sys.stderr)
    return 0
