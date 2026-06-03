"""The benchmark: fetch LibriSpeech dev-clean, build mixtures, score systems.

Three subcommands, run as `python -m diarlab.bench <cmd>`:

- `fetch` downloads and extracts LibriSpeech dev-clean (337 MB, public, no
  credentials) under data/audio/.
- `build` constructs seeded synthetic conversations under data/mixtures/
  with a manifest carrying exact turns and transcripts.
- `run` transcribes and diarizes every mixture, scores WER/DER with the
  from-scratch metrics, and writes a JSON + markdown report under reports/.

Everything is deterministic given the seed except wall-clock timings (RTF),
which are measurements and reported as such.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import load_audio, save_audio
from .metrics import der, normalize_text, wer
from .mixtures import Utterance, build_mixture, interleave_speakers
from .types import Turn

DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
MIXTURE_DIR = DATA_DIR / "mixtures"
REPORT_DIR = Path("reports")
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def cmd_fetch(_: argparse.Namespace) -> int:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    archive = AUDIO_DIR / "dev-clean.tar.gz"
    extracted = AUDIO_DIR / "LibriSpeech" / "dev-clean"
    if extracted.exists():
        print(f"already extracted: {extracted}", file=sys.stderr)
        return 0
    if not archive.exists():
        print(f"downloading {DEV_CLEAN_URL} ...", file=sys.stderr)
        urllib.request.urlretrieve(DEV_CLEAN_URL, archive)  # noqa: S310 - fixed https URL
    print("extracting ...", file=sys.stderr)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(AUDIO_DIR, filter="data")
    print(f"done: {extracted}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _load_transcripts(chapter_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for trans in chapter_dir.glob("*.trans.txt"):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            out[utt_id] = text.strip()
    return out


def _speaker_utterances(
    speaker_dir: Path, max_seconds: float, limit: int
) -> list[tuple[Path, str, float]]:
    """Up to `limit` (flac, transcript, duration) tuples under max_seconds."""
    found: list[tuple[Path, str, float]] = []
    for chapter_dir in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
        transcripts = _load_transcripts(chapter_dir)
        for flac in sorted(chapter_dir.glob("*.flac")):
            duration = sf.info(str(flac)).duration
            if duration <= max_seconds and flac.stem in transcripts:
                found.append((flac, transcripts[flac.stem], duration))
            if len(found) >= limit:
                return found
    return found


def cmd_build(args: argparse.Namespace) -> int:
    corpus = AUDIO_DIR / "LibriSpeech" / "dev-clean"
    if not corpus.exists():
        print("corpus missing; run `python -m diarlab.bench fetch` first", file=sys.stderr)
        return 1
    MIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    speakers = sorted(p for p in corpus.iterdir() if p.is_dir())
    manifest: list[dict] = []
    speaker_counts = [2, 3, 4]
    for i in range(args.mixtures):
        n_speakers = speaker_counts[i % len(speaker_counts)]
        chosen = rng.choice(len(speakers), size=n_speakers, replace=False)
        per_speaker = max(2, args.utterances // n_speakers)
        by_speaker: dict[str, list[Utterance]] = {}
        for idx in chosen:
            speaker_dir = speakers[int(idx)]
            utts = []
            picked = _speaker_utterances(speaker_dir, args.max_utt_seconds, per_speaker)
            for flac, text, _ in picked:
                samples, rate = load_audio(flac, target_rate=SAMPLE_RATE)
                utts.append(Utterance(samples, rate, speaker_dir.name, text))
            if utts:
                by_speaker[speaker_dir.name] = utts
        ordered = interleave_speakers(by_speaker, seed=args.seed + i)
        mix = build_mixture(ordered, seed=args.seed + i)

        wav_name = f"mix_{i:03d}.wav"
        save_audio(MIXTURE_DIR / wav_name, mix.samples, mix.sample_rate)
        manifest.append(
            {
                "id": f"mix_{i:03d}",
                "wav": wav_name,
                "num_speakers": len(by_speaker),
                "duration": round(mix.duration, 3),
                "turns": [asdict(t) for t in mix.turns],
                "text": mix.reference_text(),
            }
        )
        print(
            f"built {wav_name}: {len(by_speaker)} speakers, "
            f"{len(mix.turns)} turns, {mix.duration:.1f}s",
            file=sys.stderr,
        )

    with open(MIXTURE_DIR / "manifest.jsonl", "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    total = sum(m["duration"] for m in manifest)
    print(f"manifest written: {len(manifest)} mixtures, {total/60:.1f} min audio", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _load_manifest() -> list[dict]:
    path = MIXTURE_DIR / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError("no manifest; run `python -m diarlab.bench build` first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _diarize_mixture(wav: Path, backend: str, num_speakers: int | None, device: str) -> list[Turn]:
    if backend == "pyannote":
        from .diarize import diarize_pyannote

        return diarize_pyannote(wav, num_speakers=num_speakers, device=device)
    from .diarize import ClusteredConfig, diarize_clustered

    audio, rate = load_audio(wav)
    return diarize_clustered(audio, rate, ClusteredConfig(num_speakers=num_speakers, device=device))


def cmd_run(args: argparse.Namespace) -> int:
    from .asr import transcribe

    manifest = _load_manifest()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    pooled = {"errors": 0, "ref_words": 0, "miss": 0.0, "fa": 0.0, "conf": 0.0, "ref_time": 0.0}
    for entry in manifest:
        wav = MIXTURE_DIR / entry["wav"]
        reference_turns = [Turn(**t) for t in entry["turns"]]
        num_speakers = entry["num_speakers"] if args.oracle_speaker_count else None

        result = transcribe(
            wav,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
        w = wer(normalize_text(entry["text"]), normalize_text(result.text))

        hyp_turns = _diarize_mixture(wav, args.backend, num_speakers, args.device)
        d = der(reference_turns, hyp_turns, collar=args.collar)

        pooled["errors"] += w.errors
        pooled["ref_words"] += w.reference_words
        pooled["miss"] += d.missed
        pooled["fa"] += d.false_alarm
        pooled["conf"] += d.confusion
        pooled["ref_time"] += d.total_reference

        rows.append(
            {
                "id": entry["id"],
                "duration": entry["duration"],
                "num_speakers_ref": entry["num_speakers"],
                "num_speakers_hyp": len({t.speaker for t in hyp_turns}),
                "wer": round(w.wer, 4),
                "der": round(d.der, 4),
                "rtf_asr": round(result.real_time_factor, 4),
            }
        )
        print(
            f"{entry['id']}: wer={w.wer:.3f} der={d.der:.3f} "
            f"spk {entry['num_speakers']}->{len({t.speaker for t in hyp_turns})}",
            file=sys.stderr,
        )

    overall_wer = pooled["errors"] / pooled["ref_words"] if pooled["ref_words"] else float("nan")
    overall_der = (
        (pooled["miss"] + pooled["fa"] + pooled["conf"]) / pooled["ref_time"]
        if pooled["ref_time"]
        else float("nan")
    )
    summary = {
        "config": {
            "model": args.model,
            "compute_type": args.compute_type,
            "device": args.device,
            "backend": args.backend,
            "collar": args.collar,
            "oracle_speaker_count": args.oracle_speaker_count,
            "mixtures": len(rows),
        },
        "overall": {
            "wer": round(overall_wer, 4),
            "der": round(overall_der, 4),
            "der_miss": round(pooled["miss"] / pooled["ref_time"], 4),
            "der_false_alarm": round(pooled["fa"] / pooled["ref_time"], 4),
            "der_confusion": round(pooled["conf"] / pooled["ref_time"], 4),
            "mean_rtf_asr": round(float(np.mean([r["rtf_asr"] for r in rows])), 4),
        },
        "per_mixture": rows,
    }

    tag = f"{args.backend}_{args.model}_{args.compute_type}_{args.device}"
    if args.oracle_speaker_count:
        tag += "_oracle"
    out_json = REPORT_DIR / f"benchmark_{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"wrote {out_json}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m diarlab.bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download LibriSpeech dev-clean")
    p_fetch.set_defaults(func=cmd_fetch)

    p_build = sub.add_parser("build", help="construct synthetic mixtures")
    p_build.add_argument("--mixtures", type=int, default=12)
    p_build.add_argument("--utterances", type=int, default=9, help="total utterances per mixture")
    p_build.add_argument("--max-utt-seconds", type=float, default=10.0)
    p_build.add_argument("--seed", type=int, default=0)
    p_build.set_defaults(func=cmd_build)

    p_run = sub.add_parser("run", help="score a configuration on the mixtures")
    p_run.add_argument("--model", default="small")
    p_run.add_argument("--compute-type", default="int8")
    p_run.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p_run.add_argument("--backend", default="clustered", choices=["clustered", "pyannote"])
    p_run.add_argument("--collar", type=float, default=0.25)
    p_run.add_argument(
        "--oracle-speaker-count",
        action="store_true",
        help="give the diarizer the true speaker count (upper-bound condition)",
    )
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
