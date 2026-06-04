"""The benchmark: fetch LibriSpeech dev-clean, build mixtures, score systems.

Three subcommands, run as `python -m diarlab.bench <cmd>`:

- `fetch` downloads and extracts LibriSpeech dev-clean (337 MB, public, no
  credentials) under data/audio/.
- `build` constructs seeded synthetic conversations under data/mixtures/
  with a manifest carrying exact turns and transcripts.
- `run` transcribes and diarizes every mixture, scores WER/DER with the
  from-scratch metrics, and writes a JSON + markdown report under reports/.
- `sweep` calibrates the clustering distance threshold: embeddings are
  computed once per mixture, clustering is rerun across a threshold grid,
  the threshold is chosen on half the mixtures and scored on the other
  half, and the full grid lands in reports/threshold_sweep.json.

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
from .diarize import ClusteredConfig
from .metrics import der, normalize_text, wer
from .mixtures import Utterance, build_mixture, interleave_speakers
from .types import Turn

DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
MIXTURE_DIR = DATA_DIR / "mixtures"
OVERLAP_MIXTURE_DIR = DATA_DIR / "mixtures_overlap"
REPORT_DIR = Path("reports")
SAMPLE_RATE = 16_000


def _mixture_dir(overlap: bool) -> Path:
    return OVERLAP_MIXTURE_DIR if overlap else MIXTURE_DIR


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
    out_dir = _mixture_dir(args.overlap_prob > 0)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        mix = build_mixture(ordered, seed=args.seed + i, overlap_prob=args.overlap_prob)

        # overlapped time between consecutive placements, for the manifest
        overlapped = sum(
            max(0.0, mix.turns[j - 1].end - mix.turns[j].start)
            for j in range(1, len(mix.turns))
        )
        wav_name = f"mix_{i:03d}.wav"
        save_audio(out_dir / wav_name, mix.samples, mix.sample_rate)
        manifest.append(
            {
                "id": f"mix_{i:03d}",
                "wav": wav_name,
                "num_speakers": len(by_speaker),
                "duration": round(mix.duration, 3),
                "overlap_prob": args.overlap_prob,
                "overlapped_seconds": round(overlapped, 3),
                "turns": [asdict(t) for t in mix.turns],
                "text": mix.reference_text(),
            }
        )
        print(
            f"built {wav_name}: {len(by_speaker)} speakers, {len(mix.turns)} turns, "
            f"{mix.duration:.1f}s, {overlapped:.1f}s overlapped",
            file=sys.stderr,
        )

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    total = sum(m["duration"] for m in manifest)
    print(f"manifest written: {len(manifest)} mixtures, {total/60:.1f} min audio", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _load_manifest(mixture_dir: Path) -> list[dict]:
    path = mixture_dir / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no manifest in {mixture_dir}; run `python -m diarlab.bench build` first"
            " (with --overlap-prob for the overlap set)"
        )
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

    mixture_dir = _mixture_dir(args.overlap)
    manifest = _load_manifest(mixture_dir)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    pooled = {"errors": 0, "ref_words": 0, "miss": 0.0, "fa": 0.0, "conf": 0.0, "ref_time": 0.0}
    for entry in manifest:
        wav = mixture_dir / entry["wav"]
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
            "distance_threshold": ClusteredConfig.distance_threshold
            if args.backend == "clustered"
            else None,
            "vad_pad": ClusteredConfig.vad_pad if args.backend == "clustered" else None,
            "overlap": args.overlap,
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
    if args.overlap:
        tag = "overlap_" + tag
    out_json = REPORT_DIR / f"benchmark_{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"wrote {out_json}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


# Default grid per sweepable parameter: (lo, hi, step).
SWEEP_GRIDS = {
    "threshold": (0.30, 0.90, 0.05),
    "pad": (0.00, 0.40, 0.05),
}


def pooled_component_at(rows: list[dict], value: str, component: str) -> float:
    """Pool one DER component (as a rate) across mixtures at one grid value."""
    total = ref = 0.0
    for row in rows:
        cell = row["by_value"][value]
        total += cell[component]
        ref += cell["ref_time"]
    return total / ref if ref else float("nan")


def pooled_der_at(rows: list[dict], value: str) -> float:
    """Pool DER across mixtures at one grid value."""
    miss = fa = conf = ref = 0.0
    for row in rows:
        cell = row["by_value"][value]
        miss += cell["miss"]
        fa += cell["false_alarm"]
        conf += cell["confusion"]
        ref += cell["ref_time"]
    return (miss + fa + conf) / ref if ref else float("nan")


def pick_value(rows: list[dict], values: list[str]) -> str:
    """The grid value with the lowest pooled DER (lowest value on ties)."""
    return min(values, key=lambda v: (pooled_der_at(rows, v), float(v)))


def cmd_sweep(args: argparse.Namespace) -> int:
    from .cluster import cluster_embeddings
    from .embeddings import EcapaEmbedder
    from .vad import detect_speech
    from .windows import merge_regions, slice_windows, windows_to_turns

    manifest = _load_manifest(MIXTURE_DIR)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lo, hi, step = SWEEP_GRIDS[args.param]
    lo, hi, step = (
        args.lo if args.lo is not None else lo,
        args.hi if args.hi is not None else hi,
        args.step if args.step is not None else step,
    )
    values = [f"{v:.2f}" for v in np.arange(lo, hi + step / 2, step)]
    cfg = ClusteredConfig(device=args.device)
    embedder = EcapaEmbedder(device=args.device)

    def score(reference_turns, windows, labels):
        hyp_turns = windows_to_turns(windows, labels)
        d = der(reference_turns, hyp_turns, collar=args.collar)
        return {
            "der": round(d.der, 4),
            "miss": d.missed,
            "false_alarm": d.false_alarm,
            "confusion": d.confusion,
            "ref_time": d.total_reference,
            "num_speakers_hyp": len({x.speaker for x in hyp_turns}),
        }

    rows: list[dict] = []
    for entry in manifest:
        audio, rate = load_audio(MIXTURE_DIR / entry["wav"])
        reference_turns = [Turn(**t) for t in entry["turns"]]
        raw = detect_speech(audio, rate, threshold=cfg.vad_threshold)
        row = {"id": entry["id"], "num_speakers_ref": entry["num_speakers"], "by_value": {}}

        if args.param == "threshold":
            # the threshold only touches clustering, so embed once
            regions = merge_regions(
                raw, min_gap=cfg.min_gap, min_duration=cfg.min_duration, pad=cfg.vad_pad
            )
            windows = slice_windows(regions, window=cfg.window, stride=cfg.stride)
            embeddings = embedder.embed_windows(audio, rate, windows)
            for v in values:
                labels = cluster_embeddings(embeddings, distance_threshold=float(v))
                row["by_value"][v] = score(reference_turns, windows, labels)
        else:
            # the pad changes the regions, so re-embed per value
            for v in values:
                regions = merge_regions(
                    raw, min_gap=cfg.min_gap, min_duration=cfg.min_duration, pad=float(v)
                )
                windows = slice_windows(regions, window=cfg.window, stride=cfg.stride)
                embeddings = embedder.embed_windows(audio, rate, windows)
                labels = cluster_embeddings(
                    embeddings, distance_threshold=cfg.distance_threshold
                )
                row["by_value"][v] = score(reference_turns, windows, labels)

        rows.append(row)
        print(f"{entry['id']}: swept {args.param} over {len(values)} values", file=sys.stderr)

    # Even-index mixtures calibrate, odd-index mixtures evaluate. The
    # speaker counts cycle 2/3/4 by index, so both halves stay balanced.
    calibration = rows[0::2]
    heldout = rows[1::2]
    chosen = pick_value(calibration, values)
    defaults = {"threshold": cfg.distance_threshold, "pad": cfg.vad_pad}
    default_key = f"{defaults[args.param]:.2f}"

    grid = {}
    for v in values:
        count_ok = sum(
            1 for r in rows if r["by_value"][v]["num_speakers_hyp"] == r["num_speakers_ref"]
        )
        grid[v] = {
            "calibration_der": round(pooled_der_at(calibration, v), 4),
            "heldout_der": round(pooled_der_at(heldout, v), 4),
            "all_der": round(pooled_der_at(rows, v), 4),
            "all_miss": round(pooled_component_at(rows, v, "miss"), 4),
            "all_false_alarm": round(pooled_component_at(rows, v, "false_alarm"), 4),
            "all_confusion": round(pooled_component_at(rows, v, "confusion"), 4),
            "speaker_count_correct": f"{count_ok}/{len(rows)}",
        }

    summary = {
        "config": {
            "param": args.param,
            "device": args.device,
            "collar": args.collar,
            "grid": [float(v) for v in values],
            "calibration_mixtures": [r["id"] for r in calibration],
            "heldout_mixtures": [r["id"] for r in heldout],
        },
        "chosen_value": float(chosen),
        "chosen_heldout_der": grid[chosen]["heldout_der"],
        "default_value": defaults[args.param],
        "default_heldout_der": grid[default_key]["heldout_der"] if default_key in grid else None,
        "grid": grid,
        "per_mixture": rows,
    }
    out_json = REPORT_DIR / f"{args.param}_sweep.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("grid", "per_mixture")}))
    for v in values:
        print(
            f"  {v}: calib={grid[v]['calibration_der']:.4f} heldout={grid[v]['heldout_der']:.4f} "
            f"all={grid[v]['all_der']:.4f} miss={grid[v]['all_miss']:.4f} "
            f"fa={grid[v]['all_false_alarm']:.4f} conf={grid[v]['all_confusion']:.4f} "
            f"count={grid[v]['speaker_count_correct']}"
        )
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
    p_build.add_argument(
        "--overlap-prob",
        type=float,
        default=0.0,
        help="probability a speaker change partially overlaps; >0 writes data/mixtures_overlap/",
    )
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
    p_run.add_argument(
        "--overlap",
        action="store_true",
        help="score the overlapped mixture set (data/mixtures_overlap/)",
    )
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser("sweep", help="calibrate one pipeline parameter on held-out mixtures")
    p_sweep.add_argument("--param", default="threshold", choices=sorted(SWEEP_GRIDS))
    p_sweep.add_argument("--lo", type=float, default=None, help="grid start (per-param default)")
    p_sweep.add_argument("--hi", type=float, default=None, help="grid end (per-param default)")
    p_sweep.add_argument("--step", type=float, default=None, help="grid step (per-param default)")
    p_sweep.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p_sweep.add_argument("--collar", type=float, default=0.25)
    p_sweep.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
