"""Command-line interface.

Three subcommands, one per stage of interest:

- `diarlab transcribe audio.wav` -- ASR only, with word timestamps.
- `diarlab diarize audio.wav` -- who spoke when, no transcription.
- `diarlab attribute audio.wav` -- the full pipeline: transcribe, diarize,
  align, and emit speaker-attributed JSON and/or SRT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .align import assign_words, group_segments
from .formats import segments_to_json, segments_to_srt, turns_to_rttm
from .types import Segment


def _add_asr_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", default="small", help="Whisper size (tiny/base/small/medium/large-v3)"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--compute-type", default="int8", help="ctranslate2 compute type (int8, float16, ...)"
    )
    parser.add_argument(
        "--language", default=None, help="force a language code instead of detecting"
    )


def _add_diarize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="clustered", choices=["clustered", "pyannote"])
    parser.add_argument("--num-speakers", type=int, default=None, help="fix the speaker count")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.6,
        help="clustered backend: cosine distance cut for the dendrogram",
    )


def _write_or_print(text: str, path: str | None) -> None:
    if path:
        Path(path).write_text(text, encoding="utf-8")
        print(f"wrote {path}", file=sys.stderr)
    else:
        print(text)


def _diarize(path: str, args: argparse.Namespace):
    if args.backend == "pyannote":
        from .diarize import diarize_pyannote

        return diarize_pyannote(path, num_speakers=args.num_speakers, device=args.device)
    from .audio import load_audio
    from .diarize import ClusteredConfig, diarize_clustered

    audio, rate = load_audio(path)
    config = ClusteredConfig(
        num_speakers=args.num_speakers,
        distance_threshold=args.distance_threshold,
        device=args.device,
    )
    return diarize_clustered(audio, rate, config)


def cmd_transcribe(args: argparse.Namespace) -> int:
    from .asr import transcribe

    result = transcribe(
        args.audio,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )
    segments = group_segments([(w, None) for w in result.words])
    print(
        f"language={result.language} duration={result.audio_duration:.1f}s "
        f"rtf={result.real_time_factor:.3f}",
        file=sys.stderr,
    )
    if args.srt:
        _write_or_print(segments_to_srt(segments), args.srt)
    if args.json or not args.srt:
        _write_or_print(segments_to_json(segments), args.json)
    return 0


def cmd_diarize(args: argparse.Namespace) -> int:
    turns = _diarize(args.audio, args)
    _write_or_print(turns_to_rttm(turns, file_id=Path(args.audio).stem), args.rttm)
    return 0


def cmd_attribute(args: argparse.Namespace) -> int:
    from .asr import transcribe

    result = transcribe(
        args.audio,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )
    turns = _diarize(args.audio, args)
    segments: list[Segment] = group_segments(assign_words(result.words, turns))
    print(
        f"language={result.language} duration={result.audio_duration:.1f}s "
        f"speakers={len({t.speaker for t in turns})} rtf={result.real_time_factor:.3f}",
        file=sys.stderr,
    )
    if args.srt:
        _write_or_print(segments_to_srt(segments), args.srt)
    if args.json or not args.srt:
        _write_or_print(segments_to_json(segments), args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diarlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("transcribe", help="speech to text with word timestamps")
    p_tr.add_argument("audio")
    _add_asr_args(p_tr)
    p_tr.add_argument("--json", default=None, help="write JSON here (default: stdout)")
    p_tr.add_argument("--srt", default=None, help="write SRT subtitles here")
    p_tr.set_defaults(func=cmd_transcribe)

    p_di = sub.add_parser("diarize", help="who spoke when (RTTM output)")
    p_di.add_argument("audio")
    p_di.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    _add_diarize_args(p_di)
    p_di.add_argument("--rttm", default=None, help="write RTTM here (default: stdout)")
    p_di.set_defaults(func=cmd_diarize)

    p_at = sub.add_parser("attribute", help="transcribe + diarize + align")
    p_at.add_argument("audio")
    _add_asr_args(p_at)
    _add_diarize_args(p_at)
    p_at.add_argument("--json", default=None, help="write JSON here (default: stdout)")
    p_at.add_argument("--srt", default=None, help="write SRT subtitles here")
    p_at.set_defaults(func=cmd_attribute)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
