"""CLI surface: parsing only (model execution is covered by the benchmark)."""

import pytest

from diarlab.cli import build_parser


def test_attribute_defaults():
    args = build_parser().parse_args(["attribute", "x.wav"])
    assert args.model == "small"
    assert args.device == "cpu"
    assert args.compute_type == "int8"
    assert args.backend == "clustered"
    assert args.num_speakers is None


def test_diarize_backend_choices_enforced():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["diarize", "x.wav", "--backend", "nope"])


def test_command_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
