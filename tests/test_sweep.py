"""Grid selection logic for the calibration sweep (pure, no models)."""

import math

from diarlab.bench import pick_value, pooled_component_at, pooled_der_at


def _row(mix_id, cells):
    return {"id": mix_id, "num_speakers_ref": 2, "by_value": cells}


def _cell(miss, fa, conf, ref):
    return {"miss": miss, "false_alarm": fa, "confusion": conf, "ref_time": ref}


def test_pooled_der_pools_components_not_ratios():
    # one long and one short mixture: pooling must weight by reference time,
    # not average the per-mixture DERs
    rows = [
        _row("a", {"0.5": _cell(miss=10.0, fa=0.0, conf=0.0, ref=100.0)}),
        _row("b", {"0.5": _cell(miss=0.0, fa=0.0, conf=1.0, ref=10.0)}),
    ]
    assert pooled_der_at(rows, "0.5") == (10.0 + 1.0) / 110.0


def test_pooled_component_isolates_one_component():
    rows = [
        _row("a", {"0.5": _cell(miss=10.0, fa=4.0, conf=2.0, ref=100.0)}),
        _row("b", {"0.5": _cell(miss=5.0, fa=1.0, conf=0.0, ref=100.0)}),
    ]
    assert pooled_component_at(rows, "0.5", "miss") == 15.0 / 200.0
    assert pooled_component_at(rows, "0.5", "false_alarm") == 5.0 / 200.0


def test_pooled_der_empty_reference_is_nan():
    rows = [_row("a", {"0.5": _cell(0.0, 0.0, 0.0, 0.0)})]
    assert math.isnan(pooled_der_at(rows, "0.5"))


def test_pick_value_takes_minimum():
    rows = [
        _row(
            "a",
            {
                "0.4": _cell(5.0, 0.0, 3.0, 100.0),
                "0.5": _cell(5.0, 0.0, 1.0, 100.0),
                "0.6": _cell(5.0, 0.0, 2.0, 100.0),
            },
        )
    ]
    assert pick_value(rows, ["0.4", "0.5", "0.6"]) == "0.5"


def test_pick_value_tie_goes_to_lower_value():
    rows = [
        _row(
            "a",
            {
                "0.5": _cell(5.0, 0.0, 1.0, 100.0),
                "0.6": _cell(5.0, 0.0, 1.0, 100.0),
            },
        )
    ]
    assert pick_value(rows, ["0.5", "0.6"]) == "0.5"
