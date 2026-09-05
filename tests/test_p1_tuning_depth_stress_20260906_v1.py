"""No model fits or source/official data; fixed two-feature stress contract."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_p1_tuning_depth_stress_20260906_v1 as m  # noqa: E402


def test_actual_source_lookup_and_encoder_differential():
    receipt = m.synthetic_audit()
    assert receipt["status"] == "PASS" and receipt["passed"] == receipt["total"]
    assert receipt["model_fits"] == receipt["source_observation_rows"] == receipt["official_rows"] == 0


def toy_bundle(frame):
    features = pd.DataFrame({"nominal_depth_m": np.float32(0), "depth_regime": pd.Series(["old"] * len(frame), dtype="string"),
                             "calendar": np.arange(len(frame)), "depth_raw": frame.depth})
    return m.old.FeatureBundle(features, tuple(features), ("depth_regime",))


def test_fallback_median_of_year_medians_then_round_no_zero_fill():
    frame = m.toy(n=4)
    frame.loc[3, "station"] = "UNSUPPORTED"
    stats = {"depth": {("S-ORS", 2023, 1): 4, ("S-ORS", 2024, 1): 20,
                       ("S-ORS", 2025, 1): 21, ("S-ORS", 2022, 1): np.nan}}
    bundle = toy_bundle(frame)
    altered = m.replace_metadata(bundle, frame, stats, m.CONDITIONS[2])
    np.testing.assert_array_equal(altered.frame.nominal_depth_m.to_numpy(), [20, 20, 20, np.nan])
    assert altered.frame.calendar.equals(bundle.frame.calendar)
    assert altered.frame.depth_raw.equals(bundle.frame.depth_raw)
    assert altered.frame.depth_regime.iloc[3] == "UNSUPPORTED|unknown|l1"


def test_differential_rejects_nonmetadata_change():
    frame = m.toy(n=4)
    bundle = toy_bundle(frame)
    encoder = m.old.TabularEncoder().fit(bundle, np.arange(4))
    altered = m.replace_metadata(bundle, frame, {"depth": {}}, m.CONDITIONS[1])
    altered.frame.loc[0, "calendar"] += 1
    with pytest.raises(ValueError, match="nonmetadata"):
        m.matrix_guard(encoder, bundle, altered)


def test_unregistered_rule_not_allowed_and_inputs_unchanged():
    frame = m.toy(n=4)
    bundle = toy_bundle(frame)
    before_frame, before_bundle = frame.copy(deep=True), bundle.frame.copy(deep=True)
    m.replace_metadata(bundle, frame, {"depth": {}}, m.CONDITIONS[1])
    assert frame.equals(before_frame) and bundle.frame.equals(before_bundle)
    with pytest.raises(ValueError, match="unregistered"):
        m.replace_metadata(bundle, frame, {"depth": {}}, "result_selected_fallback")


def test_normal_training_range_is_not_future_fp_or_fn_guarantee():
    training_normal = np.array([0., 1.])
    future_values = np.array([2., .5])
    future_labels = np.array([0, 1])
    predicted = (future_values < training_normal.min()) | (future_values > training_normal.max())
    assert int(((future_labels == 0) & predicted).sum()) == 1
    assert int(((future_labels == 1) & ~predicted).sum()) == 1


def test_no_model_training_official_or_csv_code():
    code = Path(m.__file__).read_text(encoding="utf-8")
    for forbidden in (' / "test.csv"', "sample_submission", ".to_csv(", "model.fit(", "_fit_model(", "select_inner("):
        assert forbidden not in code
