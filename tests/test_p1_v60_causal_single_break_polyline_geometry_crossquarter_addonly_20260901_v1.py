from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v60_causal_single_break_polyline_geometry_crossquarter_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v60_causal_single_break_polyline_geometry_crossquarter_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v60_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()
config = json.loads(CONFIG.read_text(encoding="utf-8"))


def test_single_break_geometry_separates_line_kink_and_step() -> None:
    rows = config["representation"]["trajectory_rows"]
    tolerance = config["representation"]["split_tolerance_prefix_sigma"]
    x = np.arange(rows, dtype=np.float64)
    straight = (0.05 * x)[None, :]
    kink = np.where(
        x < rows // 2,
        0.03 * x,
        0.03 * (rows // 2) + 0.24 * (x - rows // 2),
    )[None, :]
    step = np.where(x < rows // 2, 0.0, 2.5)[None, :]
    straight_stats = mod.single_break_polyline_statistics(straight, tolerance)[0]
    kink_stats = mod.single_break_polyline_statistics(kink, tolerance)[0]
    step_stats = mod.single_break_polyline_statistics(step, tolerance)[0]
    assert straight_stats[1] < 1e-10 and straight_stats[-1] == 0.0
    assert kink_stats[1] > tolerance and abs(kink_stats[6]) > 0.05
    assert step_stats[1] > tolerance and step_stats[-1] == 1.0
    assert 0.0 < kink_stats[3] < 1.0 and 0.0 < step_stats[3] < 1.0


def test_feature_future_invariance_group_and_gap_reset() -> None:
    rows = 360
    window = config["representation"]["trajectory_rows"]
    times = pd.date_range("2024-02-01", periods=rows, freq="10min", tz="UTC")
    signal = np.sin(np.arange(rows, dtype=np.float64) / 9.0)
    frame = pd.DataFrame(
        {
            "station": np.repeat(["S-A", "S-B"], rows),
            "layer": np.repeat(["L1", "L2"], rows),
            "_time": np.tile(times, 2),
            "temp": np.tile(signal, 2),
        }
    )
    boundary = int(times[199].value)
    first = mod.causal_polyline_features(frame, boundary, config["representation"])
    changed = frame.copy()
    future = mod.base._time_ns(changed["_time"]) > boundary
    changed.loc[future, "temp"] += 1000.0
    second = mod.causal_polyline_features(changed, boundary, config["representation"])
    assert np.array_equal(first[~future], second[~future])
    assert np.array_equal(first[:rows], first[rows:])

    gap_frame = frame.iloc[:rows].copy()
    gap_frame.loc[210:, "_time"] += pd.Timedelta(minutes=10)
    gap = mod.causal_polyline_features(gap_frame, boundary, config["representation"])
    assert np.all(gap[210 : 210 + window - 1, -1] == 0.0)
    assert first.shape == (2 * rows, 11) and np.isfinite(first).all()


def test_negative_fingerprint_and_scratch_provenance_are_sealed() -> None:
    audit = config["semantic_audit"]
    assert audit["decision"] == "NOVEL_P1_OBJECTIVE_PROCEED_ONCE"
    assert not audit["exact_duplicate"] and not audit["semantic_duplicate"]
    assert len(audit["negative_fingerprint"]) == 6
    assert config["policy_binding"]["distributed_data_only"]
    assert config["policy_binding"]["pretrained_weights"] == 0
    assert config["policy_binding"]["non_distributed_iors_lineages"] == 0
    assert config["model"]["pretrained_weights"] == 0
    assert config["source"]["allowed_files"] == ["README.md", "train.csv"]
    assert config["source"]["external_lineage_inputs"] == []


def test_add_only_crossquarter_and_exactly_once_contract() -> None:
    assert config["model"]["maximum_fits"] == 9
    assert config["selection"]["q2_q3_refits"] == 0
    assert config["selection"]["q2_q3_threshold_selection"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"]
    assert config["anchor"]["removals"] == 0
    assert config["operations"]["exactly_once"]
    protected = [
        "official",
        "test",
        "sample_submission",
        "submission",
        "hidden",
        "csv",
        "uploads",
    ]
    assert all(config["operations"][key] == 0 for key in protected)


def test_synthetic_guards() -> None:
    assert all(mod._synthetic_guards(config["representation"]).values())


def test_source_lineage_audit_is_distributed_only() -> None:
    data = Path(os.environ["P1_DATA_DIR"])
    audit = mod._source_lineage_audit(data, config)
    assert audit["gate"] == "PASS"
    assert all(audit["checks"].values())
    assert all(audit["dependency_checks"].values())
    assert audit["external_lineages"] == 0
    assert audit["non_distributed_iors_lineages"] == 0
    assert audit["pretrained_weights"] == 0


def test_real_preflight_is_ready_zero_operation_and_support_qualified() -> None:
    data = Path(os.environ["P1_DATA_DIR"])
    if mod.ARTIFACT.exists() or mod.LOCK.exists():
        try:
            mod.preflight(data)
        except FileExistsError:
            pass
        else:
            raise AssertionError("a consumed namespace must fail closed")
        ready = json.loads((mod.ARTIFACT / "preflight.json").read_text(encoding="utf-8"))
    else:
        ready = mod.preflight(data)
    assert ready["status"] == "READY_ZERO_OPERATION" and ready["ready"]
    assert ready["runner_sha256"] == mod.base._sha(RUNNER)
    assert ready["support_qualification"]["gate"] == "PASS"
    assert all(ready["support_qualification"]["checks"].values())
    assert ready["source_lineage_audit"]["gate"] == "PASS"
    assert all(value == 0 for value in ready["counters"].values())
    assert ready["policy_binding"]["external_lineages"] == 0
    assert ready["policy_binding"]["non_distributed_iors_lineages"] == 0
