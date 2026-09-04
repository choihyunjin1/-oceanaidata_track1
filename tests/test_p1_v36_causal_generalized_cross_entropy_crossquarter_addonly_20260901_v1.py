from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v36_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()


def test_gce_formula_monotone_and_finite_gradient() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    loss = mod.generalized_cross_entropy(logits, torch.ones(3), 0.7)
    loss.sum().backward()
    assert torch.all(loss[:-1] > loss[1:])
    assert torch.isfinite(loss).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and torch.all(logits.grad < 0.0)


def test_gce_cross_entropy_limit() -> None:
    logits = torch.tensor([-1.5, 0.0, 1.5])
    targets = torch.ones(3)
    gce = mod.generalized_cross_entropy(logits, targets, 1e-6)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(gce, ce, atol=2e-4, rtol=2e-4)


def test_causal_features_reset_groups_and_ignore_future() -> None:
    rows = 50
    times = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    frame = pd.DataFrame(
        {
            "station": np.repeat(["A", "B"], rows),
            "layer": np.repeat([1, 2], rows),
            "_time": np.tile(times, 2),
            "temp": np.r_[np.arange(rows, dtype=float), 100.0 + np.arange(rows, dtype=float)],
        }
    )
    boundary = int(times[39].value)
    representation = {"lag_rows": [1, 6, 36]}
    original = mod.CAUSAL_FEATURES(frame, boundary, representation)
    changed_frame = frame.copy()
    future = mod.base._time_ns(frame["_time"]) > boundary
    changed_frame.loc[future, "temp"] += 10000.0
    changed = mod.CAUSAL_FEATURES(changed_frame, boundary, representation)
    assert original.shape == (100, 8) and np.isfinite(original).all()
    assert np.array_equal(original[~future], changed[~future])
    assert np.all(original[[0, rows], 1:4] == 0.0)


def test_fixed_objective_crossquarter_and_auditability_contract() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    assert config["model"]["gce_q"] == 0.7
    assert config["model"]["fits"] == 3 <= config["model"]["maximum_fits"] <= 9
    assert config["selection"]["candidate_threshold_fixed_before_q2"]
    assert config["selection"]["q2_q3_threshold_selection"] == config["selection"]["q2_q3_refits"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"]
    assert config["anchor"]["removals"] == 0
    assert config["auditability_amendment"]["preserve_all_pre_q2_threshold_q2_label_blind_actions"]


def test_gce_classifier_scores_are_finite() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))["model"]
    rng = np.random.default_rng(19)
    features = rng.normal(size=(200, 8)).astype(np.float32)
    labels = np.r_[np.zeros(150, dtype=np.int8), np.ones(50, dtype=np.int8)]
    model = mod.GCEClassifier(8, config, 20260901).fit(features, labels)
    scores = model.predict_score(features)
    assert scores.shape == (200,) and np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_real_preflight_is_zero_operation_and_lifecycle_aware() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    assert ready["status"] == "PASS_ZERO_OPERATION"
    assert ready["representation_support"]["gate"] == "PASS"
    assert all(ready["synthetic_guards"].values())
    assert ready["auditability"]["q2_target_reads"] == 0
    assert all(value == 0 for value in ready["counters"].values())
