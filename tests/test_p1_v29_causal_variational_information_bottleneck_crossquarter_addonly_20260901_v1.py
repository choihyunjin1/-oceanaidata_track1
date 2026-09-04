from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v29_causal_variational_information_bottleneck_crossquarter_addonly_20260901_v1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v29_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()


def test_vib_loss_is_finite_and_differentiable() -> None:
    logits = torch.tensor([0.2, -0.4, 1.1], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0])
    mean = torch.zeros((3, 4), requires_grad=True)
    log_variance = torch.zeros((3, 4), requires_grad=True)
    loss = mod.vib_loss(
        logits,
        labels,
        mean,
        log_variance,
        positive_weight=3.0,
        kl_coefficient=0.001,
    ).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def _environment_fixture() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    times = pd.date_range("2024-01-01", periods=40, freq="10min", tz="UTC")
    metadata = pd.DataFrame(
        {
            "station": ["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10,
            "year": [2024] * 40,
            "layer": [1] * 10 + [2] * 10 + [1] * 10 + [2] * 10,
            "time": times.astype(str),
        }
    )
    return np.ones(40, dtype=bool), np.ones(40, dtype=np.int8), metadata


def test_environment_gate_requires_both_halves_and_identity_diversity() -> None:
    use, labels, metadata = _environment_fixture()
    contract = json.loads(mod.CONFIG.read_text(encoding="utf-8"))["selection"]["environment"]
    receipt = mod._environment_gate(use, labels, metadata, contract)
    assert receipt["passed"]
    collapsed = metadata.copy()
    collapsed["station"] = "A"
    collapsed["layer"] = 1
    assert not mod._environment_gate(use, labels, collapsed, contract)["passed"]


def test_fixed_numeric_threshold_and_budget_are_deterministic_add_only() -> None:
    scores = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    incumbent = np.zeros(1000, dtype=np.int8)
    incumbent[-1] = 1
    first = mod._fixed_threshold_additions(scores, incumbent, 0.8, 0.01)
    second = mod._fixed_threshold_additions(scores, incumbent, 0.8, 0.01)
    assert np.array_equal(first, second)
    assert first.sum() == 10
    assert not first[-1]


def test_causal_feature_group_reset_and_future_invariance() -> None:
    time = list(pd.date_range("2024-01-01", periods=8, freq="10min", tz="UTC")) * 2
    frame = pd.DataFrame(
        {
            "station": ["A"] * 8 + ["B"] * 8,
            "layer": [1] * 16,
            "time": [value.isoformat() for value in time],
            "_time": time,
            "temp": np.r_[np.arange(8.0), 100.0 + np.arange(8.0)],
        }
    )
    boundary = pd.Timestamp("2024-01-01T00:40:00Z").value
    representation = {"lag_rows": [1, 6, 36]}
    original = mod.shared.causal_evidential_features(frame, boundary, representation)
    perturbed = frame.copy()
    perturbed.loc[perturbed["_time"].map(pd.Timestamp).map(lambda value: value.value) > boundary, "temp"] += 10000.0
    changed = mod.shared.causal_evidential_features(perturbed, boundary, representation)
    prefix = np.asarray([pd.Timestamp(value).value <= boundary for value in frame["_time"]])
    assert np.array_equal(original[prefix], changed[prefix])
    assert np.all(original[[0, 8], 4:7] == 0.0)


def test_preregistered_crossquarter_lifecycle_and_fit_budget() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    selection = config["selection"]
    assert selection["candidate_threshold_fixed_before_q2"]
    assert selection["q2_q3_threshold_selection"] == 0
    assert selection["q2_q3_refits"] == 0
    assert selection["q4_open_only_after_q2_q3_pass"]
    assert config["model"]["fits"] == 3 <= config["model"]["maximum_fits"] <= 9
    assert config["anchor"]["removals"] == 0


def test_real_preflight_has_disjoint_windows_and_zero_operations() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    assert ready["status"] == "PASS_ZERO_OPERATION"
    assert ready["representation_support"]["gate"] == "PASS"
    assert all(value == 0 for value in ready["counters"].values())
    assert ready["parts"]["2025_q2"]["role"] == "transport_veto_1"
    assert ready["parts"]["2025_q3"]["role"] == "transport_veto_2"
    assert ready["parts"]["2025_q4"]["role"] == "single_unused_performance_window"
