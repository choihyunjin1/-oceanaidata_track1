from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v38_causal_focal_loss_crossquarter_addonly_20260901_v1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v38_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()


def test_gamma_zero_is_binary_cross_entropy() -> None:
    logits = torch.tensor([-3.0, -0.5, 0.5, 3.0])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    focal = mod.focal_loss(logits, labels, 0.0)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    assert torch.allclose(focal, expected, atol=1e-6, rtol=1e-6)


def test_gamma_two_suppresses_easy_not_hard_examples() -> None:
    logits = torch.tensor([-4.0, 4.0])
    labels = torch.ones(2)
    focal = mod.focal_loss(logits, labels, 2.0)
    baseline = mod.focal_loss(logits, labels, 0.0)
    assert focal[0] / baseline[0] > 0.9
    assert focal[1] / baseline[1] < 0.01


def test_focal_gradient_is_finite_and_target_directed() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    loss = mod.focal_loss(logits, torch.ones(3), 2.0).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad < 0.0)


def test_frozen_crossquarter_contract() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    assert config["model"]["focal_gamma"] == 2.0
    assert config["model"]["maximum_fits"] == 9
    assert config["selection"]["q2_q3_refits"] == config["selection"]["q2_q3_threshold_selection"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"]
    assert config["anchor"]["removals"] == 0
    assert config["operations"]["official"] == config["operations"]["csv"] == config["operations"]["uploads"] == 0


def test_real_preflight_is_zero_operation_and_wrapper_identified() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    assert ready["runner_sha256"] == mod.base._sha(RUNNER)
    assert all(ready["synthetic_guards"].values())
    assert all(value == 0 for value in ready["counters"].values())
    assert np.max(ready["representation_support"]["feature_variances"]) > 0.0
