from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v39_causal_rdrop_consistency_crossquarter_addonly_20260901_v1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v39_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()


def test_symmetric_bernoulli_kl_zero_and_symmetric() -> None:
    first = torch.tensor([-2.0, 0.5, 2.0])
    second = torch.tensor([1.0, -0.5, -1.0])
    same = mod.symmetric_bernoulli_kl(first, first)
    forward = mod.symmetric_bernoulli_kl(first, second)
    reverse = mod.symmetric_bernoulli_kl(second, first)
    assert torch.max(torch.abs(same)) < 1e-7
    assert torch.all(forward > 0.0)
    assert torch.allclose(forward, reverse, atol=1e-7, rtol=1e-7)


def test_dropout_is_stochastic_only_during_training() -> None:
    torch.manual_seed(20260901)
    network = mod._RDropNetwork(4, 8, 0.1)
    values = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    network.train()
    assert not torch.equal(network(values), network(values))
    network.eval()
    assert torch.equal(network(values), network(values))


def test_consistency_has_finite_gradients() -> None:
    first = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    second = torch.tensor([1.0, 0.5, -1.0], requires_grad=True)
    loss = mod.symmetric_bernoulli_kl(first, second).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()


def test_frozen_crossquarter_contract() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    assert config["model"]["dropout_probability"] == 0.1
    assert config["model"]["consistency_coefficient"] == 1.0
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
