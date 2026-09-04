from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_layer_conditioned_calendar_month_cmd_deepset_20260901_v37 as runner  # noqa: E402


def test_config_is_fixed_v13_single_change() -> None:
    config = runner.load_config()
    training = config["training"]
    cmd = training["central_moment_discrepancy"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert cmd["coefficient"] == 1.0
    assert cmd["orders"] == [1, 2, 3, 4, 5]
    assert not any(cmd[name] for name in ("kernel", "adversary", "ema", "sweep"))


def test_environment_encoding_is_layer_month() -> None:
    ids = runner.environment_ids(
        np.array([2, 3, 4]),
        pd.DatetimeIndex(
            [
                "2025-01-01T00:00:00+09:00",
                "2025-07-01T00:00:00+09:00",
                "2025-12-01T00:00:00+09:00",
            ]
        ),
    )
    assert ids.tolist() == [201, 307, 412]


def test_cmd_is_permutation_invariant_and_differentiable() -> None:
    torch.manual_seed(7)
    latent = torch.randn(18, 6, requires_grad=True)
    environment = torch.tensor(
        [201] * 3 + [202] * 3 + [203] * 3 + [301] * 3 + [302] * 3 + [303] * 3
    )
    value, receipt = runner.layer_conditioned_month_cmd(latent, environment)
    order = torch.randperm(18, generator=torch.Generator().manual_seed(8))
    permuted, second = runner.layer_conditioned_month_cmd(
        latent[order], environment[order]
    )
    value.backward()
    assert torch.allclose(value.detach(), permuted.detach(), atol=1e-7, rtol=0.0)
    assert receipt == second
    assert receipt["orders"] == [1, 2, 3, 4, 5]
    assert receipt["moment_term_count"] == 30
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


def test_forward_exposes_bounded_width32_latent_without_prediction_change() -> None:
    torch.manual_seed(9)
    model = runner.v12.VerticalDeepSet(8, 11, hidden=32).eval()
    tokens = torch.randn(7, 5, 8)
    mask = torch.ones(7, 5)
    context = torch.randn(7, 11)
    with torch.inference_mode():
        reference = model(tokens, mask, context)
        candidate, latent = runner.forward_with_latent(model, tokens, mask, context)
    assert torch.equal(reference, candidate)
    assert latent.shape == (7, 32)
    assert float(torch.tanh(latent).abs().max()) < 1.0


def test_masked_token_and_set_permutation_isolation() -> None:
    receipt = runner._isolation_receipt()
    assert max(receipt.values()) <= 1e-6


def test_semantic_audit_is_distinct_and_feedback_free() -> None:
    audit = runner.semantic_audit(runner.load_config())
    assert audit["repository_p2_exact_execution_hits"] == 0
    assert audit["v20_covariance_only_distinguished"]
    assert audit["v31_adversarial_distinguished"]
    assert audit["v36_gradient_variance_distinguished"]
    assert not audit["official_v23_feedback_used_for_selection"]


def test_two_preflights_are_byte_identical_and_zero_operation() -> None:
    first = runner.preflight()
    second = runner.preflight()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    for key in (
        "data_rows_read",
        "model_fits",
        "artifacts_written",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert first[key] == 0


def test_tiny_training_contract_is_finite() -> None:
    generator = np.random.default_rng(37)
    rows = 36
    runner._CURRENT_ENVIRONMENT_IDS = np.array(
        [201] * 6 + [202] * 6 + [203] * 6 + [301] * 6 + [302] * 6 + [303] * 6,
        dtype=np.int64,
    )
    config = runner.load_config()
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = rows
    output, receipt = runner.train_predict_seed(
        generator.normal(size=(rows, 5, 8)).astype(np.float32),
        np.ones((rows, 5), dtype=np.float32),
        generator.normal(size=(rows, 11)).astype(np.float32),
        generator.normal(size=rows).astype(np.float32),
        np.ones(rows, dtype=np.float32),
        generator.normal(size=(8, 5, 8)).astype(np.float32),
        np.ones((8, 5), dtype=np.float32),
        generator.normal(size=(8, 11)).astype(np.float32),
        config,
        37,
    )
    assert output.shape == (8,)
    assert np.isfinite(output).all()
    assert receipt["cmd_penalty_steps"] == receipt["optimizer_steps"] == 1
    assert receipt["moment_term_count_min"] == 30
