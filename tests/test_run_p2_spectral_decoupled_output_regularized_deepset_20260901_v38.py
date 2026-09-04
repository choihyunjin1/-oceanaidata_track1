from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_spectral_decoupled_output_regularized_deepset_20260901_v38 as runner  # noqa: E402


def test_config_is_exact_v13_plus_one_fixed_output_penalty() -> None:
    config = runner.load_config()
    training = config["training"]
    sd = training["spectral_decoupling"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["optimizer"] == "exact_v13_AdamW"
    assert training["epochs"] == 60
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["maximum_fit_count"] == 9
    assert sd["coefficient"] == 0.01
    assert sd["target_independent"]
    assert not any(
        sd[name]
        for name in (
            "parameter_norm_constraint",
            "input_gradient",
            "environment_alignment",
            "sweep",
        )
    )


def test_penalty_formula_gradient_and_zero_noop() -> None:
    receipt = runner._penalty_contract_receipt()
    assert receipt["formula_exact"]
    assert receipt["gradient_exact"]
    assert receipt["zero_output_noop"]
    assert receipt["target_values_used"] == 0


def test_penalty_is_weighted_and_permutation_invariant() -> None:
    prediction = torch.tensor([1.0, -2.0, 0.5])
    weights = torch.tensor([3.0, 1.0, 2.0])
    value = runner.spectral_decoupling_penalty(prediction, weights)
    order = torch.tensor([2, 0, 1])
    permuted = runner.spectral_decoupling_penalty(
        prediction[order], weights[order]
    )
    assert torch.equal(value, permuted)


def test_masked_token_and_set_permutation_isolation() -> None:
    receipt = runner.v37._isolation_receipt()
    assert max(receipt.values()) <= 1e-6


def test_semantic_audit_discloses_cross_problem_adjacency() -> None:
    audit = runner.semantic_audit(runner.load_config())
    assert audit["repository_p2_exact_execution_hits"] == 0
    assert audit["p1_v46_cross_problem_adjacency_disclosed"]
    assert audit["v27_parameter_spectral_norm_distinguished"]
    assert audit["v37_latent_cmd_distinguished"]
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
    generator = np.random.default_rng(38)
    rows = 24
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
        38,
    )
    assert output.shape == (8,)
    assert np.isfinite(output).all()
    assert receipt["output_penalty_steps"] == receipt["optimizer_steps"] == 1
    assert receipt["coefficient"] == 0.01
