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

import run_p2_predictive_dropout_consistency_deepset_20260901_v40 as runner  # noqa: E402


def test_config_is_exact_v13_plus_one_rdrop_package() -> None:
    config = runner.load_config()
    training = config["training"]
    rdrop = training["predictive_dropout_consistency"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["epochs"] == 60
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["maximum_fit_count"] == 9
    assert rdrop["dropout_probability"] == 0.1
    assert rdrop["training_passes_per_batch"] == 2
    assert rdrop["coefficient"] == 1.0
    assert rdrop["inseparable_single_intervention"]
    assert not rdrop["inference_dropout"]
    assert not rdrop["sweep"]


def test_predictive_consistency_formula_symmetry_and_equal_noop() -> None:
    receipt = runner._penalty_contract_receipt()
    assert receipt["formula_exact"]
    assert receipt["symmetric"]
    assert receipt["equal_predictions_zero"]
    assert receipt["finite_gradient"]


def test_stochastic_pair_is_seed_reproducible_and_inference_is_fixed() -> None:
    receipt = runner._stochastic_contract_receipt()
    assert receipt["pair_reproducible_under_seed_reset"]
    assert receipt["stochastic_pair_maximum_abs_difference"] > 0.0
    assert receipt["dropout_off_inference_deterministic"]
    assert receipt["dropout_module_count"] == 4


def test_custom_model_masked_token_and_set_permutation_isolation() -> None:
    receipt = runner._isolation_receipt()
    assert max(receipt.values()) <= 1e-6


def test_semantic_audit_discloses_p1_adjacency_without_transfer() -> None:
    audit = runner.semantic_audit(runner.load_config())
    assert audit["repository_p2_exact_execution_hits"] == 0
    assert audit["p1_v39_cross_problem_adjacency_disclosed"]
    assert audit["p1_code_result_gate_transfer_count"] == 0
    assert audit["v24_sam_distinguished"]
    assert audit["v26_mixup_distinguished"]
    assert audit["v38_output_shrinkage_distinguished"]
    assert audit["v39_gradient_mask_distinguished"]
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
    generator = np.random.default_rng(40)
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
        40,
    )
    assert output.shape == (8,)
    assert np.isfinite(output).all()
    assert receipt["two_pass_steps"] == receipt["optimizer_steps"] == 1
    assert receipt["training_passes_per_batch"] == 2
    assert receipt["mean_abs_disagreement_first"] > 0.0
    assert receipt["inference_dropout"] is False
    assert receipt["parameters"] == 4865


def test_penalty_rejects_nonpositive_weights() -> None:
    with torch.no_grad():
        first = torch.zeros(2)
        second = torch.ones(2)
        weights = torch.tensor([1.0, 0.0])
    try:
        runner.predictive_consistency_penalty(first, second, weights)
    except runner.v12.ContractError:
        return
    raise AssertionError("nonpositive weights must fail closed")
