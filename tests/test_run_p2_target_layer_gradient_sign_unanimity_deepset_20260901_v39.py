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

import run_p2_target_layer_gradient_sign_unanimity_deepset_20260901_v39 as runner  # noqa: E402


def test_config_is_exact_v13_plus_unanimity_gradient_replacement() -> None:
    config = runner.load_config()
    training = config["training"]
    mask = training["gradient_sign_unanimity"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["epochs"] == 60
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["maximum_fit_count"] == 9
    assert mask["tasks"] == [2, 3, 4]
    assert mask["agreement_threshold"] == 1.0
    assert mask["zero_gradient_counts_as_non_unanimous"]
    assert not any(
        mask[name]
        for name in ("projection", "partial_agreement", "group_reweighting", "sweep")
    )


def test_and_mask_formula_conflict_zero_and_mean_identity() -> None:
    receipt = runner._and_mask_contract_receipt()
    assert receipt["formula_exact"]
    assert receipt["conflicting_coordinate_zeroed"]
    assert receipt["zero_coordinate_non_unanimous"]
    assert receipt["all_agree_mean_identity"]
    assert receipt["task_permutation_invariant"]


def test_and_mask_is_task_permutation_invariant() -> None:
    gradients = [
        [torch.tensor([1.0, -1.0, 2.0])],
        [torch.tensor([3.0, -2.0, -1.0])],
        [torch.tensor([2.0, -3.0, 4.0])],
    ]
    first, first_receipt = runner.and_mask_gradients(gradients)
    second, second_receipt = runner.and_mask_gradients(
        [gradients[2], gradients[0], gradients[1]]
    )
    assert torch.equal(first[0], second[0])
    assert first_receipt == second_receipt
    assert first[0].tolist() == [2.0, -2.0, 0.0]


def test_masked_token_and_set_permutation_isolation() -> None:
    receipt = runner.v37._isolation_receipt()
    assert max(receipt.values()) <= 1e-6


def test_semantic_audit_discloses_cross_problem_adjacency() -> None:
    audit = runner.semantic_audit(runner.load_config())
    assert audit["repository_p2_exact_execution_hits"] == 0
    assert audit["p1_v45_cross_problem_adjacency_disclosed"]
    assert audit["v28_pcgrad_distinguished"]
    assert audit["v36_fishr_distinguished"]
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
    generator = np.random.default_rng(39)
    rows = 30
    runner._CURRENT_TARGET_LAYERS = np.repeat(np.array([2, 3, 4]), 10)
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
        39,
    )
    assert output.shape == (8,)
    assert np.isfinite(output).all()
    assert receipt["gradient_mask_steps"] == receipt["optimizer_steps"] == 1
    assert receipt["agreement_threshold"] == 1.0
    assert receipt["minimum_task_rows_per_batch"] == {2: 10, 3: 10, 4: 10}
