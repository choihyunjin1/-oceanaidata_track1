from __future__ import annotations

import copy
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

import run_p2_lookahead_domain_balanced_deepset_20260901_v29 as M  # noqa: E402


def test_config_is_one_fixed_v13_lookahead_change_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    geometry = training["optimizer_geometry"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    assert geometry["name"] == "Lookahead_AdamW"
    assert geometry["inner_optimizer"] == "AdamW"
    assert geometry["synchronization_period_steps"] == 5
    assert geometry["slow_weight_interpolation_alpha"] == 0.5
    assert geometry["posthoc_weight_ensemble"] is False
    assert geometry["hyperparameter_sweep"] is False
    assert geometry["scheduler"] is False
    assert training["row_deletion"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_lookahead_formula_zero_noop_and_determinism() -> None:
    receipt = M._lookahead_contract_receipt()
    assert receipt["period_steps"] == 5
    assert receipt["alpha"] == 0.5
    assert receipt["formula_exact"] is True
    assert receipt["post_sync_fast_equals_slow"] is True
    assert receipt["zero_difference_exact_noop"] is True
    assert receipt["byte_identical_replay"] is True
    assert receipt["posthoc_weight_ensemble"] is False


def test_masked_future_and_layer_permutation_isolation() -> None:
    receipt = M._isolation_receipt()
    assert receipt["masked_token_maximum_abs_error"] <= 1e-6
    assert receipt["permutation_maximum_abs_error"] <= 1e-6


def test_prefix_cutoffs_and_prospective_gate_are_sealed() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    preflight = M.preflight()
    for fold, start in config["training"]["fold_starts_kst"].items():
        expected = pd.Timestamp(start) - pd.Timedelta(days=7)
        assert pd.Timestamp(preflight["prefix_cutoffs"][fold]) == expected
    gate = config["evaluation"]["safety_gate"]
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003
    assert preflight["data_rows_read"] == 0
    assert preflight["model_fits"] == 0
    assert preflight["official_rows_read"] == 0
    assert preflight["hidden_rows_read"] == 0


def test_zero_operation_preflight_is_byte_identical() -> None:
    first = M.preflight()
    second = M.preflight()
    assert M.v12.sha256_json(first) == M.v12.sha256_json(second)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    assert first["semantic_audit"]["repository_p2_exact_execution_hits"] == 0


def test_tiny_training_has_finite_lookahead_receipts() -> None:
    rng = np.random.default_rng(29)
    rows = 30
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    target = rng.normal(size=rows).astype(np.float32)
    weights = np.ones(rows, dtype=np.float32)
    config = copy.deepcopy(json.loads(M.CONFIG.read_text(encoding="utf-8")))
    config["training"]["epochs"] = 5
    config["training"]["batch_size"] = rows
    prediction, receipt = M.train_predict_seed(
        tokens,
        mask,
        context,
        target,
        weights,
        tokens[:7],
        mask[:7],
        context[:7],
        config,
        29,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["optimizer_steps"] == 5
    assert receipt["lookahead_synchronizations"] == 1
    assert receipt["expected_synchronizations"] == 1
    assert receipt["posthoc_weight_ensemble"] == 0
    assert receipt["row_deletion"] == 0


def test_sync_matches_closed_form_on_two_parameters() -> None:
    first = torch.nn.Parameter(torch.tensor([2.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([-2.0]))
    slow = [torch.tensor([0.0, 0.0]), torch.tensor([2.0])]
    distance = M.lookahead_sync([first, second], slow, 0.5)
    assert np.isclose(distance, 6.0)
    assert torch.equal(first.detach(), torch.tensor([1.0, 2.0]))
    assert torch.equal(second.detach(), torch.tensor([0.0]))
    assert torch.equal(first.detach(), slow[0])
    assert torch.equal(second.detach(), slow[1])
