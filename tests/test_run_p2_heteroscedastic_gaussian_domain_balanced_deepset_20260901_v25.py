from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_heteroscedastic_gaussian_domain_balanced_deepset_20260901_v25 as M  # noqa: E402


def test_config_is_one_fixed_heteroscedastic_change_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    gaussian = training["heteroscedastic_gaussian"]
    assert training["architecture"].startswith("v13_exact_shared_element")
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    assert gaussian == {
        "mean_output_index": 0,
        "log_variance_output_index": 1,
        "log_variance_min": -6.0,
        "log_variance_max": 3.0,
        "inference_uses_mean_only": True,
        "variance_router": False,
        "variance_abstention": False,
        "variance_head_sweep": False,
    }
    assert training["row_deletion"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0
    assert config["source_contract"]["official_inputs_allowed"] is False
    assert config["source_contract"]["hidden_truth_allowed"] is False


def test_gaussian_nll_formula_and_bounds() -> None:
    receipt = M._nll_formula_receipt()
    assert receipt["maximum_abs_formula_error"] <= 1e-6
    assert receipt["log_variance_min"] >= -6.0
    assert receipt["log_variance_max"] <= 3.0


def test_mean_only_masked_future_and_permutation_contracts() -> None:
    receipt = M._model_contract_receipt()
    assert receipt["output_width"] == 2
    assert receipt["permutation_maximum_abs_error"] <= 1e-6
    assert receipt["masked_future_token_maximum_abs_error"] <= 1e-6
    assert receipt["mean_invariance_to_variance_head_perturbation_error"] <= 1e-6


def test_prefix_cutoffs_are_validation_start_minus_seven_days() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    preflight = M.preflight()
    for fold, start in config["training"]["fold_starts_kst"].items():
        expected = pd.Timestamp(start) - pd.Timedelta(days=7)
        assert pd.Timestamp(preflight["prefix_cutoffs"][fold]) == expected
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


def test_tiny_training_is_finite_and_mean_only() -> None:
    rng = np.random.default_rng(25)
    rows = 24
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    target = rng.normal(size=rows).astype(np.float32)
    weights = np.ones(rows, dtype=np.float32)
    config = copy.deepcopy(json.loads(M.CONFIG.read_text(encoding="utf-8")))
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 12
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
        25,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["mean_only_inference"] is True
    assert receipt["log_variance_min_contract"] == -6.0
    assert receipt["log_variance_max_contract"] == 3.0
    assert -6.0 <= receipt["query_log_variance_min"] <= 3.0
    assert -6.0 <= receipt["query_log_variance_max"] <= 3.0
    assert receipt["variance_router"] == 0
    assert receipt["variance_abstention"] == 0
