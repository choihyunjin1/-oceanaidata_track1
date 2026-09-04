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

import run_p2_sharpness_aware_domain_balanced_deepset_20260901_v24 as M  # noqa: E402


def test_config_is_one_fixed_v13_optimizer_change_and_access_zero() -> None:
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
    assert geometry == {
        "name": "vanilla_SAM",
        "rho": 0.05,
        "gradient_norm": "global_L2_over_all_trainable_parameters",
        "adaptive_asam": False,
        "rho_sweep": False,
        "scheduler": False,
        "base_optimizer": "AdamW",
        "apply_every_batch": True,
    }
    assert training["row_deletion"] is False
    assert training["input_perturbation"] is False
    assert training["data_augmentation"] is False
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
    assert config["source_contract"]["submission_csv_allowed"] is False
    assert config["source_contract"]["upload_allowed"] is False


def test_sam_global_radius_restore_second_loss_and_zero_grad_contract() -> None:
    receipt = M._sam_contract_receipt()
    assert receipt["rho"] == 0.05
    assert receipt["first_gradient_global_l2_norm"] > 0.0
    assert abs(receipt["actual_parameter_perturbation_l2_norm"] - 0.05) <= 1e-6
    assert receipt["parameter_restore_bit_exact"] is True
    assert receipt["second_loss_finite"] is True
    assert receipt["zero_gradient_exact_noop"] is True


def test_masked_future_token_and_layer_permutation_isolation() -> None:
    assert M._masked_token_isolation_receipt()["maximum_abs_error"] <= 1e-6
    assert M.v12.permutation_invariance_receipt()["maximum_abs_error"] <= 1e-6


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


def test_tiny_training_has_finite_sam_receipts() -> None:
    rng = np.random.default_rng(24)
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
        24,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["sam_rho"] == 0.05
    assert receipt["perturbation_radius_first"] > 0.049999
    assert receipt["perturbation_radius_first"] < 0.050001
    assert np.isfinite(receipt["sam_second_loss_first"])
    assert receipt["parameter_restore_before_adamw_step"] is True
