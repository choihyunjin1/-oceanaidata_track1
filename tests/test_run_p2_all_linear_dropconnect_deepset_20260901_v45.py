from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_all_linear_dropconnect_deepset_20260901_v45 as M  # noqa: E402


def test_config_is_exact_v13_plus_fixed_dropconnect_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    dropconnect = training["dropconnect"]
    assert config["experiment_id"] == M.EXPERIMENT_ID
    assert training["objective"] == "exact_v13_weighted_SmoothL1_beta_1.0"
    assert training["optimizer"] == "exact_v13_AdamW"
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert dropconnect["drop_probability"] == 0.1
    assert dropconnect["keep_probability"] == 0.9
    assert dropconnect["linear_module_count"] == 5
    assert not dropconnect["bias_masked"]
    assert not dropconnect["activation_dropout"]
    assert not dropconnect["prediction_consistency"]
    assert not dropconnect["monte_carlo_inference"]
    assert not dropconnect["ensemble_or_model_aggregation"]
    assert dropconnect["evaluation_uses_raw_weight_once"]
    assert not dropconnect["sweep"]
    assert config["operation_limits"]["official_rows_read"] == 0
    assert config["operation_limits"]["hidden_rows_read"] == 0
    assert config["operation_limits"]["submission_csv_created"] == 0
    assert config["operation_limits"]["uploads"] == 0


def test_dropconnect_contract_shapes_state_and_mask_stream() -> None:
    receipt = M._dropconnect_contract_receipt()
    assert receipt["module_count"] == 5
    assert receipt["module_shapes"] == [
        [32, 8],
        [32, 32],
        [32, 75],
        [32, 32],
        [1, 32],
    ]
    assert receipt["all_biases_unmasked"]
    assert receipt["parameters"] == 4865
    assert receipt["parameter_tensors"] == 10
    assert receipt["buffers"] == 0
    assert receipt["parameter_names_equal_v13"]
    assert receipt["evaluation_initial_function_maximum_abs_error"] == 0.0
    assert receipt["deterministic_same_seed_training_maximum_abs_error"] == 0.0
    assert receipt["deterministic_same_seed_mask_hashes"]
    assert receipt["consecutive_step_masks_distinct"]
    assert 0.85 <= receipt["first_step_keep_share"] <= 0.95
    assert receipt["zero_probability_training_maximum_abs_error"] == 0.0
    assert receipt["evaluation_repeat_maximum_abs_error"] == 0.0
    assert receipt["evaluation_rng_unchanged"]
    assert receipt["dropout_module_count"] == 0


def test_dropconnect_masks_weight_not_bias_and_uses_inverted_scale() -> None:
    module = M.DropConnectLinear(4, 3, bias=True, drop_probability=0.1).train()
    with torch.no_grad():
        module.weight.fill_(1.0)
        module.bias.fill_(2.0)
    value = torch.zeros(5, 4)
    torch.manual_seed(45)
    output = module(value)
    assert torch.equal(output, torch.full((5, 3), 2.0))
    stats = M.dropconnect_statistics(module)
    assert stats["module_count"] == 1
    assert stats["mask_calls"] == 1
    assert stats["total_weight_draws"] == 12
    assert stats["modules"][""]["bias_present"]
    assert stats["modules"][""]["drop_probability"] == 0.1


def test_eval_permutation_masked_future_repeat_and_rng_isolation() -> None:
    receipt = M._isolation_receipt()
    assert receipt["masked_or_future_token_maximum_abs_error"] <= 1e-6
    assert receipt["permutation_maximum_abs_error"] <= 1e-6
    assert receipt["repeat_maximum_abs_error"] == 0.0


def test_target_free_preflight_is_byte_identical_and_namespace_zero() -> None:
    first = M.preflight()
    second = M.preflight()
    assert M.v12.sha256_json(first) == M.v12.sha256_json(second)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["model_fits"] == 0
    assert first["data_rows_read"] == 0
    assert first["official_rows_read"] == 0
    assert first["hidden_rows_read"] == 0
    assert first["submission_csv_created"] == 0
    assert first["uploads"] == 0


def test_tiny_training_uses_five_masks_and_raw_weight_inference() -> None:
    rng = np.random.default_rng(45)
    rows = 18
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    tokens[:, :, 4:] = 1.0
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    context[:, 1:4] = np.eye(3, dtype=np.float32)[np.arange(rows) % 3]
    target = rng.normal(size=rows).astype(np.float32)
    weights = np.linspace(0.5, 1.5, rows, dtype=np.float32)
    config = copy.deepcopy(json.loads(M.CONFIG.read_text(encoding="utf-8")))
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 9
    prediction, receipt = M.train_predict_seed(
        tokens,
        mask,
        context,
        target,
        weights,
        tokens[:4],
        mask[:4],
        context[:4],
        config,
        20260901,
    )
    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()
    assert receipt["optimizer_steps"] == 2
    assert receipt["parameters"] == 4865
    assert receipt["parameter_tensors"] == 10
    assert receipt["buffers"] == 0
    assert receipt["dropconnect_statistics"]["module_count"] == 5
    assert receipt["dropconnect_statistics"]["mask_calls"] == 10
    assert receipt["expected_mask_calls"] == 10
    assert receipt["initial_rng_sha256"] != receipt["final_rng_sha256"]
    assert receipt["evaluation_rng_unchanged"]
    assert receipt["dropout_module_count"] == 0
    assert receipt["prediction_consistency_loss"] == 0
    assert receipt["monte_carlo_inference"] == 0
    assert receipt["ensemble_models"] == 1
    assert receipt["loss_finite"]


def test_prospective_gate_is_unchanged() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    gate = config["evaluation"]["safety_gate"]
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003
    assert (
        config["authorization_evidence"]["prospective_gate_amendment_sha256"]
        == "c7fde8c5d9f535ab8080eb561bf082c55e5c7172117d00e12e1479f9b4417680"
    )
