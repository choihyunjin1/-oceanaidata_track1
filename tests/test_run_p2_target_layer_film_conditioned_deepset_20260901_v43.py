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

import run_p2_target_layer_film_conditioned_deepset_20260901_v43 as M  # noqa: E402


def test_config_is_exact_identity_film_change_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    film = training["target_layer_film"]
    assert config["experiment_id"] == M.EXPERIMENT_ID
    assert training["objective"] == "exact_v13_weighted_SmoothL1_beta_1.0"
    assert training["optimizer"] == "exact_v13_AdamW"
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert film["conditioning_context_columns"] == [1, 2, 3]
    assert film["generator"] == "Linear_3_to_64_bias_false"
    assert film["added_parameters"] == 192
    assert film["gamma_initialization"] == 1.0
    assert film["beta_initialization"] == 0.0
    assert film["broadcast_same_gamma_beta_to_all_public_tokens_in_row"]
    assert not film["normalization"]
    assert not film["attention"]
    assert not film["sweep"]
    assert not training["extra_loss"]
    assert config["operation_limits"]["official_rows_read"] == 0
    assert config["operation_limits"]["hidden_rows_read"] == 0
    assert config["operation_limits"]["submission_csv_created"] == 0
    assert config["operation_limits"]["uploads"] == 0


def test_film_generator_identity_initial_function_and_parameter_contract() -> None:
    receipt = M._film_contract_receipt()
    assert receipt["generator_class"] == "Linear"
    assert receipt["generator_count"] == 1
    assert receipt["generator_shape"] == [64, 3]
    assert not receipt["generator_bias"]
    assert receipt["added_parameters"] == 192
    assert receipt["film_parameters"] - receipt["base_parameters"] == 192
    assert receipt["gamma_initialization_maximum_abs_error"] == 0.0
    assert receipt["beta_initialization_maximum_abs_error"] == 0.0
    assert receipt["initial_function_maximum_abs_error"] <= 1e-6
    assert receipt["normalization_module_count"] == 0
    assert receipt["attention_module_count"] == 0
    assert receipt["finite"]


def test_target_layer_one_hot_selects_only_its_column() -> None:
    model = M.TargetLayerFilmDeepSet(8, 11, hidden=32)
    with torch.no_grad():
        model.target_layer_film.weight.fill_(0.0)
        model.target_layer_film.weight[0, 0] = 2.0
        model.target_layer_film.weight[0, 1] = 3.0
        model.target_layer_film.weight[0, 2] = 4.0
    output = model.target_layer_film(torch.eye(3))
    assert torch.equal(output[:, 0], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.count_nonzero(output[:, 1:]) == 0


def test_permutation_masked_future_and_repeat_isolation() -> None:
    receipt = M._isolation_receipt()
    assert receipt["masked_or_future_token_maximum_abs_error"] <= 1e-6
    assert receipt["permutation_maximum_abs_error"] <= 1e-6
    assert receipt["repeat_maximum_abs_error"] == 0.0


def test_target_free_preflight_is_byte_identical_and_namespace_zero() -> None:
    first = M.preflight()
    second = M.preflight()
    assert M.v12.sha256_json(first) == M.v12.sha256_json(second)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    assert first["model_fits"] == 0
    assert first["data_rows_read"] == 0
    assert first["official_rows_read"] == 0
    assert first["hidden_rows_read"] == 0
    assert first["submission_csv_created"] == 0
    assert first["uploads"] == 0


def test_tiny_training_updates_only_one_film_generator_contract() -> None:
    rng = np.random.default_rng(43)
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
    assert receipt["parameters"] == 5057
    assert receipt["film_generator_count"] == 1
    assert receipt["film_added_parameters"] == 192
    assert receipt["initial_film_state_sha256"] != receipt["final_film_state_sha256"]
    assert receipt["normalization_modules"] == 0
    assert receipt["attention_modules"] == 0
    assert receipt["extra_loss_terms"] == 0


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
