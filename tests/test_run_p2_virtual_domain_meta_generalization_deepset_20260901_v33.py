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

import run_p2_virtual_domain_meta_generalization_deepset_20260901_v33 as M  # noqa: E402


def test_config_is_exact_v13_with_one_mldg_change() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    meta = training["meta_objective"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["objective"] == "weighted_SmoothL1_MLDG_virtual_layer_month_transfer"
    assert training["optimizer"] == "exact_v13_AdamW_outer"
    assert training["learning_rate"] == 0.001
    assert training["weight_decay"] == 0.0001
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    assert meta["environment"] == "target_layer_x_calendar_month"
    assert meta["inner_learning_rate"] == 0.001
    assert meta["inner_steps"] == 1
    assert meta["meta_test_coefficient"] == 1.0
    assert meta["second_order"] is True
    assert meta["loss_sweep"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_environment_encoding_is_exact_layer_by_month() -> None:
    layer = np.array([2, 2, 3, 3, 4, 4])
    time = pd.to_datetime(
        [
            "2025-01-01T00:00:00+09:00",
            "2025-02-01T00:00:00+09:00",
            "2025-01-02T00:00:00+09:00",
            "2025-02-02T00:00:00+09:00",
            "2025-01-03T00:00:00+09:00",
            "2025-02-03T00:00:00+09:00",
        ]
    )
    encoded, labels = M.encode_layer_month_environments(layer, time)
    assert encoded.tolist() == [0, 1, 2, 3, 4, 5]
    assert labels[0] == "layer2:month01"
    assert labels[5] == "layer4:month02"


def test_virtual_domain_cycle_is_sorted_and_deterministic() -> None:
    receipt = M._virtual_domain_cycle_receipt()
    assert receipt["present_sorted"] == [2, 5, 9]
    assert receipt["first_two_cycles"] == [2, 5, 9, 2, 5, 9]
    assert receipt["deterministic"] is True


def test_scalar_contract_is_second_order_and_finite() -> None:
    receipt = M._mldg_scalar_contract_receipt()
    assert receipt["inner_learning_rate"] == 0.001
    assert receipt["inner_steps"] == 1
    assert receipt["meta_test_coefficient"] == 1.0
    assert receipt["second_order_graph"] is True
    assert receipt["all_finite"] is True
    assert np.isfinite(receipt["outer_gradient"])


def test_batch_objective_has_finite_second_order_gradient() -> None:
    torch.manual_seed(33)
    model = M.v12.VerticalDeepSet(8, 11, hidden=32)
    rows = 12
    tokens = torch.randn(rows, 5, 8)
    mask = torch.ones(rows, 5)
    context = torch.randn(rows, 11)
    target = torch.randn(rows)
    weights = torch.linspace(0.5, 1.5, rows)
    environments = torch.tensor([0, 1, 2] * 4)
    objective, raw, receipt = M.mldg_batch_objective(
        model,
        tokens,
        mask,
        context,
        target,
        weights,
        environments,
        cycle_index=1,
        inner_learning_rate=0.001,
    )
    objective.backward()
    assert raw.shape == (rows,)
    assert receipt["held_domain"] == 1
    assert receipt["meta_train_rows"] == 8
    assert receipt["meta_test_rows"] == 4
    assert receipt["adapted_parameters_finite"] is True
    assert receipt["second_order_graph"] is True
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_masked_future_permutation_and_prospective_gate_are_sealed() -> None:
    isolation = M._isolation_receipt()
    assert isolation["masked_token_maximum_abs_error"] <= 1e-6
    assert isolation["permutation_maximum_abs_error"] <= 1e-6
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    gate = config["evaluation"]["safety_gate"]
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003


def test_two_zero_operation_preflights_are_byte_identical_and_access_zero() -> None:
    first = M.preflight()
    second = M.preflight()
    assert M.v12.sha256_json(first) == M.v12.sha256_json(second)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    assert first["semantic_audit"]["repository_p2_exact_execution_hits"] == 0
    assert first["semantic_audit"]["official_v23_feedback_used_for_selection"] is False
    for name in (
        "data_rows_read",
        "model_fits",
        "artifacts_written",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert first[name] == 0


def test_tiny_training_has_finite_mldg_receipt_and_domain_coverage() -> None:
    rng = np.random.default_rng(33)
    rows = 36
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    target = rng.normal(size=rows).astype(np.float32)
    weights = np.linspace(0.5, 1.5, rows).astype(np.float32)
    M._CURRENT_ENVIRONMENTS = np.tile(np.arange(3, dtype=np.int16), 12)
    M._CURRENT_ENVIRONMENT_LABELS = {0: "layer2:month01", 1: "layer3:month01", 2: "layer4:month01"}
    config = copy.deepcopy(json.loads(M.CONFIG.read_text(encoding="utf-8")))
    config["training"]["epochs"] = 3
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
        33,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["epochs"] == 3
    assert receipt["second_order"] is True
    assert receipt["second_order_steps"] == 3
    assert receipt["inner_steps_per_outer_step"] == 1
    assert receipt["inner_learning_rate"] == 0.001
    assert receipt["meta_test_coefficient"] == 1.0
    assert receipt["minimum_present_domains_per_batch"] == 3
    assert all(value == 1 for value in receipt["held_domain_steps"].values())
    assert np.isfinite(receipt["maximum_inner_gradient_norm"])
    assert np.isfinite(receipt["maximum_outer_gradient_norm"])
    assert receipt["row_deletion"] == 0
