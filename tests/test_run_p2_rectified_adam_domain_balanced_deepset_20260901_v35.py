from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_rectified_adam_domain_balanced_deepset_20260901_v35 as M  # noqa: E402


def test_config_is_exact_v13_with_one_fixed_radam_change() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    optimizer = training["optimizer"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["objective"] == "weighted_SmoothL1_beta_1.0_with_fixed_RAdam"
    assert optimizer == {
        "class": "torch.optim.RAdam",
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0001,
        "decoupled_weight_decay": True,
        "warmup": False,
        "scheduler": False,
        "lookahead": False,
        "gradient_projection": False,
        "sweep": False,
    }
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_radam_contract_is_exact_and_deterministic() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    receipt = M._radam_contract_receipt(config)
    assert receipt["class"] == "RAdam"
    assert receipt["learning_rate"] == 0.001
    assert receipt["betas"] == [0.9, 0.999]
    assert receipt["epsilon"] == 1e-8
    assert receipt["weight_decay"] == 0.0001
    assert receipt["decoupled_weight_decay"] is True
    assert receipt["foreach"] is False
    assert receipt["state_keys_exact"] is True
    assert receipt["state_step_min"] == 2
    assert receipt["state_step_max"] == 2
    assert receipt["state_finite"] is True
    assert receipt["deterministic"] is True
    assert receipt["contract_exact"] is True
    assert receipt["warmup"] is False
    assert receipt["scheduler"] is False
    assert receipt["sweep"] is False


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


def test_tiny_training_has_finite_fixed_radam_receipt() -> None:
    rng = np.random.default_rng(35)
    rows = 30
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    target = rng.normal(size=rows).astype(np.float32)
    weights = np.linspace(0.5, 1.5, rows).astype(np.float32)
    config = copy.deepcopy(json.loads(M.CONFIG.read_text(encoding="utf-8")))
    config["training"]["epochs"] = 2
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
        35,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["epochs"] == 2
    assert receipt["optimizer_steps"] == 2
    assert receipt["optimizer"]["class"] == "RAdam"
    assert receipt["optimizer"]["state_count"] == 10
    assert receipt["optimizer"]["state_step_min"] == 2
    assert receipt["optimizer"]["state_step_max"] == 2
    assert receipt["warmup_steps"] == 0
    assert receipt["scheduler_steps"] == 0
    assert receipt["lookahead_steps"] == 0
    assert receipt["gradient_projection_steps"] == 0
    assert receipt["row_deletion"] == 0


def test_semantic_axis_is_distinct_and_official_feedback_selection_is_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    audit = M.semantic_audit(config)
    assert audit["classification"] == "NEW_P2_FIXED_RECTIFIED_ADAM_DEEPSET"
    assert audit["repository_p2_exact_execution_hits"] == 0
    assert audit["v13_adamw_distinguished"] is True
    assert audit["v24_sam_distinguished"] is True
    assert audit["v29_lookahead_distinguished"] is True
    assert audit["v34_gradient_centralization_distinguished"] is True
    assert audit["official_v23_feedback_used_for_selection"] is False
