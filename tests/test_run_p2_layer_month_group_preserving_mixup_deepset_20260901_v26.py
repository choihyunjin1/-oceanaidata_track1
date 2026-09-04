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

import run_p2_layer_month_group_preserving_mixup_deepset_20260901_v26 as M  # noqa: E402


def test_config_is_fixed_group_mixup_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    mixup = training["mixup"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["maximum_final_action_C"] == 0.5
    assert mixup["alpha"] == 0.2
    assert mixup["partner_group"] == "exact_target_layer_x_calendar_month"
    assert mixup["cross_group_mixing"] is False
    assert mixup["alpha_sweep"] is False
    assert mixup["inference_augmentation"] is False
    assert training["row_deletion"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_mixup_is_deterministic_group_isolated_and_fallback_exact() -> None:
    receipt = M._mixup_contract_receipt()
    assert receipt["byte_identical_seed_replay"] is True
    assert receipt["cross_group_pairs"] == 0
    assert receipt["insufficient_intersection_exact_noop"] is True
    assert receipt["minimum_output_mask_tokens"] >= 2.0


def test_convex_token_context_target_and_weight_formula() -> None:
    tokens = torch.stack((torch.zeros(5, 8), torch.full((5, 8), 10.0)))
    mask = torch.ones(2, 5)
    context = torch.stack((torch.zeros(11), torch.full((11,), 20.0)))
    target = torch.tensor([0.0, 30.0])
    weight = torch.tensor([1.0, 5.0])
    mixed, receipt = M.group_preserving_mixup(
        [tokens, mask, context, target, weight],
        np.array([205, 205]),
        0.2,
        np.random.default_rng(2601),
    )
    assert receipt["cross_group_pairs"] == 0
    partner_fraction = float(mixed[3][0] / 30.0)
    assert torch.allclose(mixed[0][0], torch.full((5, 8), 10.0 * partner_fraction))
    assert torch.allclose(mixed[2][0], torch.full((11,), 20.0 * partner_fraction))
    assert torch.allclose(mixed[4][0], torch.tensor(1.0 + 4.0 * partner_fraction))


def test_layer_month_capture_prefix_and_permutation_contracts() -> None:
    assert M._group_capture_receipt()["exact"] is True
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    preflight = M.preflight()
    for fold, start in config["training"]["fold_starts_kst"].items():
        assert pd.Timestamp(preflight["prefix_cutoffs"][fold]) == pd.Timestamp(
            start
        ) - pd.Timedelta(days=7)
    assert M.v12.permutation_invariance_receipt()["maximum_abs_error"] <= 1e-6
    assert preflight["data_rows_read"] == 0
    assert preflight["model_fits"] == 0


def test_zero_operation_preflight_is_byte_identical() -> None:
    first = M.preflight()
    second = M.preflight()
    assert M.v12.sha256_json(first) == M.v12.sha256_json(second)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    assert first["semantic_audit"]["repository_p2_exact_execution_hits"] == 0


def test_tiny_training_is_finite_no_cross_group_and_no_inference_mix() -> None:
    rng = np.random.default_rng(26)
    rows = 24
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    target = rng.normal(size=rows).astype(np.float32)
    weights = np.ones(rows, dtype=np.float32)
    layers = np.repeat([2, 3, 4], 8)
    local = pd.date_range("2024-05-01", periods=rows, freq="10min", tz="Asia/Seoul")
    M.domain_balanced_weights(layers, local)
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
        26,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["cross_group_pairs"] == 0
    assert receipt["minimum_output_mask_tokens"] >= 2.0
    assert receipt["mixup_alpha"] == 0.2
    assert receipt["inference_augmentation"] == 0
