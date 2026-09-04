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

import run_p2_masked_token_virtual_adversarial_deepset_20260901_v42 as M  # noqa: E402


def test_config_is_exact_single_vat_change_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    vat = training["virtual_adversarial_training"]
    assert config["experiment_id"] == M.EXPERIMENT_ID
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["optimizer"] == "exact_v13_AdamW"
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert vat["eligible_token_channels"] == [0, 1]
    assert vat["presence_indicator_channels"] == [4, 5]
    assert vat["epsilon_normalized_L2_per_row"] == 0.05
    assert vat["finite_difference_xi"] == 1e-6
    assert vat["power_iterations"] == 1
    assert vat["coefficient"] == 1.0
    assert not vat["perturb_masks"]
    assert not vat["perturb_depth_or_nominal"]
    assert not vat["perturb_presence_indicators"]
    assert not vat["perturb_context_or_target_layer_one_hot"]
    assert not vat["perturb_labels"]
    assert not vat["inference_perturbation"]
    assert not vat["sweep"]
    assert config["operation_limits"]["official_rows_read"] == 0
    assert config["operation_limits"]["hidden_rows_read"] == 0
    assert config["operation_limits"]["submission_csv_created"] == 0
    assert config["operation_limits"]["uploads"] == 0


def _synthetic() -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    model = M.v12.VerticalDeepSet(8, 11, hidden=32)
    tokens = torch.randn(5, 5, 8)
    tokens[:, :, 4:] = 1.0
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    context = torch.randn(5, 11)
    weights = torch.arange(1, 6, dtype=torch.float32)
    return model, tokens, mask, context, weights


def test_eligible_mask_only_observed_temperature_salinity() -> None:
    _, tokens, mask, _, _ = _synthetic()
    tokens[0, 0, 5] = 0.0
    eligible = M.eligible_value_mask(tokens, mask)
    assert not eligible[:, :, 2:].any()
    assert not eligible[mask == 0].any()
    assert eligible[0, 0, 0]
    assert not eligible[0, 0, 1]


def test_direction_norm_mask_zero_noop_and_repeat() -> None:
    model, tokens, mask, context, weights = _synthetic()
    clean = model(tokens, mask, context)
    first, receipt = M.virtual_adversarial_direction(
        model, tokens, mask, context, clean, weights
    )
    second, _ = M.virtual_adversarial_direction(
        model, tokens, mask, context, clean, weights
    )
    zero, _ = M.virtual_adversarial_direction(
        model, tokens, mask, context, clean, weights, epsilon=0.0
    )
    eligible = M.eligible_value_mask(tokens, mask)
    norms = torch.linalg.vector_norm(first.flatten(1), dim=1)
    assert torch.max(torch.abs(norms - 0.05)) <= 1e-6
    assert torch.equal(first[~eligible], torch.zeros_like(first[~eligible]))
    assert torch.equal(first, second)
    assert torch.equal(zero, torch.zeros_like(zero))
    assert receipt["power_iterations"] == 1
    assert receipt["label_reads"] == 0
    assert not first.requires_grad


def test_permutation_equivariance_future_isolation_and_penalty_stopgrad() -> None:
    model, tokens, mask, context, weights = _synthetic()
    clean = model(tokens, mask, context)
    direction, _ = M.virtual_adversarial_direction(
        model, tokens, mask, context, clean, weights
    )
    order = torch.tensor([4, 2, 0, 3, 1])
    permuted = tokens[:, order]
    permuted_mask = mask[:, order]
    permuted_clean = model(permuted, permuted_mask, context)
    permuted_direction, _ = M.virtual_adversarial_direction(
        model, permuted, permuted_mask, context, permuted_clean, weights
    )
    assert torch.max(torch.abs(direction[:, order] - permuted_direction)) <= 1e-6
    changed = tokens.clone()
    changed[mask == 0] = 1e6
    assert torch.max(torch.abs(model(tokens, mask, context) - model(changed, mask, context))) <= 1e-6
    adversarial = model(tokens + direction, mask, context)
    penalty = M.vat_consistency_penalty(clean, adversarial, weights)
    gradient = torch.autograd.grad(penalty, adversarial)[0]
    assert torch.isfinite(gradient).all()
    assert M.vat_consistency_penalty(clean, adversarial, weights, coefficient=0.0) == 0.0


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
    assert first["vat_contract"]["target_reads"] == 0
    assert first["validation_rows_used_for_direction"] == 0


def test_tiny_training_has_one_direction_step_per_optimizer_step() -> None:
    rng = np.random.default_rng(17)
    rows = 18
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    tokens[:, :, 4:] = 1.0
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
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
    assert receipt["vat_direction_steps"] == 2
    assert receipt["power_iterations_per_step"] == 1
    assert receipt["eligible_rows_seen"] == rows
    assert receipt["maximum_epsilon_norm_error"] <= 1e-6
    assert receipt["label_reads_for_direction"] == 0
    assert receipt["parameter_perturbation_steps"] == 0
    assert receipt["row_mixing_steps"] == 0


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
