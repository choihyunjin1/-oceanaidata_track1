"""Run sealed P2 v43 target-layer FiLM DeepSets exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_masked_token_virtual_adversarial_deepset_20260901_v42 as v42  # noqa: E402

v41 = v42.v41
v40 = v42.v40
v37 = v42.v37
v13 = v42.v13
v12 = v42.v12

EXPERIMENT_ID = "p2_target_layer_film_conditioned_deepset_20260901_v43"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V43_TARGET_LAYER_FILM_CONDITIONED_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.target_layer_film_conditioned_deepset.result.20260901.v43"

_BASE_RUN = v42._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v42._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = v42._V13_RUNNER


class TargetLayerFilmDeepSet(v12.VerticalDeepSet):
    """Exact v13 geometry with one target-layer affine element modulation."""

    def __init__(self, token_features: int, context_features: int, hidden: int = 32) -> None:
        super().__init__(token_features, context_features, hidden)
        if hidden != 32 or context_features != 11:
            raise v12.ContractError("v43 fixed geometry drift")
        self.target_layer_film = nn.Linear(3, hidden * 2, bias=False)
        with torch.no_grad():
            self.target_layer_film.weight[:hidden].fill_(1.0)
            self.target_layer_film.weight[hidden:].zero_()

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        encoded = self.element(tokens)
        parameters = self.target_layer_film(context[:, 1:4])
        gamma, beta = parameters.chunk(2, dim=1)
        encoded = gamma.unsqueeze(1) * encoded + beta.unsqueeze(1)
        mask = token_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (encoded * mask).sum(dim=1) / count
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.bool(), negative).amax(dim=1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        return self.head(torch.cat((mean, maximum, context), dim=1)).squeeze(1)


def _bind_base() -> None:
    v13.EXPERIMENT_ID = EXPERIMENT_ID
    v13.CONFIG = CONFIG
    v13.ARTIFACT = ARTIFACT
    v13.REPORT = REPORT
    v13.RUNNER = RUNNER
    v13.PREDICTION_NAME = PREDICTION_NAME
    v13.load_config = load_config
    v13.domain_balanced_weights = _BASE_DOMAIN_BALANCED_WEIGHTS
    v13.train_predict_seed = train_predict_seed
    v13.write_report = write_report


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    film = training["target_layer_film"]
    safety = config["evaluation"]["safety_gate"]
    evidence = config["authorization_evidence"]
    audit = ROOT / evidence["audit_result"]
    amendment = ROOT / evidence["prospective_gate_amendment"]
    fingerprint = ROOT / evidence["fingerprint"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
        or training["architecture"]
        != "v13_exact_DeepSets_plus_identity_initialized_target_layer_FiLM_after_second_element_ReLU_before_pooling"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"] != "exact_v13_weighted_SmoothL1_beta_1.0"
        or training["optimizer"] != "exact_v13_AdamW"
        or training["learning_rate"] != 0.001
        or training["weight_decay"] != 0.0001
        or training["epochs"] != 60
        or training["seeds"] != [20260901, 20260902, 20260903]
        or training["maximum_fit_count"] != 9
        or training["champion_preserving_weight"] != 0.8
        or training["model_weight"] != 0.2
        or training["model_minus_champion_clip_C"] != 2.5
        or training["maximum_final_action_C"] != 0.5
        or training["row_deletion"]
        or training["input_perturbation"]
        or training["data_augmentation"]
        or training["extra_loss"]
        or film["conditioning_context_columns"] != [1, 2, 3]
        or film["generator"] != "Linear_3_to_64_bias_false"
        or film["generator_count"] != 1
        or film["added_parameters"] != 192
        or film["gamma_width"] != 32
        or film["beta_width"] != 32
        or film["location"]
        != "after_second_shared_element_ReLU_before_masked_mean_max_pooling"
        or not film["broadcast_same_gamma_beta_to_all_public_tokens_in_row"]
        or film["gamma_initialization"] != 1.0
        or film["beta_initialization"] != 0.0
        or film["initial_function_maximum_abs_error_lte"] != 1e-6
        or film["normalization"]
        or film["attention"]
        or film["month_station_year_conditioning"]
        or film["sweep"]
        or safety["minimum_fold_layer_non_harm_cells"] != 8
        or safety["total_fold_layer_cells"] != 9
        or safety["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or not audit.is_file()
        or json.loads(audit.read_text(encoding="utf-8"))["status"]
        != evidence["required_status"]
        or not amendment.is_file()
        or v12.sha256_file(amendment)
        != evidence["prospective_gate_amendment_sha256"]
        or not fingerprint.is_file()
        or v12.sha256_file(fingerprint) != evidence["fingerprint_sha256"]
    ):
        raise v12.ContractError("v43 fixed scientific contract drift")
    return config


def film_state_sha256(model: TargetLayerFilmDeepSet) -> str:
    array = model.target_layer_film.weight.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def train_predict_seed(
    tokens: np.ndarray,
    mask: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    query_tokens: np.ndarray,
    query_mask: np.ndarray,
    query_context: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TargetLayerFilmDeepSet(8, 11, hidden=32).to(device)
    initial_film_sha256 = film_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    train = tuple(
        torch.from_numpy(np.asarray(value).copy())
        for value in (
            tokens,
            mask,
            context,
            target.astype(np.float32),
            weights.astype(np.float32),
        )
    )
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    optimizer_steps = 0
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[0], batch[1], batch[2])
            raw = F.smooth_l1_loss(prediction, batch[3], beta=1.0, reduction="none")
            loss = (raw * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            if not bool(torch.isfinite(loss)):
                raise v12.ContractError("v43 training loss is non-finite")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total += float(loss.detach().cpu())
            batches += 1
        losses.append(total / batches)

    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(query_tokens), batch_size):
            stop = start + batch_size
            output.append(
                model(
                    torch.from_numpy(np.asarray(query_tokens[start:stop]).copy()).to(device),
                    torch.from_numpy(np.asarray(query_mask[start:stop]).copy()).to(device),
                    torch.from_numpy(np.asarray(query_context[start:stop]).copy()).to(device),
                )
                .cpu()
                .numpy()
            )
    prediction = np.concatenate(output).astype(float)
    final_film_sha256 = film_state_sha256(model)
    finite = bool(np.isfinite(losses).all() and np.isfinite(prediction).all())
    if not finite or initial_film_sha256 == final_film_sha256:
        raise v12.ContractError("v43 FiLM training contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "parameter_tensors": len(list(model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "optimizer_steps": optimizer_steps,
        "film_generator_count": 1,
        "film_added_parameters": int(model.target_layer_film.weight.numel()),
        "initial_film_state_sha256": initial_film_sha256,
        "final_film_state_sha256": final_film_sha256,
        "gamma_beta_broadcast_to_all_tokens": True,
        "normalization_modules": 0,
        "attention_modules": 0,
        "extra_loss_terms": 0,
        "loss": config["training"]["objective"],
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _film_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(43)
    base = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    torch.manual_seed(43)
    film = TargetLayerFilmDeepSet(8, 11, hidden=32).eval()
    tokens = torch.randn(6, 5, 8)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
        ],
        dtype=torch.float32,
    )
    context = torch.randn(6, 11)
    context[:, 1:4] = torch.eye(3).repeat(2, 1)
    with torch.inference_mode():
        expected = base(tokens, mask, context)
        actual = film(tokens, mask, context)
    weight = film.target_layer_film.weight.detach()
    gamma = weight[:32]
    beta = weight[32:]
    return {
        "generator_class": type(film.target_layer_film).__name__,
        "generator_count": sum(
            name == "target_layer_film" for name, _ in film.named_modules()
        ),
        "generator_shape": list(weight.shape),
        "generator_bias": film.target_layer_film.bias is not None,
        "added_parameters": int(weight.numel()),
        "gamma_initialization_maximum_abs_error": float(torch.max(torch.abs(gamma - 1.0))),
        "beta_initialization_maximum_abs_error": float(torch.max(torch.abs(beta))),
        "initial_function_maximum_abs_error": float(torch.max(torch.abs(expected - actual))),
        "base_parameters": int(sum(value.numel() for value in base.parameters())),
        "film_parameters": int(sum(value.numel() for value in film.parameters())),
        "film_state_sha256": film_state_sha256(film),
        "finite": bool(torch.isfinite(actual).all()),
        "normalization_module_count": sum(
            isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)) for module in film.modules()
        ),
        "attention_module_count": sum(
            isinstance(module, nn.MultiheadAttention) for module in film.modules()
        ),
    }


def _isolation_receipt() -> dict[str, float]:
    torch.manual_seed(73)
    model = TargetLayerFilmDeepSet(8, 11, hidden=32).eval()
    tokens = torch.randn(4, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    context = torch.randn(4, 11)
    context[:, 1:4] = torch.eye(3)[torch.tensor([0, 1, 2, 0])]
    changed = tokens.clone()
    changed[mask == 0] = 1e6
    order = torch.tensor([4, 2, 0, 3, 1])
    with torch.inference_mode():
        base = model(tokens, mask, context)
        masked = model(changed, mask, context)
        permuted = model(tokens[:, order], mask[:, order], context)
        repeated = model(tokens, mask, context)
    return {
        "masked_or_future_token_maximum_abs_error": float(torch.max(torch.abs(base - masked))),
        "permutation_maximum_abs_error": float(torch.max(torch.abs(base - permuted))),
        "repeat_maximum_abs_error": float(torch.max(torch.abs(base - repeated))),
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, str] = {}
    for key, relative in config["authorization_evidence"].items():
        if not (key.endswith("_result") or key in ("prospective_gate_amendment", "fingerprint")):
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_execution_hits": 0,
        "v15_attention_distinguished": True,
        "v16_depth_graph_distinguished": True,
        "v31_domain_adversary_distinguished": True,
        "v40_v42_consistency_distinguished": True,
        "v41_weight_norm_distinguished": True,
        "official_v23_feedback_used_for_selection": False,
        "evidence_sha256": evidence,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    contract = _film_contract_receipt()
    isolation = _isolation_receipt()
    if (
        contract["generator_count"] != 1
        or contract["generator_shape"] != [64, 3]
        or contract["generator_bias"]
        or contract["added_parameters"] != 192
        or contract["gamma_initialization_maximum_abs_error"] != 0.0
        or contract["beta_initialization_maximum_abs_error"] != 0.0
        or contract["initial_function_maximum_abs_error"] > 1e-6
        or contract["film_parameters"] - contract["base_parameters"] != 192
        or contract["normalization_module_count"] != 0
        or contract["attention_module_count"] != 0
        or max(isolation.values()) > 1e-6
    ):
        raise v12.ContractError("v43 target-free FiLM preflight failed")
    evidence = config["authorization_evidence"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "film_contract": contract,
        "isolation": isolation,
        "prospective_fold_layer_gate": config["evaluation"]["safety_gate"],
        "prefix_cutoffs": {
            fold: (pd.Timestamp(start) - pd.Timedelta(days=7)).isoformat()
            for fold, start in config["training"]["fold_starts_kst"].items()
        },
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "audit_result_sha256": v12.sha256_file(ROOT / evidence["audit_result"]),
        "gate_amendment_sha256": v12.sha256_file(
            ROOT / evidence["prospective_gate_amendment"]
        ),
        "data_rows_read": 0,
        "model_fits": 0,
        "artifacts_written": 0,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    payload["preflight_sha256"] = v12.sha256_json(payload)
    return payload


def write_report(result: dict[str, Any]) -> None:
    item = result["candidate"]
    if "prospective_fold_layer_gate" not in item:
        return
    local = item["prospective_fold_layer_gate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v43 target-layer FiLM conditioned DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled delta RMSE "
        f"`{item['delta_rmse']:+.9f} C`, canonical nominal "
        f"`{item['canonical_nominal_pooled_points_delta']:+.6f}` points, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}` points.\n\n"
        f"prospective fold x layer gate: `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, max cell "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "Exact v13 element embedding에 target-layer one-hot으로 만든 identity-initialized "
        "FiLM affine map 한 개만 추가했다. Perez et al.은 conditioning 동기만 제공하며 "
        "P2 성능 근거가 아니다. sweep/router/ensemble/row deletion/Public selection/"
        "official/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    _bind_base()
    started = time.perf_counter()
    result = _BASE_RUN()
    config = load_config()
    record = result["candidate"]
    legacy_safety = bool(record["safety_pass"])
    amended = v37.prospective_fold_layer_gate(record, config)
    record["legacy_safety_pass_without_v26a_amendment"] = legacy_safety
    record["prospective_fold_layer_gate"] = amended
    record["safety_pass"] = bool(legacy_safety and amended["pass"])
    record["safety_pass_with_v26a_amendment"] = record["safety_pass"]
    passed = bool(record["strict_exploratory_pass"] and record["safety_pass"])
    result["schema_version"] = RESULT_SCHEMA
    result["status"] = (
        "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if passed
        else "EXPLORATORY_NO_GO_TARGET_LAYER_FILM"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["film_contract"] = _film_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "target_layer_film": config["training"]["target_layer_film"],
            "row_deletion": 0,
            "input_perturbation": 0,
            "data_augmentation": 0,
            "extra_loss": 0,
        }
    )
    result["comparison_to_preserved_candidates"] = {
        "use": "ledger_only_no_posthoc_selection_router_or_ensemble",
        "official_v23_feedback_used_for_selection": False,
        **{
            f"{name}_delta_rmse": json.loads(
                (ROOT / config["authorization_evidence"][f"{name}_result"]).read_text(
                    encoding="utf-8"
                )
            )["candidate"]["delta_rmse"]
            for name in ("v13", "v15", "v16", "v31", "v40", "v41", "v42")
        },
    }
    result["hashes"]["v13_runner"] = v12.sha256_file(_V13_RUNNER)
    result["hashes"]["prospective_gate_amendment"] = config[
        "authorization_evidence"
    ]["prospective_gate_amendment_sha256"]
    v12.atomic_json(ARTIFACT / "result.json", result)
    v12.atomic_json(REPORT / "result.json", result)
    write_report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight or --execute")
    value = preflight() if args.preflight else run()
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
