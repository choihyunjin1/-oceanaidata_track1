"""Run the sealed, aggregate-only P2 matched-budget local comparison.

Only ``observations.csv`` and historical OOF artifacts are accessible from this
program.  It has no code path for the official test index, sample submission,
submission candidates, or uploads.
"""

# ruff: noqa: E402 -- numerical thread limits must be set before NumPy imports.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "4"

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p2_restore.matched_budget_compare import (
    TARGET_LAYERS,
    build_bootstrap_plan,
    build_local_context,
    complementarity_report,
    materialize_settings,
    metric_report,
    paired_day_bootstrap,
    prepare_exact_frozen_surface,
    prepare_forward_surrogate_surface,
)
from p2_restore.public_layer_causal_residual import CausalResidualSpec

KST = ZoneInfo("Asia/Seoul")
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_matched_budget_local_compare_20260825_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "7723ce7b640da9d9eed499e046b1aa26e8a264c89047eb762dfa64c3753323bc"
)
FAMILY_ORDER = ("incumbent", "conservative_stack", "round_b")
FORBIDDEN_SOURCE_NAMES = {
    "test_index.csv",
    "sample_submission.csv",
    "baseline_interp.csv",
}


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object: {path}")
    return parsed


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _write_text_new(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip())
        handle.write("\n")


def _repo_file(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _artifact_dir(config: Mapping[str, Any]) -> Path:
    path = (PROJECT_ROOT / str(config["output"]["directory"])).resolve()
    if not path.is_relative_to(PROJECT_ROOT / "artifacts"):
        raise RuntimeError("output directory must remain under repository artifacts")
    return path


def _iter_repo_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    exact = config["surfaces"]["exact_frozen_lineage"]
    surrogate = config["surfaces"]["forward_causal_surrogate"]
    values: list[dict[str, Any]] = [
        dict(config["common_protocol"]),
        {
            "path": config["exact_prefix_refit_audit"]["audit_path"],
            "sha256": config["exact_prefix_refit_audit"]["audit_sha256"],
        },
        {
            "path": config["exact_prefix_refit_audit"]["seal_path"],
            "sha256": config["exact_prefix_refit_audit"]["seal_sha256"],
        },
        {"path": exact["path"], "sha256": exact["sha256"], "bytes": exact["bytes"]},
        {"path": exact["manifest_path"], "sha256": exact["manifest_sha256"]},
    ]
    values.extend(dict(value) for value in surrogate["oof"].values())
    values.extend(
        [
            dict(surrogate["architecture_manifest"]),
            dict(surrogate["training_recipe"]),
            dict(config["implementation_pins"]["profile_projection"]),
            dict(config["implementation_pins"]["causal_residual"]),
        ]
    )
    return values


def _validate_contract(config_path: Path, config: Mapping[str, Any]) -> None:
    if _sha256(config_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("sealed matched-budget config hash drift")
    if config.get("schema_version") != "p2_matched_budget_local_compare.v1":
        raise ValueError("unexpected schema version")
    if config.get("status") != "SEALED_BEFORE_NEW_MATCHED_COMPARISON_SCORE":
        raise ValueError("comparison is not sealed")
    if config.get("official_public_score_use") != "POST_HOC_INTERPRETATION_ONLY":
        raise ValueError("official-score policy changed")
    matched = config["matched_budget"]
    if int(matched["maximum_settings_per_family"]) != 3:
        raise ValueError("family setting budget changed")
    if int(matched["surrogate_seed_count_per_setting"]) != 3:
        raise ValueError("seed budget changed")
    if int(matched["new_model_fits"]) != 0:
        raise ValueError("new fitting is forbidden")
    if int(config["resource_policy"]["cpu_threads"]) != 4:
        raise ValueError("CPU coexistence policy changed")
    if int(config["metrics"]["bootstrap"]["replicates"]) != 5000:
        raise ValueError("bootstrap budget changed")
    for family in FAMILY_ORDER:
        settings = config["families"][family]["settings"]
        if not 1 <= len(settings) <= 3:
            raise ValueError(f"invalid setting count for {family}")
    output = config["output"]
    if not bool(output["aggregate_only"]) or bool(output["oof_predictions_written"]):
        raise ValueError("aggregate-only output contract changed")
    if int(output["submission_files_generated"]) or int(output["uploads"]):
        raise ValueError("external action contract changed")
    for spec in _iter_repo_inputs(config):
        lowered = Path(str(spec["path"])).name.lower()
        if lowered in FORBIDDEN_SOURCE_NAMES or "submission" in lowered:
            raise RuntimeError(f"forbidden input path in sealed config: {spec['path']}")


def _verify_repo_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for spec in _iter_repo_inputs(config):
        path = _repo_file(str(spec["path"]))
        digest = _sha256(path)
        if digest != str(spec["sha256"]):
            raise RuntimeError(f"immutable input drift: {spec['path']}")
        if "bytes" in spec and int(spec["bytes"]) != path.stat().st_size:
            raise RuntimeError(f"immutable input size drift: {spec['path']}")
        observed[str(spec["path"])] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return observed


def _source_files(data_dir: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    resolved = data_dir.resolve()
    observations = resolved / "observations.csv"
    readme = resolved / "README.md"
    for path in (observations, readme):
        if path.name.lower() in FORBIDDEN_SOURCE_NAMES or not path.is_file():
            raise FileNotFoundError(path)
    contract = config["data_contract"]
    if _sha256(observations) != str(contract["observations_csv_sha256"]):
        raise RuntimeError("observations.csv hash drift")
    if _sha256(readme) != str(contract["readme_sha256"]):
        raise RuntimeError("README.md hash drift")
    return {"observations.csv": observations, "README.md": readme}


def _causal_spec(config: Mapping[str, Any]) -> CausalResidualSpec:
    value = config["families"]["round_b"]["causal_residual_contract"]
    if value["jointly_masked_columns"] != ["temp", "psal"]:
        raise ValueError("joint public-layer masking contract changed")
    if value["required_anchors"] != [1, 5] or not value["coherent_sign_required"]:
        raise ValueError("causal residual gate contract changed")
    return CausalResidualSpec(
        public_layers=tuple(int(layer) for layer in value["public_layers"]),
        rolling_hours=int(value["rolling_hours"]),
        minimum_samples=int(value["minimum_samples"]),
        minimum_anchors=int(value["minimum_anchors"]),
        ridge_slope_lambda=float(value["ridge_slope_lambda"]),
        correction_scale=float(value["correction_scale"]),
        correction_clip_c=float(value["correction_clip_c"]),
        maximum_anchor_span_c=float(value["maximum_anchor_span_c"]),
    )


def _setting_order(config: Mapping[str, Any], family: str) -> list[str]:
    return [str(value["id"]) for value in config["families"][family]["settings"]]


def _all_settings(config: Mapping[str, Any]) -> list[str]:
    return [
        setting
        for family in FAMILY_ORDER
        for setting in _setting_order(config, family)
    ]


def _independent_primary(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    work = frame.loc[:, ["fold", "layer", "truth"]].copy()
    work["squared_error"] = (
        np.asarray(prediction, dtype=float) - work["truth"].to_numpy(float)
    ) ** 2
    cells = work.groupby(["fold", "layer"], sort=True)["squared_error"].mean()
    expected_cells = len(work["fold"].unique()) * len(TARGET_LAYERS)
    if len(cells) != expected_cells:
        raise RuntimeError("fold-layer metric surface is incomplete")
    return float(np.sqrt(cells.mean()))


def _family_summaries(
    config: Mapping[str, Any], metrics: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    incumbent_rmse = float(
        metrics["INCUMBENT_NOOP"]["fold_equal_layer_equal_rmse_c"]
    )
    summaries: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        order = _setting_order(config, family)
        default = str(config["families"][family]["default_setting"])
        ranked = [
            {
                "setting": setting,
                "order": index,
                "rmse_c": float(
                    metrics[setting]["fold_equal_layer_equal_rmse_c"]
                ),
            }
            for index, setting in enumerate(order)
        ]
        best = min(ranked, key=lambda item: (item["rmse_c"], item["order"]))
        default_rmse = float(metrics[default]["fold_equal_layer_equal_rmse_c"])
        summaries[family] = {
            "default_setting": default,
            "default_rmse_c": default_rmse,
            "best_setting_post_hoc_within_sealed_grid": best["setting"],
            "best_rmse_c": best["rmse_c"],
            "tuning_gain_c": default_rmse - float(best["rmse_c"]),
            "structure_gain_c": incumbent_rmse - default_rmse,
            "sealed_setting_scores": ranked,
        }
    return summaries


def _evaluate_panel(
    *,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    per_seed: Mapping[str, Sequence[np.ndarray]] | None,
    seed_labels: Sequence[str] | None,
    endpoints: pd.DataFrame,
) -> dict[str, Any]:
    settings = _all_settings(config)
    if set(settings) != set(predictions):
        raise RuntimeError("materialized setting surface differs from seal")
    metrics = {setting: metric_report(frame, predictions[setting]) for setting in settings}
    for setting in settings:
        independent = _independent_primary(frame, predictions[setting])
        reported = float(metrics[setting]["fold_equal_layer_equal_rmse_c"])
        if abs(independent - reported) > 1e-12:
            raise RuntimeError(f"independent metric mismatch: {setting}")

    bootstrap_config = config["metrics"]["bootstrap"]
    plan = build_bootstrap_plan(
        frame,
        replicates=int(bootstrap_config["replicates"]),
        seed=int(bootstrap_config["seed"]),
    )
    incumbent = predictions["INCUMBENT_NOOP"]
    bootstrap = {
        setting: paired_day_bootstrap(
            frame,
            incumbent,
            predictions[setting],
            plan,
            interval=float(bootstrap_config["interval"]),
        )
        for setting in settings
    }
    complementarity = {
        setting: complementarity_report(
            frame, incumbent, predictions[setting], endpoints
        )
        for setting in settings
    }
    if per_seed is None or seed_labels is None:
        seed_metrics: dict[str, Any] | None = None
    else:
        if any(len(per_seed[setting]) != len(seed_labels) for setting in settings):
            raise RuntimeError("per-seed prediction budget mismatch")
        seed_metrics = {
            setting: {
                label: metric_report(frame, prediction)
                for label, prediction in zip(
                    seed_labels, per_seed[setting], strict=True
                )
            }
            for setting in settings
        }
    return {
        "rows": int(len(frame)),
        "folds": sorted(frame["fold"].astype(str).unique().tolist()),
        "layers": sorted(frame["layer"].astype(int).unique().tolist()),
        "metrics_by_setting": metrics,
        "family_summaries": _family_summaries(config, metrics),
        "paired_day_bootstrap_vs_incumbent": bootstrap,
        "complementarity_vs_incumbent": complementarity,
        "per_seed_metrics": seed_metrics,
    }


def _load_exact_panel(
    config: Mapping[str, Any], context: Any, spec: CausalResidualSpec
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    surface = config["surfaces"]["exact_frozen_lineage"]
    oof = pd.read_parquet(_repo_file(str(surface["path"])))
    frame = prepare_exact_frozen_surface(oof, context)
    predictions, _, diagnostics = materialize_settings(
        frame, ("base_frozen",), context, spec
    )
    return frame, predictions, diagnostics


def _load_surrogate_panel(
    config: Mapping[str, Any], context: Any, spec: CausalResidualSpec
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, list[np.ndarray]],
    dict[str, Any],
]:
    surface = config["surfaces"]["forward_causal_surrogate"]
    seed_columns = tuple(str(value) for value in surface["seed_columns"])
    frames: list[pd.DataFrame] = []
    mean_parts: dict[str, list[np.ndarray]] = {
        setting: [] for setting in _all_settings(config)
    }
    seed_parts: dict[str, list[list[np.ndarray]]] = {
        setting: [[] for _ in seed_columns] for setting in _all_settings(config)
    }
    diagnostics: dict[str, Any] = {"by_prefix_fraction": {}}
    reference_keys: pd.DataFrame | None = None
    for key, input_spec in surface["oof"].items():
        oof = pd.read_csv(_repo_file(str(input_spec["path"])))
        frame = prepare_forward_surrogate_surface(oof, context, seed_columns)
        raw_mean = frame.loc[:, list(seed_columns)].mean(axis=1).to_numpy(float)
        stored_mean = frame["prediction_mean"].to_numpy(float)
        mean_reproduction_error = float(
            np.max(np.abs(raw_mean - stored_mean), initial=0.0)
        )
        if mean_reproduction_error > 1e-12:
            raise RuntimeError(f"stored seed mean mismatch at prefix {key}")
        keys = frame.loc[:, ["fold", "station", "layer", "_time_key"]]
        if reference_keys is None:
            reference_keys = keys.reset_index(drop=True)
        elif not reference_keys.equals(keys.reset_index(drop=True)):
            raise RuntimeError(f"prefix OOF key surface mismatch at {key}")
        fraction = float(int(key) / 100.0)
        frame["prefix_fraction"] = fraction
        means, per_seed, materialization = materialize_settings(
            frame, seed_columns, context, spec
        )
        for setting in mean_parts:
            mean_parts[setting].append(means[setting])
            for seed_index, prediction in enumerate(per_seed[setting]):
                seed_parts[setting][seed_index].append(prediction)
        diagnostics["by_prefix_fraction"][key] = {
            "fraction": fraction,
            "rows": int(len(frame)),
            "stored_prediction_mean_max_abs_error_c": mean_reproduction_error,
            **materialization,
        }
        frames.append(frame)
    combined_frame = pd.concat(frames, ignore_index=True)
    combined_means = {
        setting: np.concatenate(parts) for setting, parts in mean_parts.items()
    }
    combined_seeds = {
        setting: [np.concatenate(parts) for parts in by_seed]
        for setting, by_seed in seed_parts.items()
    }
    return combined_frame, combined_means, combined_seeds, diagnostics


def _per_prefix_metrics(
    frame: pd.DataFrame, predictions: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    fractions = frame["prefix_fraction"].to_numpy(float)
    for fraction in sorted(frame["prefix_fraction"].unique()):
        selected = np.isclose(fractions, float(fraction))
        part = frame.loc[selected].reset_index(drop=True)
        key = f"{int(round(float(fraction) * 100)):03d}"
        report[key] = {
            setting: metric_report(part, prediction[selected])
            for setting, prediction in predictions.items()
        }
    return report


def _output_file_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    }


def _render_report(result: Mapping[str, Any]) -> str:
    exact = result["panels"]["exact_frozen_lineage"]
    surrogate = result["panels"]["forward_causal_surrogate"]
    exact_stack = exact["family_summaries"]["conservative_stack"]
    exact_round_b = exact["family_summaries"]["round_b"]
    surrogate_stack = surrogate["family_summaries"]["conservative_stack"]
    surrogate_round_b = surrogate["family_summaries"]["round_b"]
    post_hoc = result.get("official_public_post_hoc")
    public_text = "제공하지 않음"
    if post_hoc is not None:
        public_text = (
            f"{post_hoc['official_public_rmse_c']:.6f}; exact frozen local과의 단순 차이 "
            f"{post_hoc['official_minus_exact_local_incumbent_c']:+.6f}℃"
        )
    return f"""# P2 matched-budget 로컬 비교 (2026-08-25 v1)

## 결론

이 비교는 **공유 가능하되 중요한 단서가 필요**합니다. 공식 incumbent와 동일한
causal prefix refit은 저장된 상태만으로 재구축할 수 없었습니다. 따라서 결과를
(1) 공식 incumbent 최종 구조가 적용된 기존 frozen OOF 진단면과 (2) 동일 3-seed,
동일 5개 training cutoff, 동일 3개 outer fold를 쓰는 time-safe surrogate 면으로
분리했습니다. 두 면 사이의 일치 여부가 구조효과의 핵심 증거이며, 어느 결과도
공식 Public 점수로 선택하거나 튜닝하지 않았습니다.

## 핵심 수치

| 비교면 | Incumbent RMSE | Round A 기본 구조효과 | Round A 봉인 grid 최선 | Round B 기본 구조효과 | Round B 봉인 grid 최선 |
|---|---:|---:|---|---:|---|
| exact frozen lineage | {exact['metrics_by_setting']['INCUMBENT_NOOP']['fold_equal_layer_equal_rmse_c']:.6f} | {exact_stack['structure_gain_c']:+.6f} | {exact_stack['best_setting_post_hoc_within_sealed_grid']} ({exact_stack['tuning_gain_c']:+.6f}) | {exact_round_b['structure_gain_c']:+.6f} | {exact_round_b['best_setting_post_hoc_within_sealed_grid']} ({exact_round_b['tuning_gain_c']:+.6f}) |
| forward causal surrogate | {surrogate['metrics_by_setting']['INCUMBENT_NOOP']['fold_equal_layer_equal_rmse_c']:.6f} | {surrogate_stack['structure_gain_c']:+.6f} | {surrogate_stack['best_setting_post_hoc_within_sealed_grid']} ({surrogate_stack['tuning_gain_c']:+.6f}) | {surrogate_round_b['structure_gain_c']:+.6f} | {surrogate_round_b['best_setting_post_hoc_within_sealed_grid']} ({surrogate_round_b['tuning_gain_c']:+.6f}) |

양수 구조효과는 incumbent 대비 RMSE 감소를 뜻합니다. `grid 최선`은 이미 봉인된
최대 3개 설정 안의 사후 진단값이며 신규 후보 선택 신호로 사용해서는 안 됩니다.

## 비교 설계

- primary: fold-equal/layer-equal RMSE. historical row-weighted pooled RMSE도 별도 기록.
- uncertainty: fold 안 KST calendar day paired bootstrap 5,000회, 동일 draw 공유.
- surrogate budget: 3 complete-pipeline seeds × 5 prefix fractions × 3 outer folds.
- 모든 arm에 동일 profile projection 적용. 신규 model fit은 0회.
- Round A 보간 기준선은 기존 Round A OOF와 byte-level 수치 정의가 같은 nominal-depth 보간.
- Round B는 joint TEMP/PSAL public-layer LOO, causal 24h median(min 72), L1/L5 포함 4-anchor,
  ridge λ=10, scale 0.25, ±0.125℃ clip, coherence gate를 고정 사용.

## 공식 점수와 로컬 점수

공식 Public RMSE: {public_text}. 이 값은 모든 local selection과 bootstrap을 끝낸 뒤
해석용으로만 붙였습니다. 서로 다른 표본·가중·lineage이므로 이 한 점의 차이를
보정계수로 쓰거나 로컬 순위가 공식 순위를 그대로 재현한다고 주장할 수 없습니다.

## 한계와 다음 단계

1. exact frozen lineage는 공식 incumbent 구조에는 정확하지만 causal forward refit이 아닙니다.
2. forward causal panel은 비교 예산은 정확히 맞지만 incumbent 자체의 surrogate입니다.
3. 공식 incumbent의 deep prefix mask API, May–Jun component OOF, 3-seed full pipeline,
   epoch/meta-refit semantics가 복구되기 전에는 “exact matched incumbent OOF”라 부를 수 없습니다.
4. 따라서 공식 제출 전 판단은 두 면에서 방향이 일치하고 bootstrap도 지지하는 고정 구조만
   보조 근거로 삼고, Public 한 점으로 재튜닝하지 않는 것이 타당합니다.
"""


def run(
    config_path: Path,
    data_dir: Path,
    *,
    execute: bool,
    post_hoc_public_rmse: float | None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    _validate_contract(config_path, config)
    repo_inputs = _verify_repo_inputs(config)
    sources = _source_files(data_dir, config)
    source_manifest = {
        name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name, path in sources.items()
    }
    preflight = {
        "status": "PASS",
        "config_sha256": _sha256(config_path),
        "repo_input_count": len(repo_inputs),
        "source_files_readable": sorted(sources),
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
        "uploads": 0,
    }
    if not execute:
        return preflight

    runner_path = Path(__file__).resolve()
    module_path = PROJECT_ROOT / "src/p2_restore/matched_budget_compare.py"
    output_dir = _artifact_dir(config)
    seal_path = output_dir / "preregistration_seal.json"
    technical_resume_path: Path | None = None
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir()}
        if existing != {seal_path.name}:
            raise FileExistsError(
                f"append-only output is not an eligible technical resume: {existing}"
            )
        prior_seal = _read_json(seal_path)
        if prior_seal.get("status") != "SEALED_BEFORE_OOF_SCORE_READ":
            raise RuntimeError("prior preregistration seal is invalid")
        if prior_seal.get("config", {}).get("sha256") != _sha256(config_path):
            raise RuntimeError("prior seal config differs from current config")
        if prior_seal.get("common_protocol") != config["common_protocol"]:
            raise RuntimeError("prior seal common protocol differs")
        technical_resume_path = output_dir / "technical_resume_seal.json"
        technical_resume = {
            "schema_version": "p2_matched_budget_local_compare.technical_resume.v1",
            "created_at_kst": _now_kst(),
            "status": "SEALED_BEFORE_TECHNICAL_RESUME_SCORE_READ",
            "prior_seal_sha256": _sha256(seal_path),
            "prior_failure": {
                "type": "duplicate_surface_key_projection_dispatch",
                "stage": "post-bootstrap complementarity aggregation",
                "effect_on_candidate_or_grid": "NONE",
                "outputs_written_before_failure": [seal_path.name],
            },
            "config_sha256": _sha256(config_path),
            "common_protocol": config["common_protocol"],
            "runner_sha256": _sha256(runner_path),
            "module_sha256": _sha256(module_path),
            "correction_scope": (
                "Dispatch already-sealed profile projection independently by "
                "prefix fraction when aggregate rows repeat station/time/layer keys."
            ),
            "candidate_or_grid_changes_after_first_score": 0,
            "official_public_score_available_to_selection": False,
        }
        _write_json_new(technical_resume_path, technical_resume)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        seal = {
            "schema_version": "p2_matched_budget_local_compare.seal.v1",
            "created_at_kst": _now_kst(),
            "status": "SEALED_BEFORE_OOF_SCORE_READ",
            "config": {
                "path": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(config_path),
            },
            "common_protocol": config["common_protocol"],
            "runner_sha256": _sha256(runner_path),
            "module_sha256": _sha256(module_path),
            "candidate_or_grid_changes_after_first_score": 0,
            "official_public_score_available_to_selection": False,
        }
        _write_json_new(seal_path, seal)

    started_at = _now_kst()
    observations = pd.read_csv(sources["observations.csv"])
    spec = _causal_spec(config)
    context = build_local_context(observations, spec)

    exact_frame, exact_predictions, exact_diagnostics = _load_exact_panel(
        config, context, spec
    )
    exact_result = _evaluate_panel(
        config=config,
        frame=exact_frame,
        predictions=exact_predictions,
        per_seed=None,
        seed_labels=None,
        endpoints=context.endpoints,
    )
    exact_result.update(
        {
            "role": config["surfaces"]["exact_frozen_lineage"]["role"],
            "exact_official_incumbent_architecture": True,
            "causal_forward_unbiased": False,
            "seed_budget": "UNAVAILABLE_FOR_FROZEN_LINEAGE",
            "materialization_diagnostics": exact_diagnostics,
        }
    )

    surrogate_frame, surrogate_predictions, surrogate_seeds, surrogate_diagnostics = (
        _load_surrogate_panel(config, context, spec)
    )
    seed_labels = tuple(
        str(value)
        for value in config["surfaces"]["forward_causal_surrogate"]["seed_columns"]
    )
    surrogate_result = _evaluate_panel(
        config=config,
        frame=surrogate_frame,
        predictions=surrogate_predictions,
        per_seed=surrogate_seeds,
        seed_labels=seed_labels,
        endpoints=context.endpoints,
    )
    surrogate_result.update(
        {
            "role": config["surfaces"]["forward_causal_surrogate"]["role"],
            "exact_official_incumbent_architecture": False,
            "causal_forward_unbiased": True,
            "seed_budget": list(seed_labels),
            "prefix_fractions": config["surfaces"]["forward_causal_surrogate"][
                "prefix_fractions"
            ],
            "metrics_by_prefix_fraction": _per_prefix_metrics(
                surrogate_frame, surrogate_predictions
            ),
            "materialization_diagnostics": surrogate_diagnostics,
        }
    )

    result: dict[str, Any] = {
        "schema_version": "p2_matched_budget_local_compare.result.v1",
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE_SHARE_WITH_CAVEATS",
        "started_at_kst": started_at,
        "completed_at_kst": _now_kst(),
        "primary_metric": config["metrics"]["primary"],
        "exact_prefix_refit_verdict": config["exact_prefix_refit_audit"]["verdict"],
        "panels": {
            "exact_frozen_lineage": exact_result,
            "forward_causal_surrogate": surrogate_result,
        },
        "selection_policy": {
            "candidate_grid_sealed_before_score": True,
            "official_public_score_used_for_selection": False,
            "official_public_score_used_for_tuning": False,
            "new_model_fits": 0,
        },
        "limitations": [
            "Exact official-incumbent same-prefix causal refit is not currently reproducible.",
            "Exact frozen lineage is architecture-exact but not causal-forward unbiased.",
            "Forward causal surface is budget-matched but is an incumbent surrogate.",
        ],
    }
    if post_hoc_public_rmse is not None:
        if not np.isfinite(post_hoc_public_rmse) or post_hoc_public_rmse <= 0.0:
            raise ValueError("post-hoc Public RMSE must be finite and positive")
        exact_local = float(
            exact_result["metrics_by_setting"]["INCUMBENT_NOOP"][
                "fold_equal_layer_equal_rmse_c"
            ]
        )
        result["official_public_post_hoc"] = {
            "source": "user-provided prior leaderboard observation",
            "official_public_rmse_c": float(post_hoc_public_rmse),
            "exact_local_incumbent_rmse_c": exact_local,
            "official_minus_exact_local_incumbent_c": float(
                post_hoc_public_rmse - exact_local
            ),
            "used_after_all_local_selection": True,
            "used_for_selection_or_tuning": False,
            "calibration_or_rank_equivalence_claimed": False,
        }

    metrics_path = output_dir / "metrics.json"
    comparison_path = output_dir / "comparison.json"
    qa_path = output_dir / "qa.json"
    result_path = output_dir / "result.json"
    report_path = output_dir / "report.md"
    metrics = {
        "schema_version": "p2_matched_budget_local_compare.metrics.v1",
        "primary_metric": config["metrics"]["primary"],
        "panels": result["panels"],
    }
    comparison = {
        "schema_version": "p2_matched_budget_local_compare.comparison.v1",
        "exact_prefix_refit_verdict": result["exact_prefix_refit_verdict"],
        "exact_frozen_lineage": {
            "family_summaries": exact_result["family_summaries"],
            "paired_day_bootstrap_vs_incumbent": exact_result[
                "paired_day_bootstrap_vs_incumbent"
            ],
            "complementarity_vs_incumbent": exact_result[
                "complementarity_vs_incumbent"
            ],
        },
        "forward_causal_surrogate": {
            "family_summaries": surrogate_result["family_summaries"],
            "paired_day_bootstrap_vs_incumbent": surrogate_result[
                "paired_day_bootstrap_vs_incumbent"
            ],
            "complementarity_vs_incumbent": surrogate_result[
                "complementarity_vs_incumbent"
            ],
        },
        "official_public_post_hoc": result.get("official_public_post_hoc"),
    }
    qa = {
        "schema_version": "p2_matched_budget_local_compare.qa.v1",
        "status": "PASS_WITH_MANDATORY_SURROGATE_CAVEAT",
        "config_hash_pass": _sha256(config_path) == EXPECTED_CONFIG_SHA256,
        "common_protocol_hash_pass": repo_inputs[
            str(config["common_protocol"]["path"])
        ]["sha256"]
        == str(config["common_protocol"]["sha256"]),
        "repo_input_hashes_pass": True,
        "source_hashes_pass": True,
        "historical_truth_only": True,
        "hidden_target_interval_overlap_rows": 0,
        "exact_independent_primary_recompute_pass": True,
        "surrogate_independent_primary_recompute_pass": True,
        "surrogate_seed_count": len(seed_labels),
        "surrogate_training_cutoff_count": len(
            config["surfaces"]["forward_causal_surrogate"]["prefix_fractions"]
        ),
        "surrogate_outer_fold_count": len(surrogate_result["folds"]),
        "candidate_or_grid_changes_after_first_score": 0,
        "new_model_fits": 0,
        "cpu_thread_limit": 4,
        "p3_era5_process_mutations": 0,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
        "submission_files_generated": 0,
        "uploads": 0,
        "exact_comparator_claim": "architecture-exact frozen-lineage diagnostic only",
        "surrogate_comparator_claim": "time-safe matched-budget surrogate, not exact incumbent",
    }
    _write_json_new(metrics_path, metrics)
    _write_json_new(comparison_path, comparison)
    _write_json_new(qa_path, qa)
    _write_json_new(result_path, result)
    _write_text_new(report_path, _render_report(result))

    output_paths = [
        seal_path,
        metrics_path,
        comparison_path,
        qa_path,
        result_path,
        report_path,
    ]
    if technical_resume_path is not None:
        output_paths.append(technical_resume_path)
    manifest = {
        "schema_version": "p2_matched_budget_local_compare.manifest.v1",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _now_kst(),
        "status": result["status"],
        "config": {
            "path": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": _sha256(config_path),
        },
        "common_protocol": config["common_protocol"],
        "exact_prefix_refit_audit": config["exact_prefix_refit_audit"],
        "implementation": {
            "runner": {
                "path": str(runner_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(runner_path),
            },
            "module": {
                "path": str(module_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(module_path),
            },
        },
        "repo_inputs": repo_inputs,
        "source_inputs": source_manifest,
        "outputs": _output_file_manifest(output_paths),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cpu_threads": 4,
        },
        "external_actions": {
            "official_test_reads": 0,
            "sample_submission_reads": 0,
            "submission_candidate_reads": 0,
            "submission_files_generated": 0,
            "uploads": 0,
            "p3_era5_process_mutations": 0,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_new(manifest_path, manifest)
    return {
        "status": result["status"],
        "output_dir": str(output_dir),
        "manifest_sha256": _sha256(manifest_path),
        "exact_incumbent_rmse_c": exact_result["metrics_by_setting"][
            "INCUMBENT_NOOP"
        ]["fold_equal_layer_equal_rmse_c"],
        "surrogate_incumbent_rmse_c": surrogate_result["metrics_by_setting"][
            "INCUMBENT_NOOP"
        ]["fold_equal_layer_equal_rmse_c"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--post-hoc-public-rmse", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir
    if data_dir is None:
        value = os.environ.get("P2_DATA_DIR")
        if not value:
            raise RuntimeError("set P2_DATA_DIR or pass --data-dir")
        data_dir = Path(value)
    result = run(
        args.config,
        data_dir,
        execute=bool(args.execute),
        post_hoc_public_rmse=args.post_hoc_public_rmse,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
