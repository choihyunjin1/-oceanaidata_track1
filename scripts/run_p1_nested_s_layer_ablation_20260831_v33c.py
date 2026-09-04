"""Exactly-once nested causal S-layer ablation of raw E150 additions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1 as e150_source  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v13 as truth_source  # noqa: E402

EXPERIMENT_ID = "p1_nested_s_layer_ablation_20260831_v33c"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
KEYS = ["station", "year", "layer", "time"]
FOLDS = ["2025_q2", "2025_q3", "2025_q4"]


class ContractError(RuntimeError):
    """Raised when the frozen v33c contract changes."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def resolve_source(spec: dict[str, str]) -> Path:
    path = (ROOT / spec["path"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ContractError(f"historical source unavailable: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise ContractError(f"historical source hash changed: {path}")
    return path


def load_contract() -> tuple[dict[str, Any], dict[str, Path]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    selector = config["selector"]
    checks = {
        "schema": config["schema_version"] == "p1.nested_s_layer_ablation.20260831.v33c",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "station_s": selector["station"] == "S-ORS",
        "support10": selector["minimum_layer_support"] == 10,
        "strict_rule": selector["marginal_precision_rule"] == "strictly_less_than_prefix_incumbent_f1_divided_by_2",
        "q2_abstain": selector["q2_action"] == "abstain",
        "q3_prefix": selector["q3_prefix_folds"] == [FOLDS[0]],
        "q4_prefix": selector["q4_prefix_folds"] == FOLDS[:2],
        "deployment_prefix": selector["deployment_prefix_folds"] == FOLDS,
        "fit0": selector["model_fits"] == 0,
        "grid0": selector["grid_size"] == 0,
        "retry0": selector["retries"] == 0,
        "folds": config["validation"]["folds"] == FOLDS,
        "primary": config["validation"]["primary_outer_folds"] == FOLDS[1:],
        "official0": config["authorization"]["official_reads"] == 0,
        "hidden0": config["authorization"]["hidden_truth_reads"] == 0,
        "csv0": config["authorization"]["submission_csv_created"] == 0,
        "upload0": config["authorization"]["uploads"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"v33c config contract failed: {checks}")
    paths = {name: resolve_source(config["historical_sources"][name]) for name in ["anchor", "truth", *FOLDS]}
    return config, paths


def load_and_seal_raw(anchor_frame: pd.DataFrame, paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    incumbent = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    raw_e150 = np.empty(len(anchor_frame), dtype=np.int8)
    receipts: list[dict[str, Any]] = []
    for fold in FOLDS:
        mask = anchor_frame["fold"].eq(fold).to_numpy()
        archive = e150_source._select_archive_arrays(fold, paths[fold])
        raw = np.asarray(archive["candidate"], dtype=np.int8)
        if raw.shape != (int(mask.sum()),) or not np.isin(raw, [0, 1]).all():
            raise ContractError(f"raw E150 fold alignment failed: {fold}")
        raw_e150[mask] = raw
        receipts.append({"fold": fold, "rows": int(mask.sum()), "raw_e150_positives": int(raw.sum())})
    if not np.all(raw_e150[incumbent == 1] == 1):
        raise ContractError("raw E150 does not preserve incumbent positives")
    sealed_path = ARTIFACT / "raw-seal.npz"
    np.savez_compressed(sealed_path, incumbent=incumbent, raw_e150=raw_e150)
    seal = {
        "schema_version": "p1.nested_s_layer_ablation.raw_seal.v33c",
        "experiment_id": EXPERIMENT_ID,
        "rows": len(anchor_frame),
        "receipts": receipts,
        "incumbent_sha256": sha256_array(incumbent),
        "raw_e150_sha256": sha256_array(raw_e150),
        "sealed_npz_sha256": sha256_file(sealed_path),
        "truth_reads_before_raw_seal": 0,
        "official_reads": 0,
        "hidden_truth_reads": 0,
    }
    write_json_new(ARTIFACT / "raw-seal.json", seal)
    return incumbent, raw_e150, seal


def select_layers(
    frame: pd.DataFrame,
    incumbent: np.ndarray,
    raw_e150: np.ndarray,
    prefix_folds: list[str],
    minimum_support: int,
) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    prefix = frame["fold"].isin(prefix_folds).to_numpy()
    if not prefix.any():
        raise ContractError("selection prefix is empty")
    incumbent_f1 = float(f1_score(truth[prefix], incumbent[prefix]))
    cutoff = incumbent_f1 / 2.0
    s_station = frame["station"].astype(str).eq("S-ORS").to_numpy()
    addition = prefix & s_station & (incumbent == 0) & (raw_e150 == 1)
    layers = sorted(int(value) for value in frame.loc[s_station, "layer"].unique())
    stats: dict[str, dict[str, Any]] = {}
    selected: list[int] = []
    for layer in layers:
        mask = addition & frame["layer"].eq(layer).to_numpy()
        support = int(mask.sum())
        tp = int((mask & (truth == 1)).sum())
        fp = support - tp
        precision = tp / support if support else None
        choose = bool(support >= minimum_support and precision is not None and precision < cutoff)
        if choose:
            selected.append(layer)
        stats[str(layer)] = {
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "marginal_precision": precision,
            "selected": choose,
        }
    return {
        "prefix_folds": prefix_folds,
        "prefix_rows": int(prefix.sum()),
        "prefix_incumbent_f1": incumbent_f1,
        "precision_cutoff_incumbent_f1_divided_by_2": cutoff,
        "minimum_support": minimum_support,
        "selected_layers": selected,
        "layer_statistics": stats,
    }


def build_nested_candidate(
    frame: pd.DataFrame,
    incumbent: np.ndarray,
    raw_e150: np.ndarray,
    minimum_support: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]], dict[str, Any]]:
    candidate = raw_e150.copy()
    removal = np.zeros(len(frame), dtype=bool)
    selections: dict[str, dict[str, Any]] = {}
    rules = {
        "2025_q3": ["2025_q2"],
        "2025_q4": ["2025_q2", "2025_q3"],
    }
    station_s = frame["station"].astype(str).eq("S-ORS").to_numpy()
    for outer_fold, prefix_folds in rules.items():
        receipt = select_layers(frame, incumbent, raw_e150, prefix_folds, minimum_support)
        selected = set(receipt["selected_layers"])
        outer = frame["fold"].eq(outer_fold).to_numpy()
        remove = outer & station_s & frame["layer"].isin(selected).to_numpy() & (incumbent == 0) & (raw_e150 == 1)
        candidate[remove] = 0
        removal |= remove
        receipt["outer_fold"] = outer_fold
        receipt["outer_labels_used_for_selection"] = 0
        receipt["outer_removals"] = int(remove.sum())
        selections[outer_fold] = receipt
    deployment = select_layers(frame, incumbent, raw_e150, FOLDS, minimum_support)
    if np.any(removal & ~station_s) or np.any(candidate[incumbent == 1] == 0):
        raise ContractError("nested action violated station or incumbent preservation")
    if np.any((candidate != raw_e150) & ~removal) or np.any(candidate > raw_e150):
        raise ContractError("nested action is not an exact raw-E150 ablation")
    return candidate, removal, selections, deployment


def metric_block(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    removal: np.ndarray,
) -> dict[str, Any]:
    removed = mask & removal
    reference_f1 = float(f1_score(truth[mask], reference[mask]))
    candidate_f1 = float(f1_score(truth[mask], candidate[mask]))
    return {
        "rows": int(mask.sum()),
        "reference_f1": reference_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": candidate_f1 - reference_f1,
        "removed_additions": int(removed.sum()),
        "removed_true_positives": int((removed & (truth == 1)).sum()),
        "removed_false_positives": int((removed & (truth == 0)).sum()),
    }


def f1_counts(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray([
        int(((truth == 1) & (prediction == 1)).sum()),
        int(((truth == 0) & (prediction == 1)).sum()),
        int(((truth == 1) & (prediction == 0)).sum()),
    ], dtype=np.int64)


def f1_from_counts(counts: np.ndarray) -> float:
    tp, fp, fn = counts
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def day_block_bootstrap(
    frame: pd.DataFrame,
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    local_day = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    blocks: list[np.ndarray] = []
    block_frame = pd.DataFrame({"fold": frame["fold"], "day": local_day, "position": np.arange(len(frame))})
    for positions in block_frame.loc[mask].groupby(["fold", "day"], sort=True, observed=True).indices.values():
        index = np.flatnonzero(mask)[np.asarray(positions, dtype=np.int64)]
        blocks.append(np.concatenate([f1_counts(truth[index], reference[index]), f1_counts(truth[index], candidate[index])]))
    counts = np.asarray(blocks, dtype=np.int64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        total = counts[rng.integers(0, len(counts), size=len(counts))].sum(axis=0)
        deltas[index] = f1_from_counts(total[3:]) - f1_from_counts(total[:3])
    return {
        "blocks": len(blocks),
        "replicates": replicates,
        "mean_delta_f1": float(deltas.mean()),
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas > 0.0)),
    }


def evaluate(
    frame: pd.DataFrame,
    incumbent: np.ndarray,
    raw_e150: np.ndarray,
    candidate: np.ndarray,
    removal: np.ndarray,
    selections: dict[str, dict[str, Any]],
    deployment: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    by_fold = {
        fold: metric_block(truth, raw_e150, candidate, frame["fold"].eq(fold).to_numpy(), removal)
        for fold in FOLDS
    }
    mask_all = frame["fold"].isin(FOLDS).to_numpy()
    mask_primary = frame["fold"].isin(FOLDS[1:]).to_numpy()
    pooled_all = metric_block(truth, raw_e150, candidate, mask_all, removal)
    pooled_primary = metric_block(truth, raw_e150, candidate, mask_primary, removal)
    bootstrap = day_block_bootstrap(
        frame,
        raw_e150,
        candidate,
        mask_primary,
        int(config["validation"]["bootstrap_replicates"]),
        int(config["validation"]["bootstrap_seed"]),
    )
    changed = frame.loc[mask_primary & removal, ["layer", "fold", "time"]].copy()
    changed["day"] = pd.to_datetime(changed["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")

    def maximum_share(columns: list[str]) -> float:
        if not len(changed):
            return 0.0
        return float(changed.groupby(columns, observed=True).size().max() / len(changed))

    anchor_removals = int((removal & (incumbent == 1)).sum())
    gates = {
        "q3_nonnegative": by_fold["2025_q3"]["delta_f1"] >= 0.0,
        "q4_nonnegative": by_fold["2025_q4"]["delta_f1"] >= 0.0,
        "pooled_all_positive": pooled_all["delta_f1"] > 0.0,
        "pooled_q3_q4_positive": pooled_primary["delta_f1"] > 0.0,
        "q3_q4_day_block_ci90_low_positive": bootstrap["ci90_low"] > 0.0,
        "anchor_removals_zero": anchor_removals == 0,
    }
    return {
        "name": config["candidate"],
        "reference": "raw_E150",
        "fit_count": 0,
        "by_fold": by_fold,
        "pooled_q2_q3_q4": pooled_all,
        "pooled_q3_q4": pooled_primary,
        "day_block_bootstrap_q3_q4": bootstrap,
        "selected_layers_by_outer_fold": {
            fold: receipt["selected_layers"] for fold, receipt in selections.items()
        },
        "selection_receipts": selections,
        "full_deployment_selection": deployment,
        "removed_true_positives_q3_q4": pooled_primary["removed_true_positives"],
        "removed_false_positives_q3_q4": pooled_primary["removed_false_positives"],
        "anchor_removals": anchor_removals,
        "expected_points_delta": pooled_primary["delta_f1"] * float(config["score"]["points_per_f1"]),
        "concentration": {
            "layer_max_share": maximum_share(["layer"]),
            "fold_max_share": maximum_share(["fold"]),
            "day_max_share": maximum_share(["day"]),
            "unique_days": int(changed["day"].nunique()) if len(changed) else 0,
        },
        "gates": gates,
        "strict_pass": bool(all(gates.values())),
        "information_value_positive": bool(len(deployment["selected_layers"]) > 0 and pooled_primary["removed_additions"] > 0),
    }


def independent_qa(
    result: dict[str, Any],
    frame: pd.DataFrame,
    incumbent: np.ndarray,
    raw_e150: np.ndarray,
    candidate: np.ndarray,
    removal: np.ndarray,
) -> dict[str, Any]:
    q2 = frame["fold"].eq("2025_q2").to_numpy()
    station_s = frame["station"].astype(str).eq("S-ORS").to_numpy()
    checks = {
        "raw_sealed_before_truth": result["raw_seal"]["truth_reads_before_raw_seal"] == 0,
        "q2_abstained": int(removal[q2].sum()) == 0 and np.array_equal(candidate[q2], raw_e150[q2]),
        "q3_prefix_q2_only": result["candidate"]["selection_receipts"]["2025_q3"]["prefix_folds"] == ["2025_q2"],
        "q4_prefix_q2_q3_only": result["candidate"]["selection_receipts"]["2025_q4"]["prefix_folds"] == ["2025_q2", "2025_q3"],
        "deployment_prefix_all_historical": result["candidate"]["full_deployment_selection"]["prefix_folds"] == FOLDS,
        "support_floor_10": all(
            receipt["minimum_support"] == 10
            for receipt in [*result["candidate"]["selection_receipts"].values(), result["candidate"]["full_deployment_selection"]]
        ),
        "only_s_removed": not np.any(removal & ~station_s),
        "only_raw_additions_removed": not np.any(removal & ~((incumbent == 0) & (raw_e150 == 1))),
        "incumbent_preserved": not np.any((incumbent == 1) & (candidate == 0)),
        "g_i_unchanged": not np.any((candidate != raw_e150) & ~station_s),
        "fit_count_zero": result["fit_count"] == 0,
        "official_reads_zero": result["operations"]["official_reads"] == 0,
        "hidden_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
        "retry_zero": result["operations"]["retries"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def render_report(result: dict[str, Any], qa: dict[str, Any]) -> str:
    candidate = result["candidate"]
    fold_rows = []
    for fold in FOLDS:
        values = candidate["by_fold"][fold]
        fold_rows.append(
            f"| {fold} | {values['delta_f1']:.9f} | {values['removed_additions']} | "
            f"{values['removed_true_positives']} | {values['removed_false_positives']} |"
        )
    verdict = "PASS / materializer preflight 가능" if candidate["strict_pass"] else "TERMINAL NO_GO / 재시도 금지"
    return f"""# P1 nested S-layer ablation v33c

## 결론

**{verdict}**. Q2는 prior가 없어 abstain했고, Q3는 Q2만, Q4는 Q2+Q3만 사용해 제거 layer를 선택했다. 공식·hidden·test·sample 접근과 CSV·upload는 모두 0이다.

| fold | delta F1 vs raw E150 | removed | removed TP | removed FP |
|---|---:|---:|---:|---:|
{chr(10).join(fold_rows)}

- Q3+Q4 pooled delta F1: `{candidate['pooled_q3_q4']['delta_f1']:.9f}`
- Q3+Q4 day-block CI90: `[{candidate['day_block_bootstrap_q3_q4']['ci90_low']:.9f}, {candidate['day_block_bootstrap_q3_q4']['ci90_high']:.9f}]`
- 개선 확률: `{candidate['day_block_bootstrap_q3_q4']['probability_improved']:.6f}`
- 예상 점수 delta: `{candidate['expected_points_delta']:.9f}`
- Q3 선택 layer: `{candidate['selected_layers_by_outer_fold']['2025_q3']}`
- Q4 선택 layer: `{candidate['selected_layers_by_outer_fold']['2025_q4']}`
- full deployment 선택 layer: `{candidate['full_deployment_selection']['selected_layers']}`
- independent QA: `{qa['status']}`
- fit count: `0`; retry: `0`

Full deployment layer set은 Q2-Q4 historical prefix에 같은 support>=10 및 marginal precision < incumbent F1/2 규칙을 정확히 한 번 적용해 계산했다. 실제 공식 materialization은 수행하지 않았다.
"""


def execute() -> dict[str, Any]:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("v33c exactly-once namespace already exists")
    started = time.perf_counter()
    config, paths = load_contract()
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    write_json_new(ARTIFACT / "attempt_lock.json", {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "fit_count": 0,
        "retry_budget": 0,
        "official_reads": 0,
        "hidden_truth_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    })
    anchor_frame = pd.read_parquet(paths["anchor"], columns=[*KEYS, "fold", "current_router_prediction"])
    incumbent, raw_e150, raw_seal = load_and_seal_raw(anchor_frame, paths)
    historical, raw_e150 = truth_source.attach_truth(anchor_frame, raw_e150)
    candidate, removal, selections, deployment = build_nested_candidate(
        historical,
        incumbent,
        raw_e150,
        int(config["selector"]["minimum_layer_support"]),
    )
    action_path = ARTIFACT / "nested-action.npz"
    np.savez_compressed(action_path, candidate=candidate, removal=removal)
    record = evaluate(historical, incumbent, raw_e150, candidate, removal, selections, deployment, config)
    status = "PASS_MATERIALIZER_READY" if record["strict_pass"] else "TERMINAL_NO_GO"
    result = {
        "schema_version": "p1.nested_s_layer_ablation.result.v33c",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 0,
        "candidate": record,
        "raw_seal": raw_seal,
        "operations": {
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "test_reads": 0,
            "sample_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
            "retries": 0,
        },
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "attempt_lock_sha256": sha256_file(ARTIFACT / "attempt_lock.json"),
            "raw_seal_sha256": sha256_file(ARTIFACT / "raw-seal.json"),
            "raw_npz_sha256": sha256_file(ARTIFACT / "raw-seal.npz"),
            "nested_action_sha256": sha256_file(action_path),
            "candidate_array_sha256": sha256_array(candidate),
            "removal_array_sha256": sha256_array(removal.astype(np.uint8)),
        },
    }
    qa = independent_qa(result, historical, incumbent, raw_e150, candidate, removal)
    if qa["status"] != "PASS":
        result["status"] = "TERMINAL_QA_FAILURE"
    write_json_new(ARTIFACT / "result.json", result)
    write_json_new(REPORT / "result.json", result)
    write_json_new(REPORT / "independent-qa.json", qa)
    (REPORT / "report-source.md").write_text(render_report(result, qa), encoding="utf-8", newline="\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-once", action="store_true")
    args = parser.parse_args()
    if not args.execute_once:
        raise SystemExit("--execute-once required")
    try:
        result = execute()
    except Exception as exc:
        failure = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            write_json_new(ARTIFACT / "terminal_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
