"""Independent aggregate QA for the P1 matched-budget OOF comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/experiments/p1_matched_budget_local_compare_20260825_v1.json"
)
KEY_COLUMNS = ("station", "year", "layer", "time")
FAMILIES = (
    "incumbent_offline_xgboost",
    "causal_event_rescue_ensemble",
    "event_day_balanced_lightgbm",
)
KST = ZoneInfo("Asia/Seoul")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(path)
    return parsed


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _resolve(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _metrics(truth: Sequence[int], prediction: Sequence[int]) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    denominator = 2 * tp + fp + fn
    return {
        "f1": float(2 * tp / denominator if denominator else 0.0),
        "precision": float(tp / (tp + fp) if tp + fp else 0.0),
        "recall": float(tp / (tp + fn) if tp + fn else 0.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": tp + fp,
        "positive_support": tp + fn,
        "rows": len(y),
    }


def _assert_metrics_equal(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in ("f1", "precision", "recall"):
        if not np.isclose(float(observed[key]), float(expected[key]), rtol=0.0, atol=1e-14):
            raise RuntimeError(f"metric mismatch: {key}: {observed[key]} != {expected[key]}")
    for key in ("tp", "fp", "fn", "tn", "predicted_positive", "positive_support", "rows"):
        if int(observed[key]) != int(expected[key]):
            raise RuntimeError(f"count mismatch: {key}")


def _normal_fp(truth: np.ndarray, prediction: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    parsed = pd.to_datetime(frame["time"], utc=True, format="mixed", errors="raise")
    normal = truth == 0
    work = frame.loc[normal, ["station", "layer"]].reset_index(drop=True).copy()
    work["kst_day"] = parsed[normal].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d").to_numpy()
    false_positive = prediction[normal] == 1
    blocks = int(work.drop_duplicates(["station", "layer", "kst_day"]).shape[0])
    rows = int(false_positive.sum())
    days = int(
        work.loc[false_positive, ["station", "layer", "kst_day"]].drop_duplicates().shape[0]
    )
    return {
        "normal_rows": int(normal.sum()),
        "normal_station_layer_kst_days": blocks,
        "false_positive_rows": rows,
        "false_positive_rows_per_normal_station_layer_kst_day": rows / blocks,
        "normal_station_layer_kst_days_with_fp": days,
        "normal_station_layer_kst_day_fp_incidence": days / blocks,
    }


def _bootstrap(
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    day = pd.to_datetime(frame["time"], utc=True, format="mixed", errors="raise").dt.tz_convert(
        "Asia/Seoul"
    ).dt.strftime("%Y-%m-%d")
    codes, unique_days = pd.factorize(day, sort=True)
    block = np.zeros((len(unique_days), 6), dtype=np.int64)
    for code in range(len(unique_days)):
        mask = codes == code
        baseline_values = _metrics(truth[mask], baseline[mask])
        candidate_values = _metrics(truth[mask], candidate[mask])
        block[code] = (
            baseline_values["tp"],
            baseline_values["fp"],
            baseline_values["fn"],
            candidate_values["tp"],
            candidate_values["fp"],
            candidate_values["fn"],
        )
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=float)
    for index in range(replicates):
        values = block[rng.integers(0, len(block), size=len(block))].sum(axis=0)
        btp, bfp, bfn, ctp, cfp, cfn = values
        bden = 2 * btp + bfp + bfn
        cden = 2 * ctp + cfp + cfn
        delta[index] = (2 * ctp / cden if cden else 0.0) - (
            2 * btp / bden if bden else 0.0
        )
    quantiles = np.quantile(delta, [0.05, 0.5, 0.95])
    return {
        "unit": "KST calendar day",
        "days": len(unique_days),
        "replicates": replicates,
        "seed": seed,
        "delta_mean": float(delta.mean()),
        "delta_median": float(quantiles[1]),
        "delta_ci90": [float(quantiles[0]), float(quantiles[2])],
        "probability_delta_positive": float(np.mean(delta > 0)),
    }


def _assert_float_tree(observed: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, Mapping):
        if set(observed) != set(expected):
            raise RuntimeError(f"key mismatch at {path}")
        for key in expected:
            _assert_float_tree(observed[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(observed) != len(expected):
            raise RuntimeError(f"length mismatch at {path}")
        for index, value in enumerate(expected):
            _assert_float_tree(observed[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not np.isclose(float(observed), expected, rtol=0.0, atol=1e-14):
            raise RuntimeError(f"float mismatch at {path}: {observed} != {expected}")
    else:
        if observed != expected:
            raise RuntimeError(f"value mismatch at {path}: {observed} != {expected}")


def _setting_order(config: Mapping[str, Any], family: str) -> list[str]:
    source = family
    if family == "event_day_balanced_lightgbm":
        source = str(config["families"][family]["settings_from_family"])
    return [str(value["id"]) for value in config["families"][source]["settings"]]


def validate(config_path: Path) -> Path:
    config = _json(config_path)
    artifact = (PROJECT_ROOT / str(config["artifact_dir"])).resolve()
    qa_path = artifact / "independent_qa.json"
    final_manifest_path = artifact / "manifest_final.json"
    if qa_path.exists() or final_manifest_path.exists():
        raise FileExistsError("independent QA is append-only and already exists")

    preseal = _json(artifact / "preexecution_seal.json")
    result = _json(artifact / "result.json")
    manifest = _json(artifact / "manifest.json")
    recorded = _json(artifact / "metrics.json")
    predictions = pd.read_parquet(artifact / "predictions.parquet")
    incumbent_path = _resolve(str(config["inputs"]["incumbent_oof"]["path"]))
    incumbent = pd.read_parquet(incumbent_path)
    truth = incumbent["label"].to_numpy(dtype=np.int8)
    frame = incumbent.loc[:, [*KEY_COLUMNS, "fold"]].copy()

    checks: dict[str, Any] = {}
    checks["common_protocol_pin"] = (
        _sha256(_resolve(str(config["common_protocol"]["path"])))
        == config["common_protocol"]["sha256"]
        == result["common_protocol_sha256"]
        == manifest["common_protocol"]["sha256"]
    )
    checks["config_sealed"] = preseal["config_sha256"] == _sha256(config_path)
    checks["seal_verified_before_scoring"] = bool(result["seal_verified_before_scoring"])
    checks["exact_row_count"] = len(predictions) == int(config["surface"]["expected_rows"])
    checks["exact_key_order"] = predictions.loc[:, list(KEY_COLUMNS)].equals(
        incumbent.loc[:, list(KEY_COLUMNS)]
    )
    checks["fold_assignment_exact"] = predictions["fold"].astype(str).equals(
        incumbent["fold"].astype(str)
    )
    checks["duplicate_keys_zero"] = not predictions.duplicated(list(KEY_COLUMNS)).any()
    checks["operation_counters_zero"] = all(
        int(value) == 0 for value in result["operation_counters"].values()
    )
    checks["manifest_artifact_hashes"] = all(
        _sha256(_resolve(path)) == value["sha256"]
        and _resolve(path).stat().st_size == int(value["bytes"])
        for path, value in manifest["artifacts"].items()
    )
    checks["manifest_input_hashes"] = all(
        _sha256(_resolve(path)) == value["sha256"]
        for path, value in manifest["inputs"].items()
    )

    folds = list(config["surface"]["fold_order"])
    seeds = [int(value) for value in config["surface"]["seeds"]]
    numerical_checks = 0
    binary_columns = [column for column in predictions if column not in {*KEY_COLUMNS, "fold"}]
    checks["all_prediction_columns_binary"] = all(
        np.isin(predictions[column].to_numpy(), [0, 1]).all() for column in binary_columns
    )

    for family in FAMILIES:
        setting_order = _setting_order(config, family)
        default_setting = str(config["families"][family]["default_setting"])
        default_column = f"{family}__default"
        tuned_column = f"{family}__walk_forward_tuned"
        oracle_column = f"{family}__oracle_upper_bound"
        if not np.array_equal(
            predictions[default_column].to_numpy(dtype=np.int8),
            predictions[f"{family}__setting__{default_setting}"].to_numpy(dtype=np.int8),
        ):
            raise RuntimeError(f"default setting mismatch: {family}")

        reconstructed_tuned = np.zeros(len(frame), dtype=np.int8)
        reconstructed_tuned_seeds = {
            seed: np.zeros(len(frame), dtype=np.int8) for seed in seeds
        }
        for fold_index, fold in enumerate(folds):
            target = frame["fold"].eq(fold).to_numpy()
            if fold_index == 0:
                selected = default_setting
            else:
                calibration = frame["fold"].isin(folds[:fold_index]).to_numpy()
                scores = []
                for order, setting in enumerate(setting_order):
                    score = _metrics(
                        truth[calibration],
                        predictions.loc[
                            calibration, f"{family}__setting__{setting}"
                        ].to_numpy(dtype=np.int8),
                    )["f1"]
                    scores.append((score, -order, setting))
                selected = max(scores)[2]
            reconstructed_tuned[target] = predictions.loc[
                target, f"{family}__setting__{selected}"
            ].to_numpy(dtype=np.int8)
            for seed in seeds:
                reconstructed_tuned_seeds[seed][target] = predictions.loc[
                    target, f"{family}__setting__{selected}__seed_{seed}"
                ].to_numpy(dtype=np.int8)
        if not np.array_equal(
            reconstructed_tuned, predictions[tuned_column].to_numpy(dtype=np.int8)
        ):
            raise RuntimeError(f"walk-forward reconstruction mismatch: {family}")
        for seed in seeds:
            if not np.array_equal(
                reconstructed_tuned_seeds[seed],
                predictions[f"{tuned_column}__seed_{seed}"].to_numpy(dtype=np.int8),
            ):
                raise RuntimeError(f"seed walk-forward reconstruction mismatch: {family} {seed}")

        oracle_scores = []
        for order, setting in enumerate(setting_order):
            values = predictions[f"{family}__setting__{setting}"].to_numpy(dtype=np.int8)
            oracle_scores.append((_metrics(truth, values)["f1"], -order, setting))
        oracle_setting = max(oracle_scores)[2]
        if oracle_setting != recorded["families"][family]["selection"]["oracle_setting"]:
            raise RuntimeError(f"oracle setting mismatch: {family}")
        if not np.array_equal(
            predictions[oracle_column].to_numpy(dtype=np.int8),
            predictions[f"{family}__setting__{oracle_setting}"].to_numpy(dtype=np.int8),
        ):
            raise RuntimeError(f"oracle prediction mismatch: {family}")

        for mode in ("default", "walk_forward_tuned", "oracle_upper_bound"):
            prediction = predictions[f"{family}__{mode}"].to_numpy(dtype=np.int8)
            _assert_metrics_equal(
                _metrics(truth, prediction), recorded["families"][family][mode]["pooled"]
            )
            numerical_checks += 1
            for fold in folds:
                mask = frame["fold"].eq(fold).to_numpy()
                _assert_metrics_equal(
                    _metrics(truth[mask], prediction[mask]),
                    recorded["families"][family][mode]["by_fold"][fold],
                )
                numerical_checks += 1
            for station in sorted(frame["station"].astype(str).unique()):
                mask = frame["station"].astype(str).eq(station).to_numpy()
                _assert_metrics_equal(
                    _metrics(truth[mask], prediction[mask]),
                    recorded["families"][family][mode]["by_station"][station],
                )
                numerical_checks += 1
            observed_fp = _normal_fp(truth, prediction, frame)
            _assert_float_tree(
                observed_fp, recorded["families"][family][mode]["normal_fp_day"]
            )
            numerical_checks += 1
            seed_f1 = {
                str(seed): _metrics(
                    truth,
                    predictions[f"{family}__{mode}__seed_{seed}"].to_numpy(dtype=np.int8),
                )["f1"]
                for seed in seeds
            }
            _assert_float_tree(
                seed_f1,
                recorded["families"][family][mode]["seed_robustness"]["by_seed_f1"],
            )
            numerical_checks += 1

    incumbent_tuned = predictions[
        "incumbent_offline_xgboost__walk_forward_tuned"
    ].to_numpy(dtype=np.int8)
    bootstrap_config = config["metrics"]["bootstrap"]
    for family in FAMILIES:
        candidate = predictions[f"{family}__walk_forward_tuned"].to_numpy(dtype=np.int8)
        default = predictions[f"{family}__default"].to_numpy(dtype=np.int8)
        observed_vs_incumbent = _bootstrap(
            truth,
            candidate,
            incumbent_tuned,
            frame,
            replicates=int(bootstrap_config["replicates"]),
            seed=int(bootstrap_config["seed"]),
        )
        _assert_float_tree(
            observed_vs_incumbent,
            recorded["comparisons"][family]["paired_kst_day_bootstrap_vs_incumbent_tuned"],
        )
        observed_vs_default = _bootstrap(
            truth,
            candidate,
            default,
            frame,
            replicates=int(bootstrap_config["replicates"]),
            seed=int(bootstrap_config["seed"]) + 101,
        )
        _assert_float_tree(
            observed_vs_default,
            recorded["comparisons"][family]["paired_kst_day_bootstrap_tuned_vs_own_default"],
        )
        numerical_checks += 2

    checks["independent_numerical_checks"] = numerical_checks
    checks["all_passed"] = all(
        value if isinstance(value, bool) else int(value) > 0 for value in checks.values()
    )
    if not checks["all_passed"]:
        raise RuntimeError(f"independent QA failed: {checks}")

    qa = {
        "schema_version": "p1_matched_budget_local_compare.independent_qa.v1",
        "experiment_id": config["experiment_id"],
        "validated_at_kst": datetime.now(KST).isoformat(),
        "decision": "QA_PASS",
        "checks": checks,
        "finding_counts": {"P0": 0, "P1": 0},
        "caveats": [
            "Round A has a fixed single-seed causal component and three seed-specific incumbent components.",
            "The oracle column is a non-promotable diagnostic upper bound.",
            "No official test or leaderboard information was used in this validation.",
        ],
        "protected_action_counts": result["operation_counters"],
    }
    _write_json_new(qa_path, qa)
    artifact_files = sorted(
        path
        for path in artifact.iterdir()
        if path.is_file() and path.name != final_manifest_path.name
    )
    final_manifest = {
        "schema_version": "p1_matched_budget_local_compare.final_manifest.v1",
        "experiment_id": config["experiment_id"],
        "created_at_kst": datetime.now(KST).isoformat(),
        "qa_decision": "QA_PASS",
        "common_protocol_sha256": config["common_protocol"]["sha256"],
        "files": {
            str(path.relative_to(PROJECT_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_files
        },
    }
    _write_json_new(final_manifest_path, final_manifest)
    return qa_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError("config must remain in repository")
    output = validate(config_path)
    print(json.dumps({"status": "QA_PASS", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
