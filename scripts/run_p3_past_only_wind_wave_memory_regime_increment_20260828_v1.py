"""Run the sealed P3 past-only wind-wave memory one-shot diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.corrected_repeated_forward import (  # noqa: E402
    build_corrected_repeated_forward_folds,
    paired_case_bootstrap,
)
from p3_wave.models import (  # noqa: E402
    ResidualRegressor,
    compact_feature_columns,
    threshold_case_weights,
)
from p3_wave.revin_patch import assign_storm_episodes_from_wave  # noqa: E402
from p3_wave.validation import rmse  # noqa: E402
from p3_wave.wind_wave_memory import (  # noqa: E402
    MEMORY_FEATURES,
    build_wind_wave_memory_features,
)

EXPERIMENT_ID = "p3_past_only_wind_wave_memory_regime_increment_20260828_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
LEADS = (3, 6, 9, 12, 18, 24)
KEYS = ["fold", "anchor_id", "station", "lead_h"]
TARGET_COLUMNS = [f"target_{lead}" for lead in LEADS]


class ContractError(RuntimeError):
    """Raised when an immutable or truth-late contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_config(data_dir: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment ID drifted")
    policy = config["execution_policy"]
    if any(
        (
            policy["official_test_sample_submission_read_allowed"],
            policy["submission_csv_generation_allowed"],
            policy["official_upload_authorized"],
            policy["result_based_retry"],
        )
    ):
        raise ContractError("forbidden access or retry was enabled")
    for record in config["immutable_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ContractError(f"immutable input changed: {path}")
    wave_record = config["source_train_wave"]
    wave_path = data_dir / wave_record["filename"]
    if not wave_path.is_file():
        raise FileNotFoundError(wave_path)
    if wave_path.stat().st_size != int(wave_record["bytes"]) or sha256_file(wave_path) != wave_record["sha256"]:
        raise ContractError("immutable train_wave.csv changed")
    if int(config["model"]["maximum_fits"]) != 8:
        raise ContractError("fit budget drifted")
    return config


def build_feature_surface(
    config: dict[str, Any], output_directory: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, Any]]:
    paths = config["immutable_inputs"]
    features = pd.read_parquet(ROOT / paths["features"]["path"])
    anchor_path = ROOT / paths["anchors"]["path"]
    anchors = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", "station", "anchor_time", "current_hs"],
    )
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True)
    raw = np.load(ROOT / paths["raw_contexts"]["path"], mmap_mode="r")
    station_codes = np.load(ROOT / paths["station_codes"]["path"], mmap_mode="r")
    if raw.shape != (len(anchors), 289, 10) or station_codes.shape != (len(anchors),):
        raise ContractError("raw context cache shape drifted")
    if not np.array_equal(anchors["anchor_id"].to_numpy(dtype=np.int64), np.arange(len(anchors))):
        raise ContractError("anchor_id no longer equals raw cache row")
    # The immutable sequence cache is float32 while anchors retain CSV decimal precision.
    if not np.allclose(raw[:, -1, 0], anchors["current_hs"], rtol=0.0, atol=1e-6):
        raise ContractError("raw endpoint hs differs from anchor current_hs")
    mapping = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    expected_codes = anchors["station"].astype(str).map(mapping).to_numpy(dtype=np.int64)
    if not np.array_equal(np.asarray(station_codes, dtype=np.int64), expected_codes):
        raise ContractError("station code cache differs from anchors")
    memory = build_wind_wave_memory_features(raw, anchors["anchor_id"].to_numpy())
    atomic_parquet(output_directory / "memory_features.parquet", memory)
    features = features.merge(memory, on="anchor_id", how="left", validate="one_to_one")
    if features[list(MEMORY_FEATURES)].isna().all(axis=1).any():
        raise ContractError("a memory row has no usable value or mask")
    base_columns = compact_feature_columns(features.columns.tolist())
    if len(base_columns) != 591:
        raise ContractError(f"compact baseline surface drifted: {len(base_columns)}")
    enriched_columns = [*base_columns, *MEMORY_FEATURES]
    receipt = {
        "rows": int(len(features)),
        "base_feature_count": len(base_columns),
        "enriched_feature_count": len(enriched_columns),
        "memory_missing_fraction": {
            name: float(features[name].isna().mean()) for name in MEMORY_FEATURES[:10]
        },
        "official_rows_read": 0,
    }
    return features, anchors, base_columns, enriched_columns, receipt


def target_free_expansion(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    anchor_ids: np.ndarray,
    feature_columns: list[str],
    fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    blocks: list[pd.DataFrame] = []
    metadata: list[pd.DataFrame] = []
    for lead in LEADS:
        block = feature_lookup.loc[anchor_ids, feature_columns].reset_index(drop=True)
        station = feature_lookup.loc[anchor_ids, "station"].astype(str).reset_index(drop=True)
        current = anchor_lookup.loc[anchor_ids, "current_hs"].to_numpy(dtype=np.float64)
        block.insert(0, "station", station)
        block.insert(1, "lead_h", lead)
        block.insert(2, "current_hs_for_residual", current)
        blocks.append(block)
        metadata.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "anchor_id": anchor_ids,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True), pd.concat(metadata, ignore_index=True)


def load_targets_for_ids(anchor_path: Path, anchor_ids: np.ndarray) -> pd.DataFrame:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    frame = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", *TARGET_COLUMNS],
        filters=[("anchor_id", "in", ids.tolist())],
    )
    if len(frame) != len(ids) or set(frame["anchor_id"].astype(int)) != set(ids.tolist()):
        raise ContractError("target rows do not bind exactly to requested ids")
    return frame.set_index("anchor_id")


def target_vector(targets: pd.DataFrame, anchor_ids: np.ndarray, current: np.ndarray) -> np.ndarray:
    values = [targets.loc[anchor_ids, f"target_{lead}"].to_numpy(dtype=np.float64) for lead in LEADS]
    return np.concatenate(values) - np.asarray(current, dtype=np.float64)


def fit_pair(
    *,
    fold: str,
    seed: int,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    anchor_path: Path,
    base_columns: list[str],
    enriched_columns: list[str],
    model_config: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    base_train, train_metadata = target_free_expansion(
        features, anchors, train_ids, base_columns, fold
    )
    enriched_train, enriched_metadata = target_free_expansion(
        features, anchors, train_ids, enriched_columns, fold
    )
    if not train_metadata[KEYS].equals(enriched_metadata[KEYS]):
        raise ContractError("matched train expansion differs")
    targets = load_targets_for_ids(anchor_path, train_ids)
    y = target_vector(
        targets,
        train_ids,
        train_metadata["current_hs"].to_numpy(dtype=np.float64),
    )
    weights = threshold_case_weights(train_metadata["current_hs"].to_numpy(dtype=np.float64))
    params = {
        "iterations": int(model_config["iterations"]),
        "learning_rate": float(model_config["learning_rate"]),
        "depth": int(model_config["depth"]),
        "l2_leaf_reg": float(model_config["l2_leaf_reg"]),
        "random_strength": float(model_config["random_strength"]),
        "thread_count": int(model_config["thread_count"]),
    }
    base_model = ResidualRegressor("catboost", seed=seed, parameters=params).fit(
        base_train, y, sample_weight=weights
    )
    enriched_model = ResidualRegressor("catboost", seed=seed, parameters=params).fit(
        enriched_train, y, sample_weight=weights
    )
    base_validation, metadata = target_free_expansion(
        features, anchors, validation_ids, base_columns, fold
    )
    enriched_validation, second_metadata = target_free_expansion(
        features, anchors, validation_ids, enriched_columns, fold
    )
    if not metadata[KEYS].equals(second_metadata[KEYS]):
        raise ContractError("matched validation expansion differs")
    current = metadata["current_hs"].to_numpy(dtype=np.float64)
    metadata["base_prediction"] = current + base_model.predict_delta(base_validation)
    metadata["enriched_prediction"] = current + enriched_model.predict_delta(enriched_validation)
    return metadata, 2


def attach_truth(frame: pd.DataFrame, anchor_path: Path) -> pd.DataFrame:
    targets = load_targets_for_ids(anchor_path, frame["anchor_id"].unique())
    long = []
    for lead in LEADS:
        current = targets[[f"target_{lead}"]].rename(columns={f"target_{lead}": "target_hs"})
        current = current.reset_index()
        current["lead_h"] = lead
        long.append(current)
    truth = pd.concat(long, ignore_index=True)
    result = frame.merge(truth, on=["anchor_id", "lead_h"], how="left", validate="many_to_one")
    if result["target_hs"].isna().any():
        raise ContractError("validation truth binding failed")
    return result


def metric_breakdown(frame: pd.DataFrame, baseline: str, candidate: str) -> dict[str, Any]:
    def record(group: pd.DataFrame) -> dict[str, float | int]:
        old = rmse(group["target_hs"], group[baseline])
        new = rmse(group["target_hs"], group[candidate])
        return {"rows": int(len(group)), "baseline_rmse": old, "candidate_rmse": new, "delta_rmse": new - old}

    return {
        "aggregate": record(frame),
        "by_fold": {str(key): record(group) for key, group in frame.groupby("fold", sort=True)},
        "by_station": {str(key): record(group) for key, group in frame.groupby("station", sort=True)},
        "by_lead": {str(int(key)): record(group) for key, group in frame.groupby("lead_h", sort=True)},
        "by_station_lead": {
            f"{station}|{int(lead)}": record(group)
            for (station, lead), group in frame.groupby(["station", "lead_h"], sort=True)
        },
    }


def vector_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 20 or np.std(x[valid]) <= 1e-12 or np.std(y[valid]) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def axis_correlations(candidate: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    increment = candidate[KEYS + ["increment"]].copy()
    records: dict[str, tuple[pd.DataFrame, str, str]] = {}
    champion = pd.read_parquet(
        ROOT / config["immutable_inputs"]["champion_surface"]["path"],
        columns=[*KEYS, "o_prediction", "a_prediction"],
    )
    records["alpha"] = (champion, "a_prediction", "o_prediction")
    era5 = pd.read_parquet(ROOT / config["immutable_inputs"]["era5_axis"]["path"])
    records["era5"] = (era5, "transfer_prediction", "incumbent_prediction")
    kma = pd.read_parquet(ROOT / config["immutable_inputs"]["kma_axis"]["path"])
    records["kma"] = (kma, "candidate_final", "incumbent_final")
    spectral = pd.read_parquet(ROOT / config["immutable_inputs"]["spectral_axis"]["path"])
    spectral = spectral.loc[np.isclose(spectral["prefix_fraction"], 1.0)]
    records["spectral"] = (spectral, "challenger_prediction", "incumbent_prediction")
    energy = pd.read_parquet(ROOT / config["immutable_inputs"]["energy_axis"]["path"])
    energy = energy.loc[np.isclose(energy["prefix_fraction"], 1.0)]
    records["energy"] = (energy, "challenger_prediction", "incumbent_prediction")
    output: dict[str, float] = {}
    for name, (frame, candidate_column, baseline_column) in records.items():
        current = frame[KEYS + [candidate_column, baseline_column]].copy()
        current["axis"] = current[candidate_column] - current[baseline_column]
        merged = increment.merge(current[KEYS + ["axis"]], on=KEYS, how="inner", validate="one_to_one")
        active = merged["lead_h"].isin(config["candidate"]["active_leads"])
        output[name] = vector_correlation(merged.loc[active, "increment"], merged.loc[active, "axis"])
    return output


def run(config: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_directory = ROOT / config["artifact_directory"]
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    features, anchors, base_columns, enriched_columns, feature_receipt = build_feature_surface(
        config, output_directory
    )
    wave = pd.read_csv(data_dir / config["source_train_wave"]["filename"], usecols=["station", "time", "hs"])
    anchors_with_episodes = assign_storm_episodes_from_wave(anchors, wave)
    anchor_path = ROOT / config["immutable_inputs"]["anchors"]["path"]
    model_config = config["model"]
    fit_count = 0

    shadow_folds, shadow_selected, shadow_audit = build_corrected_repeated_forward_folds(
        anchors_with_episodes,
        windows=config["windows"]["shadow"],
        gap_hours=int(config["windows"]["gap_hours"]),
        footprint_hours=int(config["windows"]["footprint_hours"]),
    )
    shadow_support = {
        "cases": int(len(shadow_selected)),
        "by_station": {str(key): int(value) for key, value in shadow_selected.groupby("station").size().items()},
    }
    shadow_predictions: list[pd.DataFrame] = []
    for fold in shadow_folds:
        current, fits = fit_pair(
            fold=fold.name,
            seed=int(model_config["fold_seeds"][fold.name]),
            train_ids=fold.train_ids,
            validation_ids=fold.validation_ids,
            features=features,
            anchors=anchors,
            anchor_path=anchor_path,
            base_columns=base_columns,
            enriched_columns=enriched_columns,
            model_config=model_config,
        )
        fit_count += fits
        active = current["lead_h"].isin(config["candidate"]["active_leads"]).to_numpy()
        current["candidate_prediction"] = current["base_prediction"].to_numpy()
        current.loc[active, "candidate_prediction"] = current.loc[active, "base_prediction"] + float(
            config["candidate"]["increment_weight"]
        ) * (current.loc[active, "enriched_prediction"] - current.loc[active, "base_prediction"])
        shadow_predictions.append(current)
    shadow = pd.concat(shadow_predictions, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    atomic_parquet(output_directory / "shadow_predictions_sealed.parquet", shadow)
    atomic_json(
        output_directory / "shadow_prediction_seal.json",
        {
            "prediction_sha256": array_sha(shadow["candidate_prediction"]),
            "truth_rows_read_before_seal": 0,
            "fit_count": fit_count,
        },
    )
    shadow = attach_truth(shadow, anchor_path)
    shadow_long = shadow.loc[shadow["lead_h"].isin(config["candidate"]["active_leads"])].copy()
    shadow_metrics = metric_breakdown(shadow_long, "base_prediction", "candidate_prediction")
    shadow_gate_config = config["shadow_gate"]
    station_deltas = [value["delta_rmse"] for value in shadow_metrics["by_station"].values()]
    shadow_checks = {
        "minimum_cases": shadow_support["cases"] >= int(shadow_gate_config["minimum_cases"]),
        "minimum_cases_per_station": min(shadow_support["by_station"].values(), default=0)
        >= int(shadow_gate_config["minimum_cases_per_station"]),
        "longlead_improves": shadow_metrics["aggregate"]["delta_rmse"]
        < float(shadow_gate_config["longlead_delta_rmse_below"]),
        "station_regression_cap": max(station_deltas, default=float("inf"))
        <= float(shadow_gate_config["maximum_station_regression"]),
        "minimum_nonregressing_stations": sum(
            value <= float(shadow_gate_config["maximum_station_regression"])
            for value in station_deltas
        )
        >= int(shadow_gate_config["minimum_nonregressing_stations"]),
    }
    shadow_passed = all(shadow_checks.values())
    if not shadow_passed:
        result = {
            "schema_version": "p3.wind_wave_memory.result.v1",
            "experiment_id": EXPERIMENT_ID,
            "decision": "NO_GO_SHADOW_GATE",
            "feature_receipt": feature_receipt,
            "shadow_support": shadow_support,
            "shadow_audit": shadow_audit,
            "shadow_metrics": shadow_metrics,
            "shadow_gate_checks": shadow_checks,
            "fit_count": fit_count,
            "outer_truth_rows_read": 0,
            "official_test_sample_submission_rows_read": 0,
            "submission_generated_or_uploaded": False,
            "runtime": {"elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(), "catboost": catboost.__version__},
        }
        atomic_json(output_directory / "result.json", result)
        return result

    outer_folds, outer_selected, outer_audit = build_corrected_repeated_forward_folds(
        anchors_with_episodes,
        windows=config["windows"]["outer"],
        gap_hours=int(config["windows"]["gap_hours"]),
        footprint_hours=int(config["windows"]["footprint_hours"]),
    )
    pinned = pd.read_parquet(ROOT / config["immutable_inputs"]["validation_keys"]["path"])
    left = outer_selected[["fold", "anchor_id", "station", "episode_id"]].sort_values(["fold", "anchor_id"]).reset_index(drop=True)
    right = pinned[["fold", "anchor_id", "station", "episode_id"]].sort_values(["fold", "anchor_id"]).reset_index(drop=True)
    if not left.equals(right):
        raise ContractError("corrected exact-181 validation surface drifted")
    outer_predictions: list[pd.DataFrame] = []
    for fold in outer_folds:
        current, fits = fit_pair(
            fold=fold.name,
            seed=int(model_config["fold_seeds"][fold.name]),
            train_ids=fold.train_ids,
            validation_ids=fold.validation_ids,
            features=features,
            anchors=anchors,
            anchor_path=anchor_path,
            base_columns=base_columns,
            enriched_columns=enriched_columns,
            model_config=model_config,
        )
        fit_count += fits
        outer_predictions.append(current)
    if fit_count > int(model_config["maximum_fits"]):
        raise ContractError("fit budget exceeded")
    outer = pd.concat(outer_predictions, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    champion = pd.read_parquet(
        ROOT / config["immutable_inputs"]["champion_surface"]["path"],
        columns=[*KEYS, "current_hs", "champion_prediction"],
    ).sort_values(KEYS).reset_index(drop=True)
    outer = outer.merge(
        champion[KEYS + ["champion_prediction"]], on=KEYS, how="left", validate="one_to_one"
    )
    if outer["champion_prediction"].isna().any() or len(outer) != len(champion):
        raise ContractError("champion exact-181 key binding failed")
    active = outer["lead_h"].isin(config["candidate"]["active_leads"]).to_numpy()
    outer["increment"] = 0.0
    outer.loc[active, "increment"] = float(config["candidate"]["increment_weight"]) * (
        outer.loc[active, "enriched_prediction"] - outer.loc[active, "base_prediction"]
    )
    outer["candidate_prediction"] = outer["champion_prediction"] + outer["increment"]
    protected = outer["lead_h"].isin(config["candidate"]["protected_leads"])
    if not np.array_equal(
        outer.loc[protected, "candidate_prediction"].to_numpy(),
        outer.loc[protected, "champion_prediction"].to_numpy(),
    ):
        raise ContractError("protected early leads changed")
    atomic_parquet(output_directory / "outer_predictions_sealed.parquet", outer)
    atomic_json(
        output_directory / "outer_prediction_seal.json",
        {
            "rows": int(len(outer)),
            "candidate_sha256": array_sha(outer["candidate_prediction"]),
            "increment_sha256": array_sha(outer["increment"]),
            "truth_rows_read_before_seal": 0,
            "fit_count": fit_count,
        },
    )
    correlations = axis_correlations(outer, config)
    outer = attach_truth(outer, anchor_path)
    outer_metrics = metric_breakdown(outer, "champion_prediction", "candidate_prediction")
    bootstrap = paired_case_bootstrap(
        outer,
        candidate_column="candidate_prediction",
        baseline_column="champion_prediction",
        replicates=int(config["outer_gate"]["bootstrap_replicates"]),
        seed=int(config["outer_gate"]["bootstrap_seed"]),
    )
    gate = config["outer_gate"]
    fold_deltas = [value["delta_rmse"] for value in outer_metrics["by_fold"].values()]
    station_deltas = [value["delta_rmse"] for value in outer_metrics["by_station"].values()]
    station_lead_deltas = [value["delta_rmse"] for value in outer_metrics["by_station_lead"].values()]
    active_lead_deltas = [outer_metrics["by_lead"][str(lead)]["delta_rmse"] for lead in config["candidate"]["active_leads"]]
    outer_checks = {
        "pooled_delta": outer_metrics["aggregate"]["delta_rmse"]
        <= float(gate["pooled_delta_rmse_at_most"]),
        "bootstrap_ci": float(bootstrap["delta_candidate_minus_persistence_ci90_m"][1])
        < float(gate["bootstrap_ci90_upper_below"]),
        "improved_folds": sum(value < 0.0 for value in fold_deltas)
        >= int(gate["minimum_improved_folds"]),
        "improved_stations": sum(value < 0.0 for value in station_deltas)
        >= int(gate["minimum_improved_stations"]),
        "active_leads_non_degrade": max(active_lead_deltas, default=float("inf")) <= 0.0,
        "worst_station_lead": max(station_lead_deltas, default=float("inf"))
        <= float(gate["maximum_worst_station_lead_regression"]),
        "axis_independence": max(map(abs, correlations.values()), default=float("inf"))
        <= float(gate["maximum_absolute_axis_correlation"]),
        "protected_leads_exact": True,
    }
    result = {
        "schema_version": "p3.wind_wave_memory.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "GO_LOCAL_ONLY_NO_UPLOAD" if all(outer_checks.values()) else "NO_GO_OUTER_GATE",
        "feature_receipt": feature_receipt,
        "shadow_support": shadow_support,
        "shadow_audit": shadow_audit,
        "shadow_metrics": shadow_metrics,
        "shadow_gate_checks": shadow_checks,
        "outer_audit": outer_audit,
        "outer_metrics": outer_metrics,
        "bootstrap": bootstrap,
        "axis_correlations": correlations,
        "outer_gate_checks": outer_checks,
        "fit_count": fit_count,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "runtime": {"elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(), "catboost": catboost.__version__},
    }
    atomic_json(output_directory / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.check == args.execute:
        raise SystemExit("choose exactly one of --check or --execute")
    data_dir = args.data_dir.expanduser().resolve()
    config = load_config(data_dir)
    if args.check:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "PASS",
                    "catboost": catboost.__version__,
                    "official_rows_read": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(json.dumps(run(config, data_dir), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
