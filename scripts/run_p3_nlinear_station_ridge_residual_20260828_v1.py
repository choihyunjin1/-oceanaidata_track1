"""Run one nested P3 compact NLinear-style ridge residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p3_wave.data import LEADS, select_independent_validation
from p3_wave.nlinear_ridge_residual import (
    StandardizedStationRidge,
    absolute_prediction,
    build_compact_features,
    compact_feature_names,
    protected_long_blend,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_nlinear_station_ridge_residual_20260828_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
MODULE_PATH = ROOT / "src" / "p3_wave" / "nlinear_ridge_residual.py"
PAIR_KEYS = ["fold", "anchor_id", "station", "lead_h"]
TARGET_COLUMNS = [f"target_{lead}" for lead in LEADS]


class ContractError(RuntimeError):
    """Raised when an immutable experiment binding changes."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    if config["status"] != "PREREGISTERED_SINGLE_BOUNDED_ATTEMPT":
        raise ContractError("experiment is not preregistered")
    if config["blend"]["protected_leads_h"] != [3, 6, 9]:
        raise ContractError("protected lead contract changed")
    if config["blend"]["challenger_weight"] != 0.2:
        raise ContractError("challenger blend weight changed")
    if config["prohibitions"]["upload"] is not True:
        raise ContractError("upload prohibition changed")
    return config


def verify_inputs(config: dict[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for name, record in config["bindings"].items():
        path = ROOT / record["path"]
        if not path.is_file():
            raise ContractError(f"missing input: {path}")
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if "bytes" in record and int(record["bytes"]) != observed["bytes"]:
            raise ContractError(f"input byte size changed: {name}")
        if record["sha256"] != observed["sha256"]:
            raise ContractError(f"input hash changed: {name}")
        verified[name] = observed
    return verified


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(truth, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
                )
            )
        )
    )


def target_matrix(anchors: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    indexed = anchors.set_index("anchor_id")
    return indexed.loc[ids, TARGET_COLUMNS].to_numpy(dtype=np.float64)


def current_vector(anchors: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    return anchors.set_index("anchor_id").loc[ids, "current_hs"].to_numpy(dtype=np.float64)


def incumbent_matrix(incumbent: pd.DataFrame, fold: str, ids: np.ndarray) -> np.ndarray:
    frame = incumbent.loc[incumbent["fold"].eq(fold)]
    pivot = frame.pivot(index="anchor_id", columns="lead_h", values="incumbent_prediction")
    pivot = pivot.reindex(index=ids, columns=list(LEADS))
    values = pivot.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ContractError(f"incumbent matrix is incomplete for {fold}")
    return values


def select_alpha(
    features: np.ndarray,
    station: np.ndarray,
    anchors: pd.DataFrame,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray,
    alphas: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    targets = np.full((len(anchors), len(LEADS)), np.nan, dtype=np.float64)
    targets[fit_ids] = target_matrix(anchors, fit_ids) - current_vector(anchors, fit_ids)[:, None]
    truth = target_matrix(anchors, validation_ids)
    history: list[dict[str, float]] = []
    best_alpha = float(alphas[0])
    best_key = (float("inf"), -best_alpha)
    for alpha in alphas:
        model = StandardizedStationRidge.fit(
            features, station, targets, fit_ids, alpha=float(alpha)
        )
        delta = model.predict(features, station, validation_ids)
        prediction = absolute_prediction(current_vector(anchors, validation_ids), delta)
        score = rmse(truth, prediction)
        history.append({"alpha": float(alpha), "standalone_rmse_m": score})
        key = (score, -float(alpha))
        if key < best_key:
            best_key = key
            best_alpha = float(alpha)
    return best_alpha, history


def metric_slices(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in ("fold", "station", "lead_h"):
        values: dict[str, Any] = {}
        for name, group in frame.groupby(column, sort=True, observed=True):
            incumbent = rmse(group["target_hs"], group["incumbent_prediction"])
            candidate = rmse(group["target_hs"], group["candidate_prediction"])
            values[str(name)] = {
                "incumbent_rmse_m": incumbent,
                "candidate_rmse_m": candidate,
                "delta_m": candidate - incumbent,
            }
        result[f"by_{column}"] = values
    return result


def bootstrap_case_delta(frame: pd.DataFrame, *, replicates: int, seed: int) -> dict[str, Any]:
    case = (
        frame.groupby(["fold", "anchor_id"], sort=True, observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "incumbent_sse": float(
                        np.square(
                            group["target_hs"].to_numpy() - group["incumbent_prediction"].to_numpy()
                        ).sum()
                    ),
                    "candidate_sse": float(
                        np.square(
                            group["target_hs"].to_numpy() - group["candidate_prediction"].to_numpy()
                        ).sum()
                    ),
                    "rows": len(group),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(case), size=len(case))
        rows = float(case.iloc[sample]["rows"].sum())
        incumbent = np.sqrt(case.iloc[sample]["incumbent_sse"].sum() / rows)
        candidate = np.sqrt(case.iloc[sample]["candidate_sse"].sum() / rows)
        deltas[index] = candidate - incumbent
    return {
        "replicates": replicates,
        "cases": int(len(case)),
        "probability_improved": float(np.mean(deltas < 0.0)),
        "ci90_m": [float(value) for value in np.quantile(deltas, [0.05, 0.95])],
    }


def execute() -> dict[str, Any]:
    config = load_config()
    verified = verify_inputs(config)
    output = ROOT / config["artifact_dir"]
    if output.exists():
        raise FileExistsError("experiment namespace already exists")
    output.mkdir(parents=True)
    atomic_json(
        output / "ATTEMPT_LOCK.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(CONFIG_PATH),
            "official_test_read": False,
            "outer_result_read": False,
        },
    )

    sequence_path = ROOT / config["bindings"]["train_sequences"]["path"]
    station_path = ROOT / config["bindings"]["train_station"]["path"]
    anchor_path = ROOT / config["bindings"]["train_anchors"]["path"]
    incumbent_path = ROOT / config["bindings"]["frozen_incumbent_oof"]["path"]
    raw = np.load(sequence_path, mmap_mode="r")
    station = np.load(station_path, mmap_mode="r")
    anchors = pd.read_parquet(anchor_path)
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True)
    if not np.array_equal(anchors["anchor_id"].to_numpy(), np.arange(len(anchors))):
        raise ContractError("anchor ids no longer equal sequence row ids")
    if raw.shape != (len(anchors), 289, 10) or station.shape != (len(anchors),):
        raise ContractError("sequence cache shape changed")
    features = build_compact_features(raw)
    if features.shape != (len(anchors), len(compact_feature_names())):
        raise ContractError("compact feature shape changed")

    incumbent = pd.read_parquet(
        incumbent_path,
        columns=[
            "prefix_fraction",
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "incumbent_prediction",
        ],
    )
    incumbent = incumbent.loc[incumbent["prefix_fraction"].eq(1.0)].drop(columns="prefix_fraction")
    if incumbent.duplicated(PAIR_KEYS).any():
        raise ContractError("frozen incumbent keys are duplicated")
    validation_ids_by_fold = {
        str(name): np.sort(group["anchor_id"].unique().astype(np.int64))
        for name, group in incumbent.groupby("fold", observed=True)
    }
    windows = [tuple(item) for item in config["outer_validation"]["windows"]]
    expected_folds = [name for name, _, _ in windows]
    if set(validation_ids_by_fold) != set(expected_folds):
        raise ContractError("outer fold membership changed")

    blind_frames: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for fold_name, start_text, _ in windows:
        validation_ids = validation_ids_by_fold[fold_name]
        validation_start = pd.Timestamp(start_text, tz="UTC")
        train_end = validation_start - pd.Timedelta(
            hours=int(config["outer_validation"]["embargo_hours"])
        )
        train_ids = anchors.loc[anchors["anchor_time"].lt(train_end), "anchor_id"].to_numpy(
            dtype=np.int64
        )
        current_or_future = np.concatenate(
            [
                ids
                for name, ids in validation_ids_by_fold.items()
                if expected_folds.index(name) >= expected_folds.index(fold_name)
            ]
        )
        if np.intersect1d(train_ids, current_or_future).size:
            raise ContractError("outer train intersects current or future validation")
        outer_train = anchors.set_index("anchor_id").loc[train_ids]
        inner_end = outer_train["anchor_time"].max() + pd.Timedelta(minutes=20)
        inner_start = inner_end - pd.Timedelta(days=int(config["inner_selection"]["window_days"]))
        inner_ids = select_independent_validation(
            anchors,
            start=inner_start,
            end=inner_end,
            gap_hours=int(config["inner_selection"]["gap_hours"]),
        )
        inner_ids = np.intersect1d(inner_ids, train_ids)
        fit_end = inner_start - pd.Timedelta(hours=int(config["inner_selection"]["gap_hours"]))
        fit_ids = outer_train.loc[outer_train["anchor_time"].lt(fit_end)].index.to_numpy(
            dtype=np.int64
        )
        if len(fit_ids) == 0 or len(inner_ids) < 6:
            raise ContractError(f"insufficient nested cases for {fold_name}")

        best_alpha, inner_history = select_alpha(
            features,
            station,
            anchors,
            fit_ids,
            inner_ids,
            config["inner_selection"]["ridge_alphas"],
        )
        target_delta = np.full((len(anchors), len(LEADS)), np.nan, dtype=np.float64)
        target_delta[train_ids] = (
            target_matrix(anchors, train_ids) - current_vector(anchors, train_ids)[:, None]
        )
        model = StandardizedStationRidge.fit(
            features, station, target_delta, train_ids, alpha=best_alpha
        )
        predicted_delta = model.predict(features, station, validation_ids)
        challenger = absolute_prediction(current_vector(anchors, validation_ids), predicted_delta)
        base = incumbent_matrix(incumbent, fold_name, validation_ids)
        candidate = protected_long_blend(
            base,
            challenger,
            long_weight=float(config["blend"]["challenger_weight"]),
        )
        anchor_lookup = anchors.set_index("anchor_id")
        station_names = anchor_lookup.loc[validation_ids, "station"].astype(str).to_numpy()
        for case, anchor_id in enumerate(validation_ids):
            for lead_index, lead in enumerate(LEADS):
                blind_frames.append(
                    pd.DataFrame(
                        {
                            "fold": [fold_name],
                            "anchor_id": [int(anchor_id)],
                            "station": [station_names[case]],
                            "lead_h": [int(lead)],
                            "incumbent_prediction": [float(base[case, lead_index])],
                            "challenger_prediction": [float(challenger[case, lead_index])],
                            "candidate_prediction": [float(candidate[case, lead_index])],
                        }
                    )
                )
        fold_records.append(
            {
                "fold": fold_name,
                "outer_train_cases": int(len(train_ids)),
                "inner_fit_cases": int(len(fit_ids)),
                "inner_validation_cases": int(len(inner_ids)),
                "outer_validation_cases": int(len(validation_ids)),
                "selected_alpha": best_alpha,
                "inner_history": inner_history,
            }
        )

    blind = pd.concat(blind_frames, ignore_index=True)
    if blind.duplicated(PAIR_KEYS).any() or len(blind) != 181 * len(LEADS):
        raise ContractError("blind prediction key contract changed")
    blind_path = output / "sealed_outer_predictions.parquet"
    blind.to_parquet(blind_path, index=False)
    seal = {
        "experiment_id": EXPERIMENT_ID,
        "blind_prediction_sha256": sha256_file(blind_path),
        "outer_truth_read_before_seal": False,
        "official_test_read": False,
    }
    atomic_json(output / "PREDICTION_SEAL.json", seal)

    target_long = anchors.melt(
        id_vars=["anchor_id"],
        value_vars=TARGET_COLUMNS,
        var_name="lead_name",
        value_name="target_hs",
    )
    target_long["lead_h"] = (
        target_long["lead_name"].str.replace("target_", "", regex=False).astype(int)
    )
    evaluated = blind.merge(
        target_long[["anchor_id", "lead_h", "target_hs"]],
        on=["anchor_id", "lead_h"],
        how="left",
        validate="one_to_one",
    )
    if not np.isfinite(evaluated["target_hs"]).all():
        raise ContractError("outer target merge is incomplete")
    evaluated_path = output / "evaluated_outer_predictions.parquet"
    evaluated.to_parquet(evaluated_path, index=False)
    incumbent_rmse = rmse(evaluated["target_hs"], evaluated["incumbent_prediction"])
    challenger_rmse = rmse(evaluated["target_hs"], evaluated["challenger_prediction"])
    candidate_rmse = rmse(evaluated["target_hs"], evaluated["candidate_prediction"])
    slices = metric_slices(evaluated)
    bootstrap = bootstrap_case_delta(
        evaluated,
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    fold_deltas = [value["delta_m"] for value in slices["by_fold"].values()]
    station_deltas = [value["delta_m"] for value in slices["by_station"].values()]
    long_deltas = [slices["by_lead_h"][str(lead)]["delta_m"] for lead in (12, 18, 24)]
    gate = config["evaluation"]["gate"]
    gates = {
        "pooled_delta_below_zero": candidate_rmse - incumbent_rmse < 0.0,
        "minimum_improved_folds": sum(value < 0.0 for value in fold_deltas)
        >= int(gate["minimum_improved_folds"]),
        "minimum_probability_improved": bootstrap["probability_improved"]
        >= float(gate["minimum_probability_improved"]),
        "maximum_station_degradation": max(station_deltas)
        <= float(gate["maximum_station_degradation_m"]),
        "maximum_long_lead_degradation": max(long_deltas)
        <= float(gate["maximum_long_lead_degradation_m"]),
    }
    information_go = all(gates.values())
    result = {
        "schema_version": "p3.nlinear_station_ridge_residual.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "GO_LOCAL_DIRECTIONAL" if information_go else "TERMINAL_NO_GO",
        "claim_scope": config["claim"],
        "feature_count": len(compact_feature_names()),
        "fold_records": fold_records,
        "metrics": {
            "incumbent_rmse_m": incumbent_rmse,
            "challenger_rmse_m": challenger_rmse,
            "candidate_rmse_m": candidate_rmse,
            "pooled_delta_m": candidate_rmse - incumbent_rmse,
            **slices,
            "bootstrap": bootstrap,
        },
        "gates": {**gates, "information_go": information_go},
        "access": {
            "outer_truth_open_count": 1,
            "truth_first_used_after_prediction_seal": True,
            "official_test_rows_read": 0,
            "sample_submission_rows_read": 0,
            "candidate_created": False,
            "uploaded": False,
        },
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "module_sha256": sha256_file(MODULE_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "sealed_prediction_sha256": sha256_file(blind_path),
            "evaluated_prediction_sha256": sha256_file(evaluated_path),
        },
        "verified_inputs": verified,
    }
    atomic_json(output / "result.json", result)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    atomic_json(output / "manifest.json", manifest)
    return result


def check_only() -> dict[str, Any]:
    config = load_config()
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "READY",
        "verified_inputs": verify_inputs(config),
        "official_test_rows_read": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = execute() if args.execute else check_only()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
