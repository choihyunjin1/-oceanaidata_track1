"""Exactly-once historical test for an independent Ordered-CatBoost P1 model.

Only the immutable training table, its label-free offline feature cache, and
frozen historical OOF references are read.  Incumbent predictions are never
features and are loaded only after this candidate's fold predictions exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1 as e150  # noqa: E402

from p1_qc.config import load_config  # noqa: E402
from p1_qc.splits import outer_folds  # noqa: E402

EXPERIMENT_ID = "p1_ordered_catboost_eventday_20260831_v32a"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
ATTEMPT_LOCK = ARTIFACT_DIR / "attempt_lock.json"
RESULT_PATH = ARTIFACT_DIR / "result.json"
OOF_PATH = ARTIFACT_DIR / "historical_oof.parquet"
KEYS = ["station", "year", "layer", "time"]
FOLDS = ["2025_q2", "2025_q3", "2025_q4"]
PUBLIC_BEST_F1 = 0.833548
PUBLIC_BEST_POINTS = 28.909341
PUBLIC_SCORE_SLOPE = 26.578120867377286


class ContractError(RuntimeError):
    """Raised when a preregistered contract is violated."""


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def metric(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "f1": float(2 * tp / denominator) if denominator else 1.0,
    }


def event_day_weight(metadata: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    """Balance positive events and normal station-layer days without future rows."""
    y = np.asarray(target, dtype=np.int8)
    if y.ndim != 1 or len(metadata) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid event/day weighting inputs")
    work = metadata.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    work["position"] = np.arange(len(work), dtype=np.int64)
    work["target"] = y
    work["parsed_time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(["station", "layer", "parsed_time", "position"], kind="mergesort", inplace=True)
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["parsed_time"].diff().dt.total_seconds().eq(600)
    prior = grouped["target"].shift(1).fillna(0).eq(1)
    starts = work["target"].eq(1) & (~contiguous | ~prior)
    work["event"] = starts.cumsum().where(work["target"].eq(1), -1).astype(np.int64)
    positive = work["target"].eq(1)
    normal = ~positive
    if not positive.any() or not normal.any():
        raise ValueError("training prefix must contain both classes")
    event_length = work.loc[positive].groupby("event", sort=False)["event"].transform("size")
    positive_weight = 1.0 / np.sqrt(event_length.to_numpy(dtype=np.float64))
    positive_weight /= positive_weight.mean()
    day = work["parsed_time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    normal_length = (
        work.loc[normal]
        .assign(day=day.loc[normal])
        .groupby(["station", "layer", "day"], sort=False, observed=True)["day"]
        .transform("size")
    )
    normal_weight = 1.0 / np.sqrt(normal_length.to_numpy(dtype=np.float64))
    normal_weight /= normal_weight.mean()
    ordered = np.empty(len(work), dtype=np.float64)
    ordered[positive.to_numpy()] = positive_weight * math.sqrt(normal.sum() / positive.sum())
    ordered[normal.to_numpy()] = normal_weight
    work["weight"] = ordered
    result = work.sort_values("position", kind="mergesort")["weight"].to_numpy(np.float32)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ContractError("nonfinite event/day weights")
    return result


def block_codes(truth: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Return positive-event or normal-station-layer-day block codes."""
    y = np.asarray(truth, dtype=np.int8)
    work = metadata.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    work["position"] = np.arange(len(work), dtype=np.int64)
    work["truth"] = y
    work["parsed_time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    ordered = work.sort_values(["station", "layer", "parsed_time", "position"], kind="mergesort")
    grouped = ordered.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["parsed_time"].diff().dt.total_seconds().eq(600)
    prior = grouped["truth"].shift(1).fillna(0).eq(1)
    start = ordered["truth"].eq(1) & (~contiguous | ~prior)
    ordered["event"] = start.cumsum().where(ordered["truth"].eq(1), -1).astype(np.int64)
    day = ordered["parsed_time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    normal_key = (
        ordered["station"].astype(str)
        + "|"
        + ordered["layer"].astype(str)
        + "|"
        + day
    )
    keys = np.where(
        ordered["truth"].eq(1).to_numpy(),
        "event:" + ordered["event"].astype(str).to_numpy(),
        "normal:" + normal_key.to_numpy(),
    )
    ordered["block"] = pd.factorize(keys, sort=True)[0]
    restored = ordered.sort_values("position", kind="mergesort")["block"].to_numpy(np.int64)
    return restored, int(restored.max() + 1)


def paired_bootstrap(
    truth: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    metadata: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    codes, blocks = block_codes(truth, metadata)
    y = np.asarray(truth, dtype=np.int8)

    def confusion(prediction: np.ndarray) -> np.ndarray:
        p = np.asarray(prediction, dtype=np.int8)
        return np.column_stack(
            [
                np.bincount(codes, weights=((y == 1) & (p == 1)), minlength=blocks),
                np.bincount(codes, weights=((y == 0) & (p == 1)), minlength=blocks),
                np.bincount(codes, weights=((y == 1) & (p == 0)), minlength=blocks),
            ]
        )

    cand = confusion(candidate)
    ref = confusion(reference)
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 100):
        size = min(100, replicates - start)
        draws = rng.integers(0, blocks, size=(size, blocks))
        cand_total = cand[draws].sum(axis=1)
        ref_total = ref[draws].sum(axis=1)
        cand_f1 = 2 * cand_total[:, 0] / np.maximum(
            1.0, 2 * cand_total[:, 0] + cand_total[:, 1] + cand_total[:, 2]
        )
        ref_f1 = 2 * ref_total[:, 0] / np.maximum(
            1.0, 2 * ref_total[:, 0] + ref_total[:, 1] + ref_total[:, 2]
        )
        differences[start : start + size] = cand_f1 - ref_f1
    return {
        "replicates": replicates,
        "blocks": blocks,
        "difference_mean": float(differences.mean()),
        "difference_ci90": [
            float(np.quantile(differences, 0.05)),
            float(np.quantile(differences, 0.95)),
        ],
        "probability_improved": float(np.mean(differences > 0)),
    }


def load_train(data_dir: Path, expected_hash: str) -> pd.DataFrame:
    path = data_dir / "train.csv"
    if sha256_file(path) != expected_hash:
        raise ContractError("train.csv hash mismatch")
    frame = pd.read_csv(
        path,
        usecols=[*KEYS, "label"],
        dtype={"station": "string", "time": "string", "label": "int8"},
        low_memory=False,
    )
    if len(frame) != 776_706 or frame.duplicated(KEYS).any() or not np.isin(frame["label"], [0, 1]).all():
        raise ContractError("train.csv row/key/label contract failed")
    return frame


def load_features(config: dict[str, Any], rows: int) -> tuple[pd.DataFrame, list[int]]:
    path = ROOT / config["inputs"]["feature_cache"]
    if sha256_file(path) != config["inputs"]["feature_cache_sha256"]:
        raise ContractError("feature cache hash mismatch")
    frame = pd.read_parquet(path)
    if len(frame) != rows or frame.shape[1] != 80:
        raise ContractError("feature cache shape mismatch")
    categorical = [frame.columns.get_loc(name) for name in ("station", "layer_category", "depth_regime")]
    for position, column in enumerate(frame.columns):
        if position in categorical:
            frame[column] = frame[column].astype("string").fillna("<NA>").astype(str)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    return frame, categorical


def model_from_config(config: dict[str, Any]) -> CatBoostClassifier:
    model = config["model"]
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="F1",
        boosting_type=model["boosting_type"],
        iterations=int(model["iterations"]),
        learning_rate=float(model["learning_rate"]),
        depth=int(model["depth"]),
        l2_leaf_reg=float(model["l2_leaf_reg"]),
        random_strength=float(model["random_strength"]),
        bootstrap_type=model["bootstrap_type"],
        subsample=float(model["subsample"]),
        rsm=float(model["rsm"]),
        random_seed=int(config["seed"]),
        thread_count=int(model["thread_count"]),
        task_type="CPU",
        allow_writing_files=False,
        verbose=False,
    )


def aligned_references(oof: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load references only after independent candidate predictions are frozen."""
    tabular = oof["deployment_prediction"].to_numpy(np.int8)
    e150_prediction = np.full(len(oof), -1, dtype=np.int8)
    bundles = e150.load_bundles()
    for fold in FOLDS:
        bundle = bundles[fold]
        reference = bundle.frame[[*KEYS, "fold"]].copy()
        reference["e150_prediction"] = bundle.raw_candidate
        positions = oof.loc[oof["fold"].eq(fold), [*KEYS, "fold"]].merge(
            reference,
            on=[*KEYS, "fold"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if positions["e150_prediction"].isna().any():
            raise ContractError(f"E150 alignment failed for {fold}")
        e150_prediction[oof["fold"].eq(fold)] = positions["e150_prediction"].to_numpy(np.int8)
    if (e150_prediction < 0).any():
        raise ContractError("E150 reference incomplete")
    return tabular, e150_prediction


def execute() -> dict[str, Any]:
    started_wall = now_kst()
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if started_wall >= datetime.fromisoformat(config["deadline_kst"]):
        raise ContractError("deadline already passed")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_new(
        ATTEMPT_LOCK,
        {
            "experiment_id": EXPERIMENT_ID,
            "started_kst": started_wall.isoformat(),
            "config_sha256": sha256_file(CONFIG_PATH),
            "exactly_once": True,
        },
    )
    data_dir_raw = os.environ.get("P1_DATA_DIR")
    if not data_dir_raw:
        raise ContractError("P1_DATA_DIR is required")
    data_dir = Path(data_dir_raw).expanduser().resolve(strict=True)
    train = load_train(data_dir, config["inputs"]["train_csv_sha256"])
    features, categorical = load_features(config, len(train))
    p1_config = load_config(ROOT / "configs" / "p1.toml", env={"P1_DATA_DIR": str(data_dir)})
    folds = outer_folds(train, config=p1_config.splits, cadence_minutes=10)
    reference_path = ROOT / config["inputs"]["tabular_reference_oof"]
    if sha256_file(reference_path) != config["inputs"]["tabular_reference_oof_sha256"]:
        raise ContractError("historical OOF hash mismatch")
    reference_oof = pd.read_parquet(
        reference_path,
        columns=[*KEYS, "fold", "label", "deployment_prediction"],
    )
    reference_oof["candidate_probability"] = np.nan
    reference_oof["candidate_prediction"] = -1
    row_lookup = train.reset_index(names="train_position")[[*KEYS, "train_position"]]
    aligned = reference_oof[[*KEYS, "fold"]].merge(
        row_lookup, on=KEYS, how="left", validate="one_to_one", sort=False
    )
    if aligned["train_position"].isna().any():
        raise ContractError("OOF to training-row alignment failed")
    threshold = float(config["model"]["probability_threshold"])
    fit_records: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        if fold.name != FOLDS[fold_index]:
            raise ContractError("unexpected fold order")
        elapsed = time.perf_counter() - started
        if elapsed >= float(config["maximum_runtime_seconds"]):
            raise TimeoutError("runtime cap reached before all fits")
        train_idx = fold.train_idx
        val_rows = np.flatnonzero(reference_oof["fold"].eq(fold.name).to_numpy())
        val_idx = aligned.loc[val_rows, "train_position"].to_numpy(np.int64)
        if set(val_idx.tolist()) != set(fold.val_idx.tolist()):
            raise ContractError(f"validation membership mismatch: {fold.name}")
        weights = event_day_weight(train.iloc[train_idx], train.loc[train_idx, "label"].to_numpy(np.int8))
        model = model_from_config(config)
        train_pool = Pool(
            features.iloc[train_idx],
            label=train.loc[train_idx, "label"].to_numpy(np.int8),
            weight=weights,
            cat_features=categorical,
        )
        val_pool = Pool(features.iloc[val_idx], cat_features=categorical)
        fit_started = time.perf_counter()
        model.fit(train_pool)
        probability = model.predict_proba(val_pool)[:, 1]
        prediction = (probability >= threshold).astype(np.int8)
        reference_oof.loc[val_rows, "candidate_probability"] = probability
        reference_oof.loc[val_rows, "candidate_prediction"] = prediction
        fit_records.append(
            {
                "fold": fold.name,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "fit_seconds": float(time.perf_counter() - fit_started),
                "positive_rate": float(prediction.mean()),
                "iterations": int(model.tree_count_),
            }
        )
        print(json.dumps({"fold_complete": fold.name, **fit_records[-1]}, sort_keys=True), flush=True)
    if reference_oof["candidate_probability"].isna().any() or (reference_oof["candidate_prediction"] < 0).any():
        raise ContractError("candidate OOF is incomplete")
    if not np.array_equal(reference_oof["label"].to_numpy(np.int8), train.loc[aligned["train_position"], "label"].to_numpy(np.int8)):
        raise ContractError("historical labels differ after alignment")
    # Candidate predictions are now complete and immutable in memory; references are loaded afterward.
    tabular, e150_prediction = aligned_references(reference_oof)
    truth = reference_oof["label"].to_numpy(np.int8)
    candidate = reference_oof["candidate_prediction"].to_numpy(np.int8)
    by_fold: dict[str, Any] = {}
    for fold_index, fold in enumerate(FOLDS):
        mask = reference_oof["fold"].eq(fold).to_numpy()
        candidate_metric = metric(truth[mask], candidate[mask])
        tabular_metric = metric(truth[mask], tabular[mask])
        e150_metric = metric(truth[mask], e150_prediction[mask])
        bootstrap = paired_bootstrap(
            truth[mask],
            candidate[mask],
            tabular[mask],
            reference_oof.loc[mask, ["station", "layer", "time"]].reset_index(drop=True),
            replicates=int(config["validation"]["bootstrap_replicates"]),
            seed=int(config["seed"]) + fold_index,
        )
        by_fold[fold] = {
            "candidate": candidate_metric,
            "tabular_reference": tabular_metric,
            "e150_reference": e150_metric,
            "delta_f1_vs_tabular": float(candidate_metric["f1"] - tabular_metric["f1"]),
            "delta_f1_vs_e150": float(candidate_metric["f1"] - e150_metric["f1"]),
            "bootstrap_vs_tabular": bootstrap,
        }
    metadata = reference_oof[["station", "layer", "time"]].reset_index(drop=True)
    pooled_candidate = metric(truth, candidate)
    pooled_tabular = metric(truth, tabular)
    pooled_bootstrap = paired_bootstrap(
        truth,
        candidate,
        tabular,
        metadata,
        replicates=int(config["validation"]["bootstrap_replicates"]),
        seed=int(config["seed"]) + 10,
    )
    q34 = reference_oof["fold"].isin(["2025_q3", "2025_q4"]).to_numpy()
    q34_candidate = metric(truth[q34], candidate[q34])
    q34_e150 = metric(truth[q34], e150_prediction[q34])
    q34_bootstrap = paired_bootstrap(
        truth[q34],
        candidate[q34],
        e150_prediction[q34],
        metadata.loc[q34].reset_index(drop=True),
        replicates=int(config["validation"]["bootstrap_replicates"]),
        seed=int(config["seed"]) + 20,
    )
    runtime = float(time.perf_counter() - started)
    gates = {
        "all_q2_q3_q4_delta_f1_vs_tabular_nonnegative": all(
            record["delta_f1_vs_tabular"] >= 0.0 for record in by_fold.values()
        ),
        "pooled_delta_f1_vs_tabular_positive": pooled_candidate["f1"] > pooled_tabular["f1"],
        "pooled_bootstrap_ci90_low_vs_tabular_positive": pooled_bootstrap["difference_ci90"][0] > 0.0,
        "q3_q4_delta_f1_vs_e150_positive": q34_candidate["f1"] > q34_e150["f1"],
        "q3_q4_bootstrap_ci90_low_vs_e150_positive": q34_bootstrap["difference_ci90"][0] > 0.0,
        "runtime_at_most_seconds": runtime <= float(config["maximum_runtime_seconds"]),
        "official_accesses_equal_zero": True,
    }
    delta_e150 = float(q34_candidate["f1"] - q34_e150["f1"])
    score_interval = [
        PUBLIC_BEST_POINTS + PUBLIC_SCORE_SLOPE * q34_bootstrap["difference_ci90"][0],
        PUBLIC_BEST_POINTS + PUBLIC_SCORE_SLOPE * q34_bootstrap["difference_ci90"][1],
    ]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PROMOTE" if all(gates.values()) else "NO_GO_INTERNAL_GATE",
        "independent_candidate": True,
        "candidate_uses_incumbent_as_input": False,
        "candidate_uses_official_values": False,
        "fit_count": len(fit_records),
        "fit_records": fit_records,
        "runtime_seconds": runtime,
        "threshold": threshold,
        "by_fold": by_fold,
        "pooled_vs_tabular": {
            "candidate": pooled_candidate,
            "reference": pooled_tabular,
            "delta_f1": float(pooled_candidate["f1"] - pooled_tabular["f1"]),
            "bootstrap": pooled_bootstrap,
        },
        "q3_q4_vs_e150": {
            "candidate": q34_candidate,
            "reference": q34_e150,
            "delta_f1": delta_e150,
            "bootstrap": q34_bootstrap,
        },
        "public_score_translation": {
            "assumption": "empirical local linear slope only; not a guarantee",
            "best_public_f1": PUBLIC_BEST_F1,
            "best_public_points": PUBLIC_BEST_POINTS,
            "expected_points_center": PUBLIC_BEST_POINTS + PUBLIC_SCORE_SLOPE * delta_e150,
            "expected_points_ci90": score_interval,
        },
        "gates": gates,
        "official_access": {
            "test_rows": 0,
            "sample_rows": 0,
            "submission_rows": 0,
            "hidden_label_rows": 0,
            "uploads": 0,
        },
        "input_hashes": {
            "train_csv": config["inputs"]["train_csv_sha256"],
            "feature_cache": config["inputs"]["feature_cache_sha256"],
            "tabular_reference_oof": config["inputs"]["tabular_reference_oof_sha256"],
            "config": sha256_file(CONFIG_PATH),
        },
        "completed_kst": now_kst().isoformat(),
    }
    reference_oof.to_parquet(OOF_PATH, index=False, compression="zstd")
    result["historical_oof"] = {
        "path": str(OOF_PATH),
        "rows": int(len(reference_oof)),
        "sha256": sha256_file(OOF_PATH),
    }
    write_json_new(RESULT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "sealed": True, "config": config}, indent=2))
        return 0
    try:
        result = execute()
    except Exception as error:
        failure = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(error).__name__,
            "error": str(error),
            "official_access": {
                "test_rows": 0,
                "sample_rows": 0,
                "submission_rows": 0,
                "hidden_label_rows": 0,
                "uploads": 0,
            },
            "completed_kst": now_kst().isoformat(),
        }
        if not RESULT_PATH.exists():
            write_json_new(RESULT_PATH, failure)
        raise
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
