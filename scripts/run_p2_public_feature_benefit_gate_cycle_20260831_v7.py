"""Frozen public-feature benefit gates for P2."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_p2_parallel_candidate_cycle_20260831_v4 import p2_public_features  # noqa: E402

EXP = "p2_public_feature_benefit_gate_cycle_20260831_v7"
ART = ROOT / "artifacts" / EXP
REP = ROOT / "reports" / EXP
BASE_ART = ROOT / "artifacts/p2_parallel_candidate_cycle_20260831_v4/P2_2_HGB_ABSOLUTE_PROFILE.npz"
OBS = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv")
SCORING = (
    ROOT
    / "artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2/scored_predictions_no_truth.parquet"
)
PENALTY = 0.12168209161000616
POINTS_PER_RMSE = 12.5475311


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def main() -> None:
    started = time.perf_counter()
    ART.mkdir(parents=True, exist_ok=False)
    REP.mkdir(parents=True, exist_ok=True)
    dump(ART / "attempt_lock.json", {"experiment_id": EXP, "adaptive_retry": False})
    sealed = np.load(BASE_ART)
    reference = sealed["reference"].astype(float)
    base = sealed["candidate"].astype(float)
    raw = base - reference
    observations = pd.read_csv(OBS)
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    scored = pd.read_parquet(SCORING)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    frame = scored[["time", "layer", "fold"]].merge(
        observations[["time", "layer", "temp"]], on=["time", "layer"], validate="one_to_one"
    )
    public = p2_public_features(observations).reindex(pd.DatetimeIndex(frame["time"]))
    local = pd.DatetimeIndex(frame["time"]).tz_convert("Asia/Seoul")
    layer = frame["layer"].to_numpy(int)
    extra = np.column_stack(
        [
            raw,
            np.abs(raw),
            np.sin(2 * np.pi * local.hour / 24),
            np.cos(2 * np.pi * local.hour / 24),
            *(layer == value for value in (2, 3, 4)),
        ]
    )
    matrix = np.column_stack([public.to_numpy(float), extra])
    truth = frame["temp"].to_numpy(float)
    beneficial = np.square(reference - truth) > np.square(base - truth)
    train = (local >= pd.Timestamp("2024-09-01", tz="Asia/Seoul")) & (
        local < pd.Timestamp("2024-10-01", tz="Asia/Seoul")
    )
    october = (local >= pd.Timestamp("2024-10-01", tz="Asia/Seoul")) & (
        local < pd.Timestamp("2024-11-01", tz="Asia/Seoul")
    )
    specs = [
        (
            "P2_V7_EXTRATREES_PUBLIC_BENEFIT_GATE",
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=32,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=4,
                random_state=20260831,
            ),
        ),
        (
            "P2_V7_RANDOMFOREST_PUBLIC_BENEFIT_GATE",
            RandomForestClassifier(
                n_estimators=240,
                min_samples_leaf=64,
                max_features=0.5,
                class_weight="balanced_subsample",
                n_jobs=4,
                random_state=20260831,
            ),
        ),
    ]
    records = []
    for name, classifier in specs:
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), classifier)
        model.fit(matrix[train], beneficial[train])
        probability = model.predict_proba(matrix[october])[:, 1]
        active = np.zeros(len(frame), dtype=bool)
        active[np.flatnonzero(october)[probability >= 0.5]] = True
        prediction = reference.copy()
        prediction[active] = base[active]
        delta = rmse(truth[october], prediction[october]) - rmse(truth[october], reference[october])
        layer_delta = {
            str(value): rmse(
                truth[october & (layer == value)], prediction[october & (layer == value)]
            )
            - rmse(truth[october & (layer == value)], reference[october & (layer == value)])
            for value in (2, 3, 4)
        }
        dates = pd.Series(local[october].date)
        oct_rows = np.flatnonzero(october)
        groups = []
        for _, indices in dates.groupby(dates).groups.items():
            rows = oct_rows[np.asarray(list(indices), int)]
            groups.append(
                (
                    np.square(prediction[rows] - truth[rows]).sum(),
                    np.square(reference[rows] - truth[rows]).sum(),
                    len(rows),
                )
            )
        rng = np.random.default_rng(20260831)
        boot = []
        for _ in range(2000):
            draw = rng.integers(0, len(groups), len(groups))
            count = sum(groups[index][2] for index in draw)
            boot.append(
                np.sqrt(sum(groups[index][0] for index in draw) / count)
                - np.sqrt(sum(groups[index][1] for index in draw) / count)
            )
        ci_high = float(np.quantile(boot, 0.95))
        raw_points = max(0.0, -ci_high * POINTS_PER_RMSE)
        calibrated = raw_points - PENALTY
        checks = {
            "calibrated_expected_points_gte_0_01": calibrated >= 0.01,
            "october_improves": delta < 0,
            "all_layers_improve": max(layer_delta.values()) < 0,
            "transfer_exact_abstention": bool(
                np.allclose(prediction[~october], reference[~october])
            ),
            "active_share_lte_0_8": float(active[october].mean()) <= 0.8,
        }
        output = ART / f"{name}.npz"
        np.savez_compressed(output, reference=reference, candidate=prediction, active=active)
        records.append(
            {
                "name": name,
                "october_delta_rmse": delta,
                "october_layer_delta_rmse": layer_delta,
                "active_rows": int(active.sum()),
                "active_share": float(active[october].mean()),
                "bootstrap_ci90_high_delta_rmse": ci_high,
                "raw_expected_points_delta": raw_points,
                "calibrated_expected_points_delta": calibrated,
                "gate_checks": checks,
                "pass": all(checks.values()),
                "prediction_sha256": sha256_file(output),
            }
        )
    result = {
        "experiment_id": EXP,
        "status": "COMPLETE_WITH_PASS"
        if any(item["pass"] for item in records)
        else "COMPLETE_NO_PASS",
        "fit_count": 2,
        "runtime_seconds": time.perf_counter() - started,
        "official_test_index_rows_read": 0,
        "hidden_truth_rows_read": 0,
        "submission_count": 0,
        "upload_count": 0,
        "candidates": records,
    }
    dump(ART / "result.json", result)
    dump(REP / "result.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
