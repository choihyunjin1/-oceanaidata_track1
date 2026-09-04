"""Frozen support-gated abstention repair for P2 Public transport."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EXP = "p2_public_support_abstention_cycle_20260831_v6"
ART = ROOT / "artifacts" / EXP
REP = ROOT / "reports" / EXP
NPZ = ROOT / "artifacts/p2_parallel_candidate_cycle_20260831_v4/P2_2_HGB_ABSOLUTE_PROFILE.npz"
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


def features(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    local = pd.DatetimeIndex(frame["time"]).tz_convert("Asia/Seoul")
    layer = frame["layer"].to_numpy(int)
    return np.column_stack(
        [
            raw,
            np.abs(raw),
            np.sin(2 * np.pi * local.hour / 24),
            np.cos(2 * np.pi * local.hour / 24),
            *(layer == value for value in (2, 3, 4)),
        ]
    )


def main() -> None:
    started = time.perf_counter()
    ART.mkdir(parents=True, exist_ok=False)
    REP.mkdir(parents=True, exist_ok=True)
    dump(ART / "attempt_lock.json", {"experiment_id": EXP, "adaptive_retry": False})
    sealed = np.load(NPZ)
    reference = sealed["reference"].astype(float)
    base = sealed["candidate"].astype(float)
    raw = base - reference
    scored = pd.read_parquet(SCORING)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    observations = pd.read_csv(OBS, usecols=["time", "layer", "temp"])
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    frame = scored[["time", "layer", "fold"]].merge(
        observations, on=["time", "layer"], how="left", validate="one_to_one"
    )
    truth = frame["temp"].to_numpy(float)
    matrix = features(frame, raw)
    local = pd.DatetimeIndex(frame["time"]).tz_convert("Asia/Seoul")
    train = (local >= pd.Timestamp("2024-09-01", tz="Asia/Seoul")) & (
        local < pd.Timestamp("2024-10-01", tz="Asia/Seoul")
    )
    october = (local >= pd.Timestamp("2024-10-01", tz="Asia/Seoul")) & (
        local < pd.Timestamp("2024-11-01", tz="Asia/Seoul")
    )
    beneficial = np.square(reference - truth) > np.square(base - truth)
    median = np.median(matrix[train], axis=0)
    mad = np.median(np.abs(matrix[train] - median), axis=0)
    scale = np.where(mad > 1e-12, mad, np.inf)
    in_support = np.all(np.abs(matrix - median) <= 6.0 * scale, axis=1)
    records = []
    total_fits = 0
    for name, scope in (
        ("P2_V6_HGB_BENEFIT_SUPPORT_GATE", "global"),
        ("P2_V6_LAYER_LOGIT_SUPPORT_GATE", "layer"),
    ):
        probability = np.zeros(len(frame))
        if scope == "global":
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=120,
                max_leaf_nodes=15,
                min_samples_leaf=128,
                l2_regularization=5.0,
                random_state=20260831,
            )
            model.fit(matrix[train], beneficial[train])
            probability[october] = model.predict_proba(matrix[october])[:, 1]
            total_fits += 1
        else:
            for layer in (2, 3, 4):
                layer_mask = frame["layer"].to_numpy() == layer
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.25, class_weight="balanced", max_iter=500),
                )
                model.fit(matrix[train & layer_mask], beneficial[train & layer_mask])
                query = october & layer_mask
                probability[query] = model.predict_proba(matrix[query])[:, 1]
                total_fits += 1
        active = october & in_support & (probability >= 0.5)
        prediction = reference.copy()
        prediction[active] = base[active]
        oct_delta = rmse(truth[october], prediction[october]) - rmse(
            truth[october], reference[october]
        )
        layer_delta = {
            str(layer): rmse(
                truth[october & (frame.layer == layer)],
                prediction[october & (frame.layer == layer)],
            )
            - rmse(
                truth[october & (frame.layer == layer)], reference[october & (frame.layer == layer)]
            )
            for layer in (2, 3, 4)
        }
        dates = pd.Series(local[october].date)
        groups = []
        oct_rows = np.flatnonzero(october)
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
            n = sum(groups[index][2] for index in draw)
            boot.append(
                np.sqrt(sum(groups[index][0] for index in draw) / n)
                - np.sqrt(sum(groups[index][1] for index in draw) / n)
            )
        ci_high = float(np.quantile(boot, 0.95))
        raw_points = max(0.0, -ci_high * POINTS_PER_RMSE)
        calibrated = raw_points - PENALTY
        checks = {
            "calibrated_expected_points_gte_0_01": calibrated >= 0.01,
            "october_improves": oct_delta < 0,
            "all_october_layers_improve": max(layer_delta.values()) < 0,
            "season_transfer_exact_abstention": bool(
                np.allclose(prediction[~october], reference[~october])
            ),
            "active_share_lte_0_8": float(active[october].mean()) <= 0.8,
        }
        output = ART / f"{name}.npz"
        np.savez_compressed(output, reference=reference, candidate=prediction, active=active)
        records.append(
            {
                "name": name,
                "fit_count": 1 if scope == "global" else 3,
                "october_delta_rmse": oct_delta,
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
        "fit_count": total_fits,
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
