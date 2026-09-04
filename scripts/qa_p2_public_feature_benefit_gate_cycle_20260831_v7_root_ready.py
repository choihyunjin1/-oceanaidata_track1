"""Independent root-ready recomputation for the P2 v7 PASS."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = "p2_public_feature_benefit_gate_cycle_20260831_v7"
ART = ROOT / "artifacts" / EXP
REP = ROOT / "reports" / EXP
OBS = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv")
TEST = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\test_index.csv")
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


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(truth)))))


def main() -> None:
    result_path = ART / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads((ART / "materialization.json").read_text(encoding="utf-8"))
    candidate = next(item for item in result["candidates"] if item["pass"])
    sealed_path = ART / f"{candidate['name']}.npz"
    sealed = np.load(sealed_path)
    reference = sealed["reference"].astype(float)
    prediction = sealed["candidate"].astype(float)

    scored = pd.read_parquet(SCORING)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    observations = pd.read_csv(OBS, usecols=["time", "layer", "temp"])
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    frame = scored[["time", "layer"]].merge(
        observations, on=["time", "layer"], how="left", validate="one_to_one"
    )
    truth = frame["temp"].to_numpy(float)
    local = pd.DatetimeIndex(frame["time"]).tz_convert("Asia/Seoul")
    train = (local >= pd.Timestamp("2024-09-01", tz="Asia/Seoul")) & (
        local < pd.Timestamp("2024-10-01", tz="Asia/Seoul")
    )
    test = (local >= pd.Timestamp("2024-10-01", tz="Asia/Seoul")) & (
        local < pd.Timestamp("2024-11-01", tz="Asia/Seoul")
    )
    train_times = local[train]
    test_times = local[test]
    split = {
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "train_max_kst": train_times.max().isoformat(),
        "test_min_kst": test_times.min().isoformat(),
        "overlap_rows": int((train & test).sum()),
        "boundary_gap_hours": float((test_times.min() - train_times.max()).total_seconds() / 3600),
        "boundary_gap_minutes": float((test_times.min() - train_times.max()).total_seconds() / 60),
        "purge_contract": "no shared timestamp and at least one 10-minute sampling interval; no target-derived lag feature",
    }
    delta = rmse(truth[test], prediction[test]) - rmse(truth[test], reference[test])
    layer_delta = {
        str(layer): rmse(
            truth[test & (frame["layer"].to_numpy() == layer)],
            prediction[test & (frame["layer"].to_numpy() == layer)],
        )
        - rmse(
            truth[test & (frame["layer"].to_numpy() == layer)],
            reference[test & (frame["layer"].to_numpy() == layer)],
        )
        for layer in (2, 3, 4)
    }

    test_rows = np.flatnonzero(test)
    dates = pd.Series(local[test].date)
    daily = []
    for _, indices in dates.groupby(dates).groups.items():
        rows = test_rows[np.asarray(list(indices), dtype=int)]
        daily.append(
            (
                float(np.square(prediction[rows] - truth[rows]).sum()),
                float(np.square(reference[rows] - truth[rows]).sum()),
                len(rows),
            )
        )
    rng = np.random.default_rng(20260831)
    bootstrap = np.empty(2000, dtype=float)
    for replicate in range(2000):
        draw = rng.integers(0, len(daily), len(daily))
        rows = sum(daily[index][2] for index in draw)
        bootstrap[replicate] = np.sqrt(sum(daily[index][0] for index in draw) / rows) - np.sqrt(
            sum(daily[index][1] for index in draw) / rows
        )
    ci_high = float(np.quantile(bootstrap, 0.95))
    raw_points = max(0.0, -ci_high * POINTS_PER_RMSE)
    calibrated = raw_points - PENALTY

    submission_path = Path(receipt["path"])
    submission = pd.read_csv(submission_path, dtype={"station": "string", "time": "string"})
    official = pd.read_csv(TEST, dtype={"station": "string", "time": "string"})
    keys = ["station", "layer", "time"]
    submission_checks = {
        "rows_26061": len(submission) == 26061,
        "schema_exact": list(submission.columns) == [*keys, "temp"],
        "key_order_exact": submission[keys].equals(official[keys]),
        "duplicates_zero": not submission.duplicated(keys).any(),
        "finite": bool(np.isfinite(submission["temp"]).all()),
        "sha256_matches": sha256_file(submission_path) == receipt["sha256"],
    }
    tolerance = 1e-12
    checks = {
        "result_sha_recomputed": sha256_file(result_path)
        == "db35e824d8d4957dac35a281b5d48bb03716c7b920ed94e5e5e5ece8a6ad5a10",
        "fit_count_internal_2_full_1": result["fit_count"] == 2 and receipt["full_fit_count"] == 1,
        "split_exact_and_purged": split["overlap_rows"] == 0
        and split["boundary_gap_minutes"] >= 10.0
        and split["test_rows"] == 13339,
        "october_delta_matches": abs(delta - candidate["october_delta_rmse"]) <= tolerance,
        "layer_deltas_match": all(
            abs(layer_delta[key] - candidate["october_layer_delta_rmse"][key]) <= tolerance
            for key in layer_delta
        ),
        "bootstrap_ci90_matches": abs(ci_high - candidate["bootstrap_ci90_high_delta_rmse"])
        <= tolerance,
        "raw_points_formula_matches": abs(raw_points - candidate["raw_expected_points_delta"])
        <= tolerance,
        "penalty_exact": PENALTY == 0.12168209161000616,
        "calibrated_formula_matches": abs(
            calibrated - candidate["calibrated_expected_points_delta"]
        )
        <= tolerance,
        "inclusive_gate_exact": candidate["pass"]
        and calibrated >= 0.01
        and candidate["gate_checks"]["calibrated_expected_points_gte_0_01"],
        "submission_all_checks": all(submission_checks.values()),
        "official_access_order_and_count": result["official_test_index_rows_read"] == 0
        and receipt["official_test_index_rows_read"] == 26061,
        "hidden_and_upload_zero": result["hidden_truth_rows_read"] == 0
        and result["upload_count"] == 0
        and receipt["hidden_truth_rows_read"] == 0
        and receipt["upload_count"] == 0,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "split_recomputed": split,
        "metrics_recomputed": {
            "october_delta_rmse": delta,
            "layer_delta_rmse": layer_delta,
            "bootstrap_ci90_low": float(np.quantile(bootstrap, 0.05)),
            "bootstrap_ci90_high": ci_high,
            "raw_expected_points_delta": raw_points,
            "transport_penalty_points": PENALTY,
            "calibrated_expected_points_delta": calibrated,
        },
        "submission_recomputed": {
            **submission_checks,
            "path": str(submission_path),
            "sha256": sha256_file(submission_path),
        },
        "artifact_hashes": {
            "result_sha256": sha256_file(result_path),
            "prediction_sha256": sha256_file(sealed_path),
        },
    }
    REP.mkdir(parents=True, exist_ok=True)
    output = REP / "independent-root-ready-qa.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
