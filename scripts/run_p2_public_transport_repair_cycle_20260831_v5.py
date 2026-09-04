"""Frozen P2 Public-transport repair: robust correction routers on sealed OOF predictions."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EXP = "p2_public_transport_repair_cycle_20260831_v5"
ART = ROOT / "artifacts" / EXP
REP = ROOT / "reports" / EXP
NPZ = ROOT / "artifacts/p2_parallel_candidate_cycle_20260831_v4/P2_2_HGB_ABSOLUTE_PROFILE.npz"
OBS = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv")
SCORING = (
    ROOT
    / "artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2/scored_predictions_no_truth.parquet"
)
PENALTY = 0.12168209161000616
SCALE = 12.5475311


def sha(p):
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def rmse(y, p, w=None):
    e = (np.asarray(p) - np.asarray(y)) ** 2
    return float(np.sqrt(np.average(e, weights=w)))


def feats(frame, raw):
    t = pd.DatetimeIndex(frame.time).tz_convert("Asia/Seoul")
    layer = frame.layer.to_numpy(int)
    return np.column_stack(
        [
            raw,
            np.abs(raw),
            np.sin(2 * np.pi * t.hour / 24),
            np.cos(2 * np.pi * t.hour / 24),
            np.sin(2 * np.pi * t.dayofyear / 365.25),
            np.cos(2 * np.pi * t.dayofyear / 365.25),
            *(layer == k for k in (2, 3, 4)),
        ]
    )


def model():
    return make_pipeline(
        StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=500, tol=1e-6)
    )


def main():
    started = time.perf_counter()
    REP.mkdir(parents=True, exist_ok=True)
    if ART.exists():
        existing = {p.name for p in ART.iterdir()}
        if existing == {"attempt_lock.json"}:
            dump(
                ART / "technical_recovery.json",
                {
                    "cause": "initial interpreter lacked scikit-learn",
                    "performance_rows_scored": 0,
                    "prediction_rows_written": 0,
                    "official_rows_read": 0,
                    "recovery": "same frozen runner under repository .venv-p1",
                },
            )
        elif existing == {"attempt_lock.json", "technical_recovery.json"}:
            dump(
                ART / "technical_recovery_2.json",
                {
                    "cause": "boolean mask was already ndarray",
                    "performance_rows_scored": 0,
                    "prediction_rows_written": 0,
                    "official_rows_read": 0,
                    "repair": "remove invalid to_numpy call only",
                },
            )
        elif existing == {
            "attempt_lock.json",
            "technical_recovery.json",
            "technical_recovery_2.json",
        }:
            dump(
                ART / "technical_recovery_3.json",
                {
                    "cause": "pandas-derived boolean view was read-only",
                    "performance_rows_scored": 0,
                    "prediction_rows_written": 0,
                    "official_rows_read": 0,
                    "repair": "copy train mask before in-place conjunction",
                },
            )
        else:
            raise FileExistsError(ART)
    else:
        ART.mkdir(parents=True)
        dump(
            ART / "attempt_lock.json",
            {"experiment_id": EXP, "runner_sha256": sha(Path(__file__)), "adaptive_retry": False},
        )
    z = np.load(NPZ)
    ref = z["reference"].astype(float)
    base = z["candidate"].astype(float)
    raw = base - ref
    scored = pd.read_parquet(SCORING)
    scored["time"] = pd.to_datetime(scored.time, utc=True)
    obs = pd.read_csv(OBS, usecols=["time", "layer", "temp"])
    obs["time"] = pd.to_datetime(obs.time, utc=True)
    frame = scored[["time", "layer", "fold"]].merge(
        obs, on=["time", "layer"], how="left", validate="one_to_one"
    )
    if len(frame) != len(ref) or frame.temp.isna().any():
        raise RuntimeError("sealed alignment failed")
    X = feats(frame, raw)
    y = frame.temp.to_numpy(float)
    target = y - ref
    local = pd.DatetimeIndex(frame.time).tz_convert("Asia/Seoul")
    tests = [
        (
            "2024_oct_nested",
            (local >= pd.Timestamp("2024-10-01", tz="Asia/Seoul"))
            & (local < pd.Timestamp("2024-11-01", tz="Asia/Seoul")),
        ),
        ("2025_jul_aug", frame.fold.eq("2025_jul_aug").to_numpy()),
        ("2025_nov_dec", frame.fold.eq("2025_nov_dec").to_numpy()),
    ]
    candidates = []
    total_fits = 0
    for name, scope in [
        ("P2_V5_GLOBAL_HUBER_ROUTER", "global"),
        ("P2_V5_LAYER_HUBER_ROUTER", "layer"),
    ]:
        pred = ref.copy()
        folds = {}
        for fold, te in tests:
            cutoff = frame.loc[te, "time"].min()
            tr = (frame.time < cutoff).to_numpy().copy()
            if fold == "2024_oct_nested":
                tr &= local >= pd.Timestamp("2024-09-01", tz="Asia/Seoul")
            if scope == "global":
                m = model()
                m.fit(X[tr], target[tr])
                corr = m.predict(X[te])
                total_fits += 1
            else:
                corr = np.zeros(te.sum())
                tl = frame.loc[te, "layer"].to_numpy()
                for layer in (2, 3, 4):
                    m = model()
                    lt = tr & (frame.layer.to_numpy() == layer)
                    m.fit(X[lt], target[lt])
                    corr[tl == layer] = m.predict(X[te][tl == layer])
                    total_fits += 1
            corr = np.clip(corr, -0.12, 0.12)
            pred[te] = ref[te] + corr
            folds[fold] = {
                "rows": int(te.sum()),
                "reference_rmse": rmse(y[te], ref[te]),
                "candidate_rmse": rmse(y[te], pred[te]),
            }
            folds[fold]["delta_rmse"] = (
                folds[fold]["candidate_rmse"] - folds[fold]["reference_rmse"]
            )
        evalmask = np.logical_or.reduce([x[1] for x in tests])
        pooled = rmse(y[evalmask], pred[evalmask]) - rmse(y[evalmask], ref[evalmask])
        off = tests[0][1]
        layers = {
            str(k): rmse(
                y[off & (frame.layer.to_numpy() == k)], pred[off & (frame.layer.to_numpy() == k)]
            )
            - rmse(y[off & (frame.layer.to_numpy() == k)], ref[off & (frame.layer.to_numpy() == k)])
            for k in (2, 3, 4)
        }
        dates = pd.Series(local[off].date)
        dayparts = []
        for _, idx in dates.groupby(dates).groups.items():
            ii = np.flatnonzero(off)[np.asarray(list(idx), int)]
            dayparts.append(
                (np.square(pred[ii] - y[ii]).sum(), np.square(ref[ii] - y[ii]).sum(), len(ii))
            )
        rng = np.random.default_rng(20260831)
        boot = []
        for _ in range(2000):
            q = rng.integers(0, len(dayparts), len(dayparts))
            c = sum(dayparts[i][0] for i in q)
            r = sum(dayparts[i][1] for i in q)
            n = sum(dayparts[i][2] for i in q)
            boot.append(np.sqrt(c / n) - np.sqrt(r / n))
        ci_hi = float(np.quantile(boot, 0.95))
        raw_points = max(0.0, -ci_hi * SCALE)
        calibrated = raw_points - PENALTY
        checks = {
            "calibrated_expected_points_gte_0_01": calibrated >= 0.01,
            "minimum_two_outer_tests_improve": sum(v["delta_rmse"] < 0 for v in folds.values())
            >= 2,
            "official_like_nested_improves": folds["2024_oct_nested"]["delta_rmse"] < 0,
            "worst_outer_delta_lte_0_002": max(v["delta_rmse"] for v in folds.values()) <= 0.002,
            "all_official_like_layers_improve": max(layers.values()) < 0,
        }
        out = ART / (name + ".npz")
        np.savez_compressed(
            out,
            reference=ref,
            candidate=pred,
            truth=y,
            time_ns=pd.DatetimeIndex(frame.time).asi8,
            layer=frame.layer.to_numpy(),
        )
        candidates.append(
            {
                "name": name,
                "scope": scope,
                "pooled_delta_rmse": pooled,
                "outer_tests": folds,
                "official_like_layer_delta_rmse": layers,
                "official_like_bootstrap_ci90_high_delta_rmse": ci_hi,
                "raw_expected_points_delta": raw_points,
                "transport_penalty_points": PENALTY,
                "calibrated_expected_points_delta": calibrated,
                "gate_checks": checks,
                "pass": all(checks.values()),
                "prediction_sha256": sha(out),
            }
        )
    result = {
        "experiment_id": EXP,
        "status": "COMPLETE_WITH_PASS"
        if any(c["pass"] for c in candidates)
        else "COMPLETE_NO_PASS",
        "candidate_count": 2,
        "fit_count": total_fits,
        "runtime_seconds": time.perf_counter() - started,
        "official_test_index_rows_read": 0,
        "hidden_truth_rows_read": 0,
        "upload_count": 0,
        "submission_count": 0,
        "candidates": candidates,
        "runner_sha256": sha(Path(__file__)),
    }
    dump(ART / "result.json", result)
    dump(REP / "result.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
