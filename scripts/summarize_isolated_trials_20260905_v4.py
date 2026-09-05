"""Post-hoc descriptive uncertainty; never changes a sealed trial decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/parallel_isolated_and_regeneration_20260905_v4"
P1 = "p1_ts_disagreement_20260905_v4"
P2 = "p2_profile_copula_residual_20260905_v4"
P3 = "p3_wind_only_dropout_20260905_v4"
REPS = 2000
SEED = 20260905


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contributions(y, pred, metric):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError("nonempty aligned vectors required")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("nonfinite evaluation value")
    if metric == "f1":
        if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
            raise ValueError("binary values required")
        denominator = (y == 1).astype(float) + (pred == 1).astype(float)
        return np.c_[2 * ((y == 1) & (pred == 1)), denominator].astype(float)
    if metric != "rmse":
        raise ValueError("unknown metric")
    return np.c_[(y - pred) ** 2, np.ones(len(y))]


def value(sums, metric):
    if np.any(sums[..., 1] <= 0):
        raise ValueError("undefined metric denominator")
    quotient = sums[..., 0] / sums[..., 1]
    return quotient if metric == "f1" else np.sqrt(quotient)


def comparison(frame, metric, dimensions):
    """Stratified paired cluster bootstrap; totals, not mean block scores."""
    if frame[["fold", "cluster"]].isna().any().any():
        raise ValueError("missing bootstrap metadata")
    a = contributions(frame.y, frame.control, metric)
    b = contributions(frame.y, frame.candidate, metric)
    total_a, total_b = float(value(a.sum(0), metric)), float(value(b.sum(0), metric))
    raw_delta = total_b - total_a
    direction = 1 if metric == "f1" else -1
    rng = np.random.default_rng(SEED)
    boot_a = np.zeros((REPS, 2))
    boot_b = np.zeros_like(boot_a)
    block_count = 0
    for _, indexes in frame.groupby("fold", sort=True).indices.items():
        ids, unique = pd.factorize(frame.iloc[indexes].cluster, sort=True)
        sums_a = np.zeros((len(unique), 2))
        sums_b = np.zeros_like(sums_a)
        np.add.at(sums_a, ids, a[indexes])
        np.add.at(sums_b, ids, b[indexes])
        draws = rng.integers(0, len(unique), (REPS, len(unique)))
        boot_a += sums_a[draws].sum(1)
        boot_b += sums_b[draws].sum(1)
        block_count += len(unique)
    defined = (boot_a[:, 1] > 0) & (boot_b[:, 1] > 0)
    deltas = value(boot_b[defined], metric) - value(boot_a[defined], metric)
    slices = []
    for dimension in ["fold", *dimensions]:
        for key, indexes in frame.groupby(dimension, dropna=False).indices.items():
            ca, cb = a[indexes].sum(0), b[indexes].sum(0)
            if min(ca[1], cb[1]) <= 0:
                slices.append(
                    {"dimension": dimension, "key": str(key), "rows": len(indexes), "delta": None}
                )
                continue
            delta = float(value(cb, metric) - value(ca, metric))
            slices.append(
                {
                    "dimension": dimension,
                    "key": str(key),
                    "rows": len(indexes),
                    "delta": delta,
                    "harm": -direction * delta,
                }
            )
    return {
        "rows": len(frame),
        "metric": metric,
        "control": total_a,
        "candidate": total_b,
        "candidate_minus_control": raw_delta,
        "ci90_descriptive": np.quantile(deltas, [0.05, 0.95]).tolist(),
        "bootstrap_improvement_fraction_not_posterior": float(np.mean(direction * deltas > 0)),
        "cluster_count": block_count,
        "valid_bootstrap_replicates": len(deltas),
        "worst_fold_harm": max(
            x["harm"] for x in slices if x["dimension"] == "fold" and x["delta"] is not None
        ),
        "maximum_slice_harm": max(
            x["harm"] for x in slices if x["dimension"] != "fold" and x["delta"] is not None
        ),
        "slices": slices,
    }


def run():
    hashes = {}

    def checked(path, expected=None):
        path = ROOT / path
        actual = sha(path)
        if expected and actual != expected:
            raise ValueError(f"source hash mismatch: {path.name}")
        hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = actual
        return path

    p1_result = json.loads(checked(f"reports/{P1}/result.json").read_text())
    parts = []
    for name, digest in p1_result["artifact_hashes"].items():
        if name.endswith("_intact.parquet"):
            parts.append(pd.read_parquet(checked(f"artifacts/{P1}/{name}", digest)))
    p1 = pd.concat(parts, ignore_index=True).rename(columns={"label": "y"})
    if p1.row_id.duplicated().any():
        raise ValueError("P1 duplicate key")
    p1["cluster"] = (
        pd.to_datetime(p1.time, utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    )

    data = np.load(
        checked(
            f"artifacts/{P2}/historical_eval.npz",
            "17a3b87e7ae106d619435627cafe57c66f73464a31de9c222d0c3fa0aa3cb8b8",
        ),
        allow_pickle=False,
    )
    p2 = pd.DataFrame(
        {
            "key": data["key"],
            "y": data["truth"],
            "control": data["intact_C"],
            "candidate": data["intact_full"],
            "fold": data["fold"],
        }
    )
    if p2.key.duplicated().any():
        raise ValueError("P2 duplicate key")
    pieces = p2.key.str.rsplit("|", n=1, expand=True)
    p2["layer"] = pieces[1]
    days = pd.to_datetime(pieces[0], utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    p2["cluster"] = ((days - pd.Timestamp("1970-01-01")).dt.days // 7).astype(str)

    keys = ["anchor_id", "station", "lead_h", "fold"]
    oof = pd.read_parquet(
        checked(
            f"artifacts/{P3}/oof.parquet",
            "0f1d6478f2e4f1cfb496267fcd150884661ac5d1e1711250d08cd695bc19b25f",
        )
    )
    reference = pd.read_parquet(
        checked(
            f"artifacts/{P3}/reference.parquet",
            "36e7bb31526d7acb63a11acfe436a65cf679ab52792a737c81d7e5ed227cbdd4",
        )
    )
    p3 = oof[oof.arm == "control"][keys + ["target_hs", "prediction"]].rename(
        columns={"target_hs": "y", "prediction": "control"}
    )
    wind = oof[oof.arm == "wind_only"][keys + ["prediction", "target_hs"]].rename(
        columns={"prediction": "candidate"}
    )
    p3 = p3.merge(wind, on=keys, validate="one_to_one", how="outer", indicator=True)
    if not p3._merge.eq("both").all() or not np.array_equal(p3.y, p3.target_hs):
        raise ValueError("P3 paired alignment mismatch")
    p3 = p3.drop(columns=["_merge", "target_hs"]).merge(
        reference[keys + ["episode_id", "target_hs", "final_prediction"]],
        on=keys,
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    if not p3._merge.eq("both").all() or not np.array_equal(p3.y, p3.target_hs):
        raise ValueError("P3 reference alignment mismatch")
    p3["cluster"] = p3.station.astype(str) + ":" + p3.episode_id.astype(str)

    results = {
        "P1_TS_vs_control": comparison(p1, "f1", ["station", "layer"]),
        "P2_profile_primary_autumn": comparison(
            p2[p2.fold == "2024_sep_oct"].reset_index(drop=True), "rmse", ["layer"]
        ),
        "P2_profile_pooled": comparison(p2, "rmse", ["layer"]),
        "P3_wind_vs_matched_control": comparison(p3, "rmse", ["station", "lead_h"]),
        "P3_wind_vs_clean": comparison(
            p3.assign(control=p3.final_prediction), "rmse", ["station", "lead_h"]
        ),
    }
    if not np.isclose(
        results["P1_TS_vs_control"]["candidate_minus_control"],
        p1_result["delta_f1"],
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("P1 canonical arithmetic mismatch")
    output = {
        "status": "DESCRIPTIVE_RECOMPUTATION_COMPLETE",
        "source_hashes": hashes,
        "results": results,
        "bootstrap": {
            "replicates": REPS,
            "seed": SEED,
            "strata": "existing fold",
            "clusters": {
                "P1": "KST calendar day, all stations together",
                "P2": "KST nonoverlapping 7-day calendar block",
                "P3": "station x saved episode",
            },
        },
        "limitations": [
            "post-hoc descriptive protocol, not preregistered confirmation",
            "historical periods previously exposed; CI ignores adaptive selection and future distribution shift",
            "bootstrap fraction is not probability of official score gain",
            "no decision threshold applied; original trial decisions unchanged",
            "baseline regeneration and clean-machine packaging are separate checks",
        ],
        "fits": 0,
        "official_rows": 0,
        "csv_written": 0,
        "uploads": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: {k: v for k, v in value.items() if k != "slices"}
                for key, value in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
