from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _q(values: pd.Series) -> dict[str, float | int | None]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {"n": 0, "q10": None, "median": None, "q90": None}
    return {
        "n": int(len(x)),
        "q10": float(x.quantile(0.1)),
        "median": float(x.median()),
        "q90": float(x.quantile(0.9)),
    }


def audit_p1(data_dir: Path) -> dict:
    df = pd.read_csv(data_dir / "train.csv")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values(["station", "layer", "time"]).reset_index(drop=True)
    positive = df["label"].eq(1)
    type_value = df["anomaly_type"].fillna("").astype(str)
    dt_min = df.groupby(["station", "layer"])["time"].diff().dt.total_seconds().div(60)
    event_break = (
        ~positive
        | ~positive.shift(fill_value=False)
        | df["station"].ne(df["station"].shift())
        | df["layer"].ne(df["layer"].shift())
        | type_value.ne(type_value.shift())
        | dt_min.gt(20)
    )
    df["event_id"] = event_break.cumsum().where(positive)

    events: list[dict] = []
    for event_id, part in df.loc[positive].groupby("event_id", sort=False):
        start_i = int(part.index.min())
        end_i = int(part.index.max())
        station = str(part["station"].iloc[0])
        layer = int(part["layer"].iloc[0])
        typ = str(part["anomaly_type"].iloc[0])
        t0 = part["time"].iloc[0]
        t1 = part["time"].iloc[-1]
        same = df["station"].eq(station) & df["layer"].eq(layer)
        before = df.loc[same & df["time"].lt(t0) & df["time"].ge(t0 - pd.Timedelta(hours=6))]
        after = df.loc[same & df["time"].gt(t1) & df["time"].le(t1 + pd.Timedelta(hours=6))]
        normal_context = pd.concat([before.loc[before["label"].eq(0)], after.loc[after["label"].eq(0)]])
        context_median = pd.to_numeric(normal_context["temp"], errors="coerce").median()
        temp = pd.to_numeric(part["temp"], errors="coerce")
        duration_h = float((t1 - t0).total_seconds() / 3600)
        slope_c_per_h = None
        if temp.notna().sum() >= 2 and duration_h > 0:
            slope_c_per_h = float((temp.iloc[-1] - temp.iloc[0]) / duration_h)
        events.append(
            {
                "event_id": int(event_id),
                "type": typ,
                "station": station,
                "layer": layer,
                "month": int(t0.month),
                "rows": int(len(part)),
                "duration_h": duration_h,
                "temp_unique": int(temp.nunique(dropna=True)),
                "temp_std": float(temp.std(ddof=0)) if temp.notna().any() else None,
                "temp_range": float(temp.max() - temp.min()) if temp.notna().any() else None,
                "slope_c_per_h": slope_c_per_h,
                "median_abs_deviation_from_context": (
                    float((temp - context_median).abs().median())
                    if temp.notna().any() and pd.notna(context_median)
                    else None
                ),
                "start_jump_from_prior": (
                    float(temp.iloc[0] - pd.to_numeric(before.loc[before["label"].eq(0), "temp"], errors="coerce").iloc[-1])
                    if temp.notna().any() and not before.loc[before["label"].eq(0), "temp"].dropna().empty
                    else None
                ),
                "end_jump_to_next": (
                    float(pd.to_numeric(after.loc[after["label"].eq(0), "temp"], errors="coerce").iloc[0] - temp.iloc[-1])
                    if temp.notna().any() and not after.loc[after["label"].eq(0), "temp"].dropna().empty
                    else None
                ),
            }
        )
    ev = pd.DataFrame(events)
    by_type = {}
    for typ, part in ev.groupby("type"):
        by_type[str(typ)] = {
            "events": int(len(part)),
            "stations": part["station"].value_counts().sort_index().astype(int).to_dict(),
            "layers": {str(int(k)): int(v) for k, v in part["layer"].value_counts().sort_index().items()},
            "months_present": sorted(int(x) for x in part["month"].unique()),
            "duration_h": _q(part["duration_h"]),
            "rows": _q(part["rows"]),
            "temp_unique": _q(part["temp_unique"]),
            "temp_std": _q(part["temp_std"]),
            "temp_range": _q(part["temp_range"]),
            "abs_slope_c_per_h": _q(part["slope_c_per_h"].abs()),
            "median_abs_deviation_from_context": _q(part["median_abs_deviation_from_context"]),
            "abs_start_jump": _q(part["start_jump_from_prior"].abs()),
            "abs_end_jump": _q(part["end_jump_to_next"].abs()),
        }
    return {
        "event_count": int(len(ev)),
        "event_count_by_station_layer": {
            f"{s}|{int(l)}": int(n)
            for (s, l), n in ev.groupby(["station", "layer"]).size().items()
        },
        "by_type": by_type,
    }


def audit_p2(data_dir: Path) -> dict:
    obs = pd.read_csv(data_dir / "observations.csv")
    idx = pd.read_csv(data_dir / "test_index.csv")
    obs["time"] = pd.to_datetime(obs["time"], utc=True)
    idx["time"] = pd.to_datetime(idx["time"], utc=True)
    t0, t1 = idx["time"].min(), idx["time"].max()
    target_layers = sorted(int(x) for x in idx["layer"].unique())
    boundary = {}
    for layer in target_layers:
        part = obs.loc[obs["layer"].eq(layer)].sort_values("time")
        item = {}
        for col in ["temp", "psal", "depth"]:
            valid = part.loc[part[col].notna()]
            before = valid.loc[valid["time"].lt(t0)]
            after = valid.loc[valid["time"].gt(t1)]
            item[col] = {
                "last_before": before["time"].max().isoformat() if not before.empty else None,
                "hours_from_last_before_to_gap": float((t0 - before["time"].max()).total_seconds() / 3600) if not before.empty else None,
                "first_after": after["time"].min().isoformat() if not after.empty else None,
                "hours_from_gap_to_first_after": float((after["time"].min() - t1).total_seconds() / 3600) if not after.empty else None,
            }
        boundary[str(layer)] = item

    local = obs["time"].dt.tz_convert("Asia/Seoul")
    obs["local_year"] = local.dt.year
    obs["local_month"] = local.dt.month
    obs["local_day"] = local.dt.day
    seasonal = obs.loc[obs["local_month"].isin([9, 10])].copy()
    seasonal_summary = {}
    for year, part in seasonal.groupby("local_year"):
        seasonal_summary[str(int(year))] = {
            "temp": _q(part["temp"]),
            "psal": _q(part["psal"]),
            "temp_missing": float(part["temp"].isna().mean()),
            "psal_missing": float(part["psal"].isna().mean()),
            "by_half": {},
        }
        halves = {
            "sep_01_30": part["local_month"].eq(9),
            "oct_01_15": part["local_month"].eq(10) & part["local_day"].le(15),
            "oct_16_31": part["local_month"].eq(10) & part["local_day"].gt(15),
        }
        for name, mask in halves.items():
            p = part.loc[mask]
            seasonal_summary[str(int(year))]["by_half"][name] = {
                "temp": _q(p["temp"]),
                "temp_missing": float(p["temp"].isna().mean()) if len(p) else None,
            }

    return {
        "gap_start_utc": t0.isoformat(),
        "gap_end_utc": t1.isoformat(),
        "gap_span_days": float((t1 - t0).total_seconds() / 86400),
        "boundary_target_observations": boundary,
        "seasonal_distribution": seasonal_summary,
    }


def audit_p3(data_dir: Path) -> dict:
    wave = pd.read_csv(data_dir / "train_wave.csv")
    wave["time"] = pd.to_datetime(wave["time"], utc=True)
    wave = wave.sort_values(["station", "time"]).reset_index(drop=True)
    leads = [12, 24, 36, 48, 60, 72]
    rows = []
    for station, part in wave.groupby("station", sort=True):
        p = part.set_index("time").sort_index()
        base = p[["hs"]].rename(columns={"hs": "hs0"}).copy()
        base["hs_m12"] = p["hs"].reindex(base.index - pd.Timedelta(hours=12)).to_numpy()
        for lead in leads:
            base[f"hs_p{lead}"] = p["hs"].reindex(base.index + pd.Timedelta(hours=lead)).to_numpy()
        base["station"] = station
        base["time"] = base.index
        rows.append(base.reset_index(drop=True))
    anchor = pd.concat(rows, ignore_index=True)
    complete = anchor.dropna(subset=["hs0", "hs_m12"] + [f"hs_p{x}" for x in leads]).copy()
    high = complete.loc[complete["hs0"].ge(1.5)].copy()
    high["slope12"] = high["hs0"] - high["hs_m12"]
    high["regime"] = pd.cut(
        high["slope12"],
        bins=[-np.inf, -0.2, 0.2, np.inf],
        labels=["falling", "flat", "rising"],
    ).astype(str)
    high["hs_bin"] = pd.cut(
        high["hs0"],
        bins=[1.5, 1.8, 2.2, 3.0, np.inf],
        right=False,
        labels=["1.5-1.8", "1.8-2.2", "2.2-3.0", "3.0+"],
    ).astype(str)

    def evaluate(part: pd.DataFrame) -> dict:
        item = {"anchors": int(len(part)), "by_lead": {}}
        for lead in leads:
            y = part[f"hs_p{lead}"].to_numpy(float)
            p = part["hs0"].to_numpy(float)
            item["by_lead"][str(lead)] = {
                "future_change_mean": float(np.mean(y - p)),
                "future_change_median": float(np.median(y - p)),
                "persistence_rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                "correlation_with_current": float(np.corrcoef(y, p)[0, 1]) if len(part) > 1 else None,
            }
        return item

    greedy_counts = {}
    for station, part in high.groupby("station"):
        selected = []
        last = None
        for t in sorted(part["time"]):
            if last is None or t - last >= pd.Timedelta(hours=78):
                selected.append(t)
                last = t
        greedy_counts[str(station)] = int(len(selected))

    return {
        "complete_high_state_anchors": int(len(high)),
        "greedy_78h_separated_counts": greedy_counts,
        "by_station": {str(k): evaluate(v) for k, v in high.groupby("station")},
        "by_regime": {str(k): evaluate(v) for k, v in high.groupby("regime", observed=True)},
        "by_hs_bin": {str(k): evaluate(v) for k, v in high.groupby("hs_bin", observed=True)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-dir", type=Path, required=True)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--p3-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": "competition_dataset_mechanism_audit.20260828.v1",
        "p1": audit_p1(args.p1_dir),
        "p2": audit_p2(args.p2_dir),
        "p3": audit_p3(args.p3_dir),
        "hidden_answers_read": False,
        "p3_absolute_time_reconstruction_attempted": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
