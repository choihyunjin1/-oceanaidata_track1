"""Zero-fit source-support audit, never a model or final-package certificate."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import ocean_evaluation_contract_v5 as cv
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ocean_forward_support_20260906_v1/result.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def normalize(frame, zone, keys):
    frame = frame.copy()
    parsed = pd.to_datetime(frame.time)
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(zone)
    frame["time"] = parsed.dt.tz_convert("UTC")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("missing/duplicate source keys")
    return frame


def slices(frame, mask, columns):
    part = frame.loc[np.asarray(mask)]
    return [{**dict(zip(columns, [v.item() if isinstance(v, np.generic) else v
                                 for v in (key if isinstance(key, tuple) else (key,))], strict=True)),
             "rows": int(n)}
            for key, n in part.groupby(columns, observed=True).size().items()]


def audit_p1(frame, contract):
    frame = normalize(frame, "Asia/Seoul", ["station", "layer", "time"])
    rows = []
    for fold in contract["P1"]["folds"]:
        if fold["warmup"]:
            continue
        split = cv.p1_split(frame, fold["id"], contract)
        tr, va = split["train"], split["validation"]
        calendar = frame.time.ge(pd.Timestamp(fold["start"])) & frame.time.lt(pd.Timestamp(fold["end"]))
        known = set(map(tuple, frame.loc[tr, ["station", "layer"]].to_numpy()))
        unseen = np.array([tuple(x) not in known for x in frame[["station", "layer"]].to_numpy()])
        rows.append({"fold": fold["id"], "train": int(tr.sum()), "validation": int(va.sum()),
                     "train_positive": int(frame.loc[tr, "label"].sum()),
                     "validation_positive": int(frame.loc[va, "label"].sum()),
                     "boundary_moved_rows": int(np.count_nonzero(va != calendar)),
                     "unseen_station_layer_validation_rows": int((va & unseen).sum()),
                     "validation_days": len(set(cv.bootstrap_groups(frame.loc[va], "P1", contract))),
                     "train_slices": slices(frame, tr, ["station", "layer"]),
                     "validation_slices": slices(frame, va, ["station", "layer"])})
    return {"source_rows": len(frame), "folds": rows,
            "feature_dependency_audit": "NOT_RUN", "model_fit": 0}


def audit_p2(frame, contract):
    frame = normalize(frame, "Asia/Seoul", ["station", "layer", "time"])
    public = frame.loc[~frame.layer.isin([2, 3, 4])].copy()
    public["finite_temp"] = np.isfinite(public.temp)
    public["finite_temp_depth"] = np.isfinite(public.temp) & np.isfinite(public.depth) & public.depth.gt(0)
    counts = public.groupby(["station", "time"])[["finite_temp", "finite_temp_depth"]].sum()
    targets = frame.loc[frame.layer.isin([2, 3, 4]), ["station", "time", "layer", "temp"]].copy()
    targets = targets.join(counts, on=["station", "time"])
    # This is only the observable support proxy, not hidden organizer QC.
    eligible = np.isfinite(targets.temp) & targets.finite_temp.ge(2)
    target = targets.loc[eligible].reset_index(drop=True)
    five = public.loc[public.layer.eq(5), ["station", "time", "finite_temp"]]
    target = target.merge(five.rename(columns={"finite_temp": "five_finite"}),
                          on=["station", "time"], how="left", validate="many_to_one")
    target["five_finite"] = target.five_finite.fillna(False).astype(bool)
    rows = []
    for fold in contract["P2"]["folds"]:
        split = cv.p2_split(target, fold["id"], contract, ["station", "layer", "time"])
        tr, va, outage = (split[key] for key in ("train", "validation", "outage"))
        remaining = target.finite_temp - target.five_finite.astype(int)
        unsupported = outage & remaining.lt(2).to_numpy()
        rows.append({"fold": fold["id"], "train": int(tr.sum()), "validation": int(va.sum()),
                     "outage_target_rows": int(outage.sum()),
                     "outage_under_two_public_temp_rows": int(unsupported.sum()),
                     "natural_missing_T5_target_rows": int((va & ~target.five_finite).sum()),
                     "under_two_temp_positive_depth_rows": int((va & target.finite_temp_depth.lt(2)).sum()),
                     "validation_7day_clusters": len(set(cv.bootstrap_groups(target.loc[va], "P2", contract))),
                     "layers": slices(target, va, ["layer"]),
                     "outage_support": "UNSUPPORTED_ROWS_PRESERVED" if unsupported.any() else "COUNT_SUPPORT_PASS"})
    return {"source_rows": len(frame), "eligible_target_proxy_rows": len(target), "folds": rows,
            "hidden_QC": "UNAVAILABLE_NOT_READ", "derived_feature_mask_audit": "NOT_RUN",
            "support_definition": "finite target temp and >=2 non-target finite temperatures; not actual model adapter eligibility",
            "model_fit": 0}


def wave_anchors(wave):
    """Raw-wave high contiguous 20-min runs; same definition as clean revin_patch.

    No six-hour storm merging or target-dependent episode boundaries. Targets
    are used only for historical anchor eligibility, never episode assignment.
    """
    wave = normalize(wave, "UTC", ["station", "time"])
    pieces = []
    for station, part in wave.groupby("station", sort=True):
        part = part.sort_values("time").set_index("time")
        high = part.hs.ge(1.5) & np.isfinite(part.hs)
        starts = high & (~high.shift(fill_value=False) | ~part.index.to_series().diff().eq(pd.Timedelta(minutes=20)))
        episodes = starts.cumsum()
        keep = high & (part.index >= part.index.min() + pd.Timedelta(hours=48))
        for lead in [3, 6, 9, 12, 18, 24]:
            future = part.hs.reindex(part.index + pd.Timedelta(hours=lead)).to_numpy()
            keep &= np.isfinite(future)
        chosen = part.loc[keep]
        pieces.append(pd.DataFrame({"station": station, "anchor_time": chosen.index,
                                    "episode_id": episodes.loc[keep].to_numpy()}))
    result = pd.concat(pieces, ignore_index=True)
    if result.empty:
        raise ValueError("no six-lead anchors")
    return result


def audit_p3(wave, contract):
    anchors = wave_anchors(wave)
    rows = []
    for fold in contract["P3"]["folds"]:
        if fold["warmup"]:
            continue
        split = cv.p3_split(anchors, fold["id"], contract)
        tr, va = split["train"], split["validation"]
        cutoff = pd.Timestamp(fold["start"]) - pd.Timedelta(hours=78)
        raw_train = anchors.anchor_time.lt(cutoff).to_numpy()
        rows.append({"fold": fold["id"], "train_anchors": int(tr.sum()), "validation_anchors": int(va.sum()),
                     "six_lead_rows": int(va.sum() * 6),
                     "episode_excluded_after_purge": int((raw_train & ~tr).sum()),
                     "validation_station_episodes": len(set(cv.bootstrap_groups(anchors.loc[va], "P3", contract))),
                     "train_stations": slices(anchors, tr, ["station"]),
                     "validation_stations": slices(anchors, va, ["station"])})
    return {"source_rows": len(wave), "dense_eligible_anchors": len(anchors), "folds": rows,
            "episode_definition": "station-specific contiguous 20-min hs>=1.5 run; gap/missing/low ends run",
            "independence_caveat": "different runs/stations may share the same meteorological storm",
            "feature_atmos_support_audit": "NOT_RUN", "model_fit": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Plan: read only P1 train.csv, P2 observations.csv, P3 train_wave.csv; fits=0")
        return
    if OUT.exists():
        raise FileExistsError("existing audit receipt preserved")
    started = time.perf_counter()
    contract = cv.load_contract()
    sources = {"P1": Path(os.environ["P1_DATA_DIR"]) / "train.csv",
               "P2": Path(os.environ["P2_DATA_DIR"]) / "observations.csv",
               "P3": Path(os.environ["P3_DATA_DIR"]) / "train_wave.csv"}
    before = {p: sha(path) for p, path in sources.items()}
    columns = {"P1": ["station", "layer", "time", "label"],
               "P2": ["station", "layer", "time", "temp", "psal", "depth"],
               "P3": ["station", "time", "hs"]}
    result = {"status": "SOURCE_COUNT_AUDIT_COMPLETE_FEATURE_ADAPTER_AUDIT_PENDING",
              "contract_sha256": sha(cv.DEFAULT_CONFIG), "runner_sha256": sha(__file__),
              "evaluator_sha256": sha(cv.__file__), "source_sha256": before,
              "model_fits": 0, "official_inputs": 0, "answer_csv": 0, "upload": 0,
              "access_scope": "explicit source-only reads in this runner, not an OS-wide access certificate"}
    for problem, auditor in [("P1", audit_p1), ("P2", audit_p2), ("P3", audit_p3)]:
        tick = time.perf_counter()
        result[problem] = auditor(pd.read_csv(sources[problem], usecols=columns[problem]), contract)
        result[problem]["seconds"] = time.perf_counter() - tick
        print(json.dumps({"problem": problem, "count_audit_complete": True}), flush=True)
    after = {p: sha(path) for p, path in sources.items()}
    if before != after:
        raise ValueError("source changed during audit")
    result["source_unchanged"] = True
    result["seconds"] = time.perf_counter() - started
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
