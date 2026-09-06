"""Read-only historical composed-error audit: metadata/labels only, zero fits."""

# ruff: noqa: E402 -- thread limits must precede numerical library imports
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("original_union", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/evaluate_union_v2.py")
u = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(u)
ID = "p1_composed_error_audit_20260906_v1"
KEYS = ["station", "year", "layer", "time"]
SOURCE_SHA = "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2"
PAIRED_SHA = "d4f5742ca60e4883f69a8271841b27dd281a91fec19403a69e471b7a3368912c"
UNION_SHA = "24ca88594c4cdb5d5dc2c3ae5defa1415823f8cfcebe463820b140dd9376146f"
EDGE = 12
TYPES = {"spike", "noise", "flatline", "offset", "drift"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def keys(frame):
    return u.engine.composition.canonical_keys(frame)


def check_binary(values):
    x = np.asarray(values)
    if x.ndim != 1 or not len(x) or not np.isin(x, [0, 1]).all():
        raise ValueError("invalid binary vector")
    return x.astype(np.int8)


def type_group(label, value):
    if label == 0:
        if pd.notna(value):
            raise ValueError("normal row unexpectedly has injected anomaly type")
        return "normal"
    if pd.isna(value):
        raise ValueError("positive row lacks anomaly type")
    atoms = set(str(value).split("+"))
    if not atoms or not atoms <= TYPES:
        raise ValueError("unrecognized anomaly type")
    return next(iter(atoms)) if len(atoms) == 1 else "mixed"


def duration_bin(rows):
    return pd.cut(rows, [0, 1, 12, 36, 144, 432, np.inf],
                  labels=["one_10min_row", "20min_to_2h", "over2_to_6h", "over6_to_24h", "over24_to_72h", "over72h"],
                  right=True).astype(str)


def annotate_source(frame):
    """True-label runs use complete released history, not prediction-derived events."""
    out = keys(frame)
    out["label"] = check_binary(frame.label)
    out["anomaly_type"] = frame.anomaly_type.to_numpy()
    out.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    out.reset_index(drop=True, inplace=True)
    out["type"] = [type_group(y, typ) for y, typ in zip(out.label, out.anomaly_type, strict=True)]
    out["raw_type"] = out.anomaly_type.fillna("normal").astype(str)
    group = out.station.ne(out.station.shift()) | out.layer.ne(out.layer.shift())
    gap = out.time.diff().ne(pd.Timedelta(minutes=10))
    out["truth_run"] = (group | gap | out.label.ne(out.label.shift())).cumsum()
    out["truth_run_rows"] = out.groupby("truth_run", sort=False).label.transform("size")
    pos = out.groupby("truth_run", sort=False).cumcount()
    out["truth_run_edge"] = (pos < EDGE) | (out.truth_run_rows - 1 - pos < EDGE)
    out["truth_duration"] = duration_bin(out.truth_run_rows)
    out.loc[out.label.eq(0), "truth_duration"] = "normal"
    # Label-based diagnostic only: not a train feature or a proposed online rule.
    time_ns = out.time.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    positive_ns = pd.Series(np.where(out.label.eq(1), time_ns, np.nan))
    grouping = [out.station, out.layer]
    before = positive_ns.groupby(grouping).ffill().to_numpy()
    after = positive_ns.groupby(grouping).bfill().to_numpy()
    distance = np.fmin(np.abs(time_ns - before), np.abs(after - time_ns))
    out["normal_near_anomaly_2h"] = out.label.eq(0) & (distance <= pd.Timedelta(hours=2).value)
    return out


def exact_join(paired, source):
    left = keys(paired)
    li = pd.MultiIndex.from_frame(left)
    right = pd.MultiIndex.from_frame(source[KEYS])
    index = right.get_indexer(li)
    if (index < 0).any() or len(np.unique(index)) != len(index):
        raise ValueError("full one-to-one source metadata coverage required")
    result = source.iloc[index].reset_index(drop=True).copy()
    if not np.array_equal(result.label, paired.label):
        raise ValueError("source and OOF target mismatch")
    for col in ("fold", "union", "proposal", "union_OR_e150"):
        result[col] = paired[col].to_numpy()
    for col in ("union", "proposal", "union_OR_e150"):
        check_binary(result[col])
    if not np.array_equal(result.union | result.proposal, result.union_OR_e150):
        raise ValueError("fixed composition changed")
    result.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    result.reset_index(drop=True, inplace=True)
    boundary = result.station.ne(result.station.shift()) | result.layer.ne(result.layer.shift()) | result.fold.ne(result.fold.shift()) | result.time.diff().ne(pd.Timedelta(minutes=10))
    segment = boundary.cumsum()
    position = result.groupby(segment).cumcount()
    count = result.groupby(segment).label.transform("size")
    result["partition_edge"] = (position < EDGE) | (count - 1 - position < EDGE)
    for kind, mask in (("both_FN", result.label.eq(1) & result.union_OR_e150.eq(0)),
                       ("composed_FP", result.label.eq(0) & result.union_OR_e150.eq(1))):
        result[kind] = mask
        run = (boundary | mask.ne(mask.shift())).cumsum()
        result[kind + "_run"] = np.where(mask, run, -1)
    return result


def counts(frame):
    y = check_binary(frame.label)
    tree, ms, pred = [check_binary(frame[k]) for k in ("union", "proposal", "union_OR_e150")]
    if not np.array_equal(pred, tree | ms):
        raise ValueError("composition mismatch")
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    output = {"rows": len(y), "positive": int(y.sum()), "normal": int((y == 0).sum()),
              "tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.,
              "FN_rate_positive": fn / int(y.sum()) if y.sum() else None,
              "FP_rate_normal": fp / int((y == 0).sum()) if (y == 0).any() else None}
    for label, prefix in ((1, "positive"), (0, "normal")):
        for bit_t, bit_m, name in ((0, 0, "neither"), (1, 0, "tree_only"), (0, 1, "MS_only"), (1, 1, "both")):
            output[prefix + "_" + name] = int(((y == label) & (tree == bit_t) & (ms == bit_m)).sum())
    return output


def grouped(frame, columns):
    rows = []
    for key, part in frame.groupby(columns, sort=True, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        row = {col: value.item() if isinstance(value, np.generic) else value for col, value in zip(columns, values, strict=True)}
        rows.append({"slice": row, **counts(part)})
    return rows


def events(frame):
    positive = frame.loc[frame.label.eq(1)]
    rows = []
    for _, part in positive.groupby("truth_run", sort=False):
        length = int(part.truth_run_rows.iloc[0])
        if len(part) != length:
            raise ValueError("partial positive event would bias full-miss rates")
        atoms = set("+".join(part.anomaly_type.astype(str)).split("+"))
        rows.append({"type": next(iter(atoms)) if len(atoms) == 1 else "mixed",
            "duration": str(duration_bin(pd.Series([length])).iloc[0]), "rows": length,
            "fold": str(part.fold.iloc[0]), "station": str(part.station.iloc[0]), "layer": int(part.layer.iloc[0]),
            "miss_rows": int(part.both_FN.sum()), "fully_missed": bool(part.both_FN.all()),
            "partially_missed": bool(part.both_FN.any() and not part.both_FN.all()),
            "fully_detected": bool(not part.both_FN.any())})
    data = pd.DataFrame(rows)
    summaries = {}
    for cols in (["type"], ["duration"], ["type", "duration"], ["station", "layer"]):
        parts = []
        for key, part in data.groupby(cols, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            parts.append({"slice": {k: v.item() if isinstance(v, np.generic) else v for k, v in zip(cols, values, strict=True)},
                "events": len(part), "positive_rows": int(part.rows.sum()), "miss_rows": int(part.miss_rows.sum()),
                "fully_missed_events": int(part.fully_missed.sum()), "partially_missed_events": int(part.partially_missed.sum()),
                "fully_detected_events": int(part.fully_detected.sum())})
        summaries["/".join(cols)] = parts
    return data, summaries


def error_bursts(frame, name):
    active = frame.loc[frame[name]]
    if active.empty:
        return []
    result = []
    for _, part in active.groupby(name + "_run", sort=False):
        result.append({"length_rows": len(part), "duration": str(duration_bin(pd.Series([len(part)])).iloc[0]),
                       "type": part.type.iloc[0] if part.type.nunique() == 1 else "mixed"})
    d = pd.DataFrame(result)
    return [{"duration": key, "bursts": len(part), "error_rows": int(part.length_rows.sum()),
             "max_burst_rows": int(part.length_rows.max())} for key, part in d.groupby("duration", sort=True)]


def run():
    start = time.monotonic()
    directory = ROOT / "artifacts" / ID
    if directory.exists():
        raise FileExistsError("new audit output required; no overwritten results")
    original = ROOT / "artifacts/p1_champion_reconstruction_20260906_v2_union_evaluation"
    cfg = load(u.CONTRACT)
    old = load(original / "result.json")
    if sha(original / "result.json") != UNION_SHA or sha(original / "paired-evaluation.parquet") != PAIRED_SHA or old["paired_artifact_sha256"] != PAIRED_SHA:
        raise ValueError("original composed OOF/result fingerprint mismatch")
    for filename in ("independent-qa.json", "independent-bootstrap-qa.json"):
        qa = load(original / filename)
        if qa["status"] != "PASS":
            raise ValueError("original independent QA missing")
    paired = pd.read_parquet(original / "paired-evaluation.parquet", columns=KEYS + ["fold", "label", "union", "proposal", "union_OR_e150"])
    # Original Q3->Q4 source key digest establishes owner; do not recalendarize.
    proposal_dir = ROOT / cfg["inputs"]["proposal_directory"]
    if sha(proposal_dir / "proposals.parquet") != cfg["inputs"]["proposal_sha256"]:
        raise ValueError("fixed original MS proposal changed")
    proposal_keys = pd.read_parquet(proposal_dir / "proposals.parquet", columns=KEYS)
    owned = u.original_fold_ownership(proposal_keys, cfg)
    left = keys(paired).assign(fold=paired.fold.to_numpy())
    li, ri = [pd.MultiIndex.from_frame(x[KEYS + ["fold"]]) for x in (left, owned)]
    if len(paired) != 287862 or not li.isin(ri).all() or not ri.isin(li).all() or len(li) != len(ri):
        raise ValueError("whole original key/fold parity required")
    source_path = Path(os.environ["P1_DATA_DIR"]) / "train.csv"
    if sha(source_path) != SOURCE_SHA:
        raise ValueError("distributed train SHA mismatch")
    source = pd.read_csv(source_path, usecols=KEYS + ["label", "anomaly_type"])
    if len(source) != 776706:
        raise ValueError("source population mismatch")
    source = annotate_source(source)
    frame = exact_join(paired, source)
    totals = counts(frame)
    expected = old["scopes"]["primary_Q3_Q4"]["union"]["candidate"]
    if any(totals[k] != expected[k] for k in ("rows", "tp", "fp", "fn", "tn", "f1")):
        raise ValueError("frozen composition score changed")
    audits = {"type": grouped(frame, ["type"]), "raw_type": grouped(frame, ["raw_type"]),
              "type_duration": grouped(frame, ["type", "truth_duration"]),
              "station_layer": grouped(frame, ["station", "layer"]), "fold": grouped(frame, ["fold"]),
              "anomaly_event_edge": grouped(frame.loc[frame.label.eq(1)], ["truth_run_edge"]),
              "type_event_edge": grouped(frame.loc[frame.label.eq(1)], ["type", "truth_run_edge"]),
              "partition_edge": grouped(frame, ["partition_edge"]),
              "normal_near_anomaly": grouped(frame.loc[frame.label.eq(0)], ["normal_near_anomaly_2h"])}
    event_table, event_summary = events(frame)
    checks = []
    for table, values in audits.items():
        for row in values:
            mask = np.ones(len(frame), dtype=bool)
            for key, value in row["slice"].items():
                mask &= frame[key].eq(value).to_numpy()
            if table in ("anomaly_event_edge", "type_event_edge"):
                mask &= frame.label.eq(1).to_numpy()
            if table == "normal_near_anomaly":
                mask &= frame.label.eq(0).to_numpy()
            part = frame.loc[mask]
            tn, fp, fn, tp = confusion_matrix(part.label, part.union_OR_e150, labels=[0, 1]).ravel()
            if [int(tp), int(fp), int(fn), int(tn)] != [row[k] for k in ("tp", "fp", "fn", "tn")] or abs(f1_score(part.label, part.union_OR_e150, zero_division=0) - row["f1"]) > 1e-12:
                raise ValueError("independent grouped confusion/F1 mismatch")
            if row["positive_neither"] != row["fn"] or sum(row["normal_" + k] for k in ("tree_only", "MS_only", "both")) != row["fp"]:
                raise ValueError("error component attribution mismatch")
            checks.append(table + "/group")
        expected_rows = totals["positive"] if table in ("anomaly_event_edge", "type_event_edge") else totals["normal"] if table == "normal_near_anomaly" else totals["rows"]
        if sum(row["rows"] for row in values) != expected_rows:
            raise ValueError("group partition denominator mismatch")
        checks.append(table + "/partition_total")
    # Independent scalar run segmentation cross-checks the vectorized source metadata.
    scalar_start, scalar_rows, n = 0, np.empty(len(source), dtype=np.int64), len(source)
    arrays = [source[c].to_numpy() for c in ("station", "layer", "time", "label")]
    for i in range(1, n + 1):
        boundary = i == n or arrays[0][i] != arrays[0][i - 1] or arrays[1][i] != arrays[1][i - 1] or arrays[3][i] != arrays[3][i - 1] or arrays[2][i] - arrays[2][i - 1] != pd.Timedelta(minutes=10)
        if boundary:
            scalar_rows[scalar_start:i] = i - scalar_start
            scalar_start = i
    if not np.array_equal(scalar_rows, source.truth_run_rows):
        raise ValueError("independent truth run-length mismatch")
    checks.append("independent_scalar_source_run_lengths")
    if int(event_table.rows.sum()) != totals["positive"] or int(event_table.miss_rows.sum()) != totals["fn"]:
        raise ValueError("whole positive-event totals mismatch")
    checks.append("complete_event_positive_and_miss_denominators")
    bursts = {name: error_bursts(frame, name) for name in ("both_FN", "composed_FP")}
    if sum(v["error_rows"] for v in bursts["both_FN"]) != totals["fn"] or sum(v["error_rows"] for v in bursts["composed_FP"]) != totals["fp"]:
        raise ValueError("error burst row totals mismatch")
    checks.append("error_burst_totals")
    boundary_spill = 0
    for fold in cfg["folds"]:
        part = frame.loc[frame.fold.eq(fold["id"])]
        boundary_spill += int((part.time.ge(pd.Timestamp(fold["end"])) | part.time.lt(pd.Timestamp(fold["start"]))).sum())
    if sha(source_path) != SOURCE_SHA or sha(original / "paired-evaluation.parquet") != PAIRED_SHA:
        raise ValueError("source changed during read-only audit")
    directory.mkdir(parents=True)
    result = {"id": ID, "status": "COMPLETE_SHARE_WITH_CAVEATS", "totals": totals, "groups": audits,
        "positive_events": {"count": len(event_table), "fully_missed": int(event_table.fully_missed.sum()),
                            "partially_missed": int(event_table.partially_missed.sum()), "fully_detected": int(event_table.fully_detected.sum()),
                            "groups": event_summary}, "error_bursts": bursts,
        "coverage": {"rows": len(frame), "source_rows": len(source), "missing_keys": 0, "duplicate_keys": 0,
                     "target_mismatch": 0, "excluded_rows": 0, "calendar_boundary_spill_retained": boundary_spill},
        "definitions": {"event": "complete released-history continuous10min station-layer positive label run; not generator event identity",
                        "type": "unique anomaly_type atoms; overlaps of different types=mixed; raw_type retained separately",
                        "duration": "truth run rows times10min; one-row and <=2h/6h/24h/72h/>72h diagnostic bins",
                        "truth_edge": "first/last12rows(2h) of complete truth run; short events can be entirely edge",
                        "partition_edge": "first/last12rows of evaluated fold-owned continuous segment",
                        "near_anomaly": "normal row within actual2h of positive label in source; label-based diagnostic only, never inference feature"},
        "input_sha256": {"train": SOURCE_SHA, "paired": PAIRED_SHA, "original_result": UNION_SHA,
                         "union_contract": sha(u.CONTRACT), "post_qa_decision": sha(ROOT / "reports/p1_champion_reconstruction_20260906_v1/post-qa-decision.json")},
        "source_sha256": sha(__file__), "runtime_seconds": time.monotonic() - start,
        "fits": 0, "official_rows": 0, "hidden_rows": 0, "csv_outputs": 0, "upload": 0, "locks_created": 0,
        "caveats": ["selected repeatedly exposed Q3/Q4; descriptive error mining, not fresh validation",
                    "historical crossfit composition corresponding to frozen b2f17 policy, not official b2f17 error labels",
                    "anomaly_type and true-run boundary are diagnosis-only labels; unavailable to deploy",
                    "FP types are normal by truth; no claim of physical sensor failure or removal/downweighting",
                    "no prediction/model/threshold/parameter change or extra fit"]}
    save(directory / "result.json", result)
    save(directory / "independent-qa.json", {"status": "PASS", "checks": checks, "check_count": len(checks),
        "result_sha256": sha(directory / "result.json"), "pid": os.getpid(), "method": "sklearn grouped confusion/F1; scalar run segmentation; attribution/event/burst partitions",
        "fits": 0, "official_rows": 0, "raw_rows_printed": 0})
    print({"status": result["status"], "rows": len(frame), "checks": len(checks), "fits": 0})


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    with threadpool_limits(limits=1):
        run()
