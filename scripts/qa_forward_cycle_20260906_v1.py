"""Root-owned independent arithmetic audit of completed internal v5 OOF only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IDS = {
    "p1": "p1_bracket_forward_20260906_v1",
    "p2": "p2_crossfit_copula_forward_20260906_v1",
    "p3": "p3_numeric_lead_forward_20260906_v1",
}


def metric(truth, prediction, name):
    y, p = np.asarray(truth), np.asarray(prediction)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("empty or misaligned evaluation")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("nonfinite evaluation")
    if name == "f1":
        if not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
            raise ValueError("F1 requires binary labels")
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        return {"value": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                "tp": tp, "fp": fp, "fn": fn, "n": len(y)}
    if name != "rmse":
        raise ValueError("unknown metric")
    residual = y.astype(np.float64) - p.astype(np.float64)
    sse = float(np.dot(residual, residual))
    return {"value": float(np.sqrt(sse / len(y))), "sse": sse, "n": len(y)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(frame, truth, arms, name):
    if not len(frame):
        return {"status": "NOT_ESTIMABLE", "rows": 0}
    values = {arm: metric(frame[truth], frame[arm], name) for arm in arms}
    return {"status": "COMPUTED", "metrics": values,
            "deltas": {arm: values[arm]["value"] - values[arms[0]]["value"]
                       for arm in arms[1:]}}


def audit(problem):
    folder = ROOT / "artifacts" / IDS[problem]
    terminal_path = (ROOT / "reports" / IDS[problem] / "result.json"
                     if problem == "p3" else folder / "terminal_result.json")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    completed = terminal["status"].startswith("COMPLETE")
    if problem == "p3":
        marker = json.loads((folder / "TERMINAL.json").read_text(encoding="utf-8"))
        completed = (marker["result_sha256"] == sha(terminal_path)
                     and terminal["status"] in {"MEAN_IMPROVEMENT_CANDIDATE_RETAINED",
                                                 "NO_OP_NUMERIC_LEAD_NOT_IMPROVED"})
    if not completed:
        raise ValueError("completed experiment required; no early metrics")
    if problem == "p1":
        path = folder / "oof.parquet"
        frame = pd.read_parquet(path)
        keys, truth, arms, name = ["station", "layer", "time"], "label", ["control", "candidate"], "f1"
        expected = 668593
        time = pd.to_datetime(frame.time, utc=True).dt.tz_convert("Asia/Seoul")
        primary = (time >= "2025-01-01") & (time < "2025-07-01")
        assert primary.sum() == 208093
    elif problem == "p2":
        path = folder / "evaluation.npz"
        with np.load(path, allow_pickle=False) as raw:
            frame = pd.DataFrame({key: raw[key] for key in raw.files})
        keys, truth = ["key"], "truth"
        arms = ["natural_C3", "natural_insample_full", "natural_crossfit_full"]
        name, expected, primary = "rmse", 166268, frame.fold.eq("B3")
        assert primary.sum() == 26273
    else:
        path = folder / "paired_oof.parquet"
        frame = pd.read_parquet(path)
        keys, truth = ["fold", "anchor_id", "station", "lead_h"], "target_hs"
        arms, name, expected = ["control", "candidate"], "rmse", 103602
        primary = np.ones(len(frame), dtype=bool)
        assert set(frame.lead_h) == {3, 6, 9, 12, 18, 24}
        assert frame.groupby(["fold", "anchor_id", "station"]).size().eq(6).all()
    assert len(frame) == expected and not frame.duplicated(keys).any()
    panels = {"primary": compare(frame.loc[primary], truth, arms, name),
              "pooled": compare(frame, truth, arms, name)}
    panels["folds"] = {str(fold): compare(part, truth, arms, name)
                       for fold, part in frame.groupby("fold", sort=True)}
    if problem == "p2":
        panels["outage"] = {
            str(fold): compare(part.loc[part.outage_mask], truth,
                               ["outage_C3", "outage_insample_full", "outage_crossfit_full"], name)
            for fold, part in frame.groupby("fold", sort=True)
        }
        assert panels["outage"]["B4"]["status"] == "NOT_ESTIMABLE"
        assert panels["outage"]["B8"]["status"] == "NOT_ESTIMABLE"
    return {"problem": problem, "experiment_id": IDS[problem],
            "status": "ROOT_ARITHMETIC_AND_KEY_QA_PASS", "metric": name,
            "terminal_sha256": sha(terminal_path), "oof_sha256": sha(path),
            "panels": panels, "fits": 0, "official_inputs": 0, "csv_written": 0,
            "scope": "Independent pooled arithmetic and key/count checks; model lineage, replay and CI require linked agent QA."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=IDS)
    args = parser.parse_args()
    result = audit(args.problem)
    output = ROOT / "reports" / IDS[args.problem] / "root-arithmetic-qa.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"problem": args.problem, "status": result["status"]}))


if __name__ == "__main__":
    main()
