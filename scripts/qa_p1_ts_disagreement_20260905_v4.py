"""Independent confusion-count aggregation and complete policy replay."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_p1_ts_disagreement_20260905_v4 as run  # noqa: E402


def independent_counts(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("binary aligned input")
    tn, fp, fn, tp = np.bincount((2 * y + p).astype(int), minlength=4).tolist()
    return {
        "rows": tn + fp + fn + tp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def audit():
    cfg, source, prior, frozen = run.load_contract()
    result = json.loads((run.OUT / "terminal_result.json").read_text(encoding="utf-8"))
    checks = {
        "terminal_complete": result["status"] == "COMPLETE",
        "six_fits": len(result["fits"]) == 6,
        "fresh_pid": result["pid"] != os.getpid(),
        "runner_hash": result["runner_sha256"] == run.old.sha(run.__file__),
        "config_hash": result["config_sha256"] == run.old.sha(run.CONFIG),
    }
    if not all(checks.values()):
        raise ValueError("terminal contract incomplete")
    frame = run.load_training(frozen).set_index("row_id", drop=False)
    for fit in result["fits"]:
        checks[f"model/{fit['fold']}/{fit['stage']}"] = (
            run.old.sha(run.OUT / f"{fit['fold']}_{fit['stage']}.joblib") == fit["model_sha256"]
        )
    all_parts = []
    for name, expected in result["artifact_hashes"].items():
        checks[f"artifact/{name}"] = run.old.sha(run.OUT / name) == expected
        part = pd.read_parquet(run.OUT / name)
        fold = part.fold.iloc[0]
        ev = frame.loc[part.row_id].reset_index(drop=True)
        checks[f"keys/{name}"] = not part.duplicated(run.old.KEYS).any()
        checks[f"targets/{name}"] = np.array_equal(ev.label, part.label)
        pack = joblib.load(run.OUT / f"{fold}_outer.joblib")
        pred = pack["model"].predict_proba(
            pack["encoder"].transform(run.bundle(ev, pack["stats"], frozen, pack["epsilon"]))
        )[:, 1]
        old_pack = joblib.load(source / f"{fold}_outer_original.joblib")
        original = old_pack["model"].predict_proba(
            old_pack["encoder"].transform(run.bundle(ev, old_pack["train_stats"], frozen))
        )[:, 1]
        selection = next(x for x in result["folds"] if x["fold"] == fold)["selections"]["candidate"]
        bits = run.evaluate(
            ev,
            {"original": original, "balanced": pred},
            run.old.rule_masks(ev, pack["stats"]),
            frozen,
            selection,
        )
        checks[f"probability/{name}"] = np.array_equal(pred, part.candidate_probability)
        checks[f"policy_bits/{name}"] = np.array_equal(bits, part.candidate)
        all_parts.append(part)
    pooled = pd.concat(all_parts, ignore_index=True)
    slices = []
    for surface, part in pooled.groupby("surface"):
        checks[f"disjoint_folds/{surface}"] = not part.duplicated(run.old.KEYS).any()
        for arm in ("control", "candidate"):
            checks[f"arithmetic/{surface}/{arm}"] = (
                independent_counts(part.label, part[arm]) == result["metrics"][surface][arm]
            )
        for dimensions in (["fold"], ["station"], ["anomaly_type"]):
            for key, block in part.groupby(dimensions, dropna=False):
                slices.append(
                    {
                        "surface": surface,
                        "dimension": dimensions[0],
                        "value": str(key[0]),
                        **{
                            arm: independent_counts(block.label, block[arm])
                            for arm in ("control", "candidate")
                        },
                    }
                )
    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "check_count": len(checks),
        "passed": sum(bool(x) for x in checks.values()),
        "slices": slices,
        "training_pid": result["pid"],
        "replay_pid": os.getpid(),
        "source_result_sha256": run.old.sha(run.OUT / "terminal_result.json"),
        "new_fits": 0,
        "official_rows": 0,
        "csv_written": 0,
        "uploads": 0,
        "baseline_regeneration_check": "SEPARATE_REPORT_REQUIRED_NOT_CERTIFIED_BY_THIS_REPLAY",
    }
    run.old.write_json(run.REPORT / "policy-replay-qa.json", output)
    if not all(checks.values()):
        raise ValueError("policy replay failed")
    print(json.dumps({"status": output["status"], "checks": len(checks)}))


if __name__ == "__main__":
    audit()
