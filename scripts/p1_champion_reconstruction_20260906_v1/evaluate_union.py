"""Zero-fit, exact-key retrospective assessment of fixed P1 e150 OR proposals.

Only a COMPLETE 24-fit tree terminal unlocks its training-derived OOF labels.
No original data, old router/candidate arrays, model, official input or CSV I/O.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / "reports/p1_champion_reconstruction_20260906_v1"
CONTRACT = REPORT / "union-contract.json"
SEAL = REPORT / "union-seal.json"
TEST = ROOT / "tests/test_p1_champion_union_20260906_v1.py"
sys.path.insert(0, str(HERE))
import composition
import numpy as np
import pandas as pd

ARMS = ("O", "B", "union", "router")
KEYS = list(composition.KEYS)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def source_pins():
    files = (Path(__file__), HERE / "composition.py", CONTRACT, TEST)
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def seal():
    save_json(SEAL, {"status": "SEALED_BEFORE_TREE_RESULTS", "created_unix": time.time(),
                    "source_pins": source_pins(),
                    "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "scikit-learn")}})


def checked_contract():
    frozen = read_json(SEAL)
    if frozen["source_pins"] != source_pins():
        raise ValueError("union source/contract/test seal changed")
    if any(importlib.metadata.version(k) != v for k, v in frozen["packages"].items()):
        raise ValueError("union package seal changed")
    cfg = read_json(CONTRACT)
    if tuple(cfg["tree_arms"]) != ARMS or cfg["fit_budget"] != 0:
        raise ValueError("fixed zero-fit arm contract violated")
    return cfg


def attach_proposal_folds(frame, cfg):
    out = composition.canonical_keys(frame)
    out["fold"] = ""
    for spec in cfg["folds"]:
        mask = out.time.ge(pd.Timestamp(spec["start"])) & out.time.lt(pd.Timestamp(spec["end"]))
        if out.loc[mask, "fold"].ne("").any():
            raise ValueError("overlapping registered proposal folds")
        out.loc[mask, "fold"] = spec["id"]
    if out.fold.eq("").any():
        raise ValueError("proposal keys outside registered Q3/Q4 periods")
    return out


def key_coverage(tree_keys, proposal_keys, cfg):
    """No target/prediction read, no intersection score, no row values in report."""
    left = composition.canonical_keys(tree_keys)
    left["fold"] = tree_keys.fold.to_numpy()
    right = attach_proposal_folds(proposal_keys, cfg)
    allowed = {f["id"] for f in cfg["folds"]}
    if not set(left.fold).issubset(allowed):
        raise ValueError("unexpected tree fold in primary selection")
    li = pd.MultiIndex.from_frame(left[KEYS + ["fold"]])
    ri = pd.MultiIndex.from_frame(right[KEYS + ["fold"]])
    missing, extra = ~li.isin(ri), ~ri.isin(li)
    detail = []
    for spec in cfg["folds"]:
        name = spec["id"]
        lm, rm = left.fold.eq(name).to_numpy(), right.fold.eq(name).to_numpy()
        detail.append({"fold": name, "tree_rows": int(lm.sum()), "proposal_rows": int(rm.sum()),
                       "tree_without_proposal": int((lm & missing).sum()),
                       "proposal_without_tree": int((rm & extra).sum())})
    report = {"tree_rows": len(left), "proposal_rows": len(right),
              "matched_rows": int((~missing).sum()), "tree_without_proposal": int(missing.sum()),
              "proposal_without_tree": int(extra.sum()), "folds": detail,
              "same_keys_and_folds": not missing.any() and not extra.any(),
              "implicit_intersection_rows_scored": 0}
    return report, ri.get_indexer(li) if report["same_keys_and_folds"] else None


def chronology(proposal_keys, qa, cfg):
    right = attach_proposal_folds(proposal_keys, cfg)
    records = []
    for spec in cfg["folds"]:
        frame = right.loc[right.fold.eq(spec["id"])]
        meta = qa["folds"][spec["phase"]]
        actual = pd.Timestamp(meta["training_max_time_utc"]).tz_convert("UTC")
        expected = pd.Timestamp(spec["training_max_utc"]).tz_convert("UTC")
        if actual != expected or len(frame) != meta["rows"] or not len(frame):
            raise ValueError("historical MS-TCN cutoff/holdout population mismatch")
        gap = (frame.time.min() - actual).total_seconds() / 3600
        if gap + 1e-10 < cfg["registered_minimum_holdout_gap_hours"] or gap < cfg["required_feature_dependency_hours"]:
            raise ValueError("historical holdout gap insufficient for registered feature dependency")
        records.append({"fold": spec["id"], "rows": len(frame),
                        "training_max_utc": actual.isoformat(),
                        "holdout_min_utc": frame.time.min().isoformat(),
                        "minimum_gap_hours": gap, "required_dependency_hours": cfg["required_feature_dependency_hours"],
                        "status": "PASS"})
    return records


def counts(truth, prediction):
    y, p = np.asarray(truth), np.asarray(prediction)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("binary same-row metric contract violated")
    return np.array([np.sum((y == 1) & (p == 1)), np.sum((y == 0) & (p == 1)),
                     np.sum((y == 1) & (p == 0)), np.sum((y == 0) & (p == 0))], dtype=np.int64)


def f1_from_counts(values):
    values = np.asarray(values)
    numerator = 2.0 * values[..., 0]
    denom = numerator + values[..., 1] + values[..., 2]
    return np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom != 0)


def metric(truth, pred):
    c = counts(truth, pred)
    return dict(zip(("tp", "fp", "fn", "tn"), map(int, c), strict=True)) | {"rows": int(c.sum()), "f1": float(f1_from_counts(c))}


def comparison(frame, arm):
    old, new = frame[arm].to_numpy(), frame[f"{arm}_OR_e150"].to_numpy()
    before, after = metric(frame.label, old), metric(frame.label, new)
    return {"control": before, "candidate": after, "delta_f1": after["f1"] - before["f1"],
            "added_rows": int(((old == 0) & (new == 1)).sum()),
            "removed_rows": int(((old == 1) & (new == 0)).sum())}


def bootstrap(frame, arm, cfg):
    spec = cfg["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    all_counts = []
    clusters = {}
    for fold, fold_frame in frame.groupby("fold", sort=True):
        block_values = []
        for _, part in fold_frame.groupby("block", sort=True):
            block_values.append(np.stack([counts(part.label, part[arm]), counts(part.label, part[f"{arm}_OR_e150"])]))
        array = np.stack(block_values)
        clusters[fold] = len(array)
        if len(array) < spec["minimum_clusters_per_fold"]:
            return {"status": "NOT_ESTIMABLE_TOO_FEW_BLOCKS", "clusters": clusters}
        draw = rng.integers(0, len(array), size=(spec["resamples"], len(array)))
        all_counts.append(array[draw].sum(axis=1))
    total = np.stack(all_counts).sum(axis=0)
    deltas = f1_from_counts(total[:, 1]) - f1_from_counts(total[:, 0])
    return {"status": "DESCRIPTIVE_PAIRED_BLOCK_BOOTSTRAP", "clusters": clusters,
            "resamples": spec["resamples"], "seed": spec["seed"],
            "mean_delta_f1": float(deltas.mean()),
            "ci90": np.quantile(deltas, spec["ci_quantiles"]).tolist(),
            "fraction_delta_positive": float((deltas > 0).mean()),
            "fraction_delta_zero": float((deltas == 0).mean()),
            "not_fresh_test_or_posterior_probability": True}


def evaluate_frames(tree, proposal, permutation, cfg):
    frame = composition.canonical_keys(tree)
    for name in ("fold", "label", *ARMS):
        frame[name] = tree[name].to_numpy()
    p = composition.binary_column(proposal, "proposal")[permutation]
    frame["proposal"] = p
    for arm in ARMS:
        frame[arm] = composition.binary_column(frame, arm)
        frame[f"{arm}_OR_e150"] = np.maximum(frame[arm], p)
    composition.binary_column(frame, "label")
    anchor = pd.Timestamp(cfg["bootstrap"]["anchor_kst"])
    days = (frame.time.dt.tz_convert("Asia/Seoul").dt.normalize() - anchor).dt.days
    frame["block"] = days.floordiv(cfg["bootstrap"]["block_days"]).astype(int)
    scopes = {"primary_Q3_Q4": {arm: comparison(frame, arm) | {"bootstrap": bootstrap(frame, arm, cfg)} for arm in ARMS}}
    slices = []
    for by in (("fold",), ("station",), ("station", "layer"), ("fold", "block")):
        for key, part in frame.groupby(list(by), sort=True):
            values = key if isinstance(key, tuple) else (key,)
            item = {"by": list(by), "key": [v.item() if isinstance(v, np.generic) else v for v in values],
                    "rows": len(part), "arms": {arm: comparison(part, arm) for arm in ARMS}}
            slices.append(item)
            if by == ("fold",):
                scopes[str(values[0])] = {arm: item["arms"][arm] | {"bootstrap": bootstrap(part, arm, cfg)} for arm in ARMS}
    risks = {}
    for arm in ARMS:
        risks[arm] = {}
        for by in (("fold",), ("station",), ("station", "layer"), ("fold", "block")):
            group = [s for s in slices if s["by"] == list(by)]
            worst = min(group, key=lambda s: s["arms"][arm]["delta_f1"])
            risks[arm]["/".join(by)] = {"slice_count": len(group),
                "negative_slice_count": sum(s["arms"][arm]["delta_f1"] < 0 for s in group),
                "worst_delta_f1": worst["arms"][arm]["delta_f1"], "worst_key": worst["key"],
                "worst_rows": worst["rows"]}
    return frame, {"scopes": scopes, "slices": slices, "risk": risks}


def independent_arithmetic(frame, result):
    """Alternate sklearn arithmetic; no runner metric/count helper reuse."""
    from sklearn.metrics import confusion_matrix, f1_score
    checks = []
    targets = [("primary", frame, result["scopes"]["primary_Q3_Q4"])]
    for item in result["slices"]:
        mask = np.ones(len(frame), dtype=bool)
        for column, value in zip(item["by"], item["key"], strict=True):
            mask &= frame[column].eq(value).to_numpy()
        targets.append((str(item["key"]), frame.loc[mask], item["arms"]))
    for name, part, arms in targets:
        for arm, values in arms.items():
            for kind, column in (("control", arm), ("candidate", f"{arm}_OR_e150")):
                tn, fp, fn, tp = confusion_matrix(part.label, part[column], labels=[0, 1]).ravel()
                expected = {"rows": len(part), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
                if any(values[kind][k] != v for k, v in expected.items()):
                    raise ValueError("independent confusion arithmetic mismatch")
                if abs(values[kind]["f1"] - f1_score(part.label, part[column], zero_division=0)) > 1e-12:
                    raise ValueError("independent F1 arithmetic mismatch")
                checks.append(f"{name}/{arm}/{kind}")
            if values["removed_rows"] != 0:
                raise ValueError("OR removed a positive tree prediction")
            expected_or = (part[arm].eq(1) | part.proposal.eq(1)).astype(int)
            if not expected_or.eq(part[f"{arm}_OR_e150"]).all():
                raise ValueError("independent Boolean OR mismatch")
            delta = f1_score(part.label, expected_or, zero_division=0) - f1_score(part.label, part[arm], zero_division=0)
            if abs(values["delta_f1"] - delta) > 1e-12:
                raise ValueError("independent delta F1 mismatch")
            if values["added_rows"] != int((expected_or.eq(1) & part[arm].eq(0)).sum()):
                raise ValueError("independent added-row arithmetic mismatch")
    return {"status": "PASS", "independent_method": "sklearn confusion_matrix/f1_score", "checks": len(checks)}


def run():
    started = time.monotonic()
    cfg = checked_contract()
    inputs = cfg["inputs"]
    tree_dir, proposal_dir = ROOT / inputs["tree_directory"], ROOT / inputs["proposal_directory"]
    terminal_path = tree_dir / "terminal_result.json"
    if not terminal_path.exists():
        raise RuntimeError("TREE_NOT_TERMINAL: evaluation reads no OOF before completion")
    terminal = read_json(terminal_path)
    if terminal.get("status") != "COMPLETE" or terminal.get("completed_fits") != 24 or terminal.get("phase") != "historical":
        raise ValueError("exact COMPLETE 24-fit historical tree terminal required")
    if terminal["seal_sha256"] != sha(ROOT / inputs["tree_seal"]):
        raise ValueError("tree seal changed")
    if sha(HERE / "tree.py") != inputs["tree_runner_sha256"]:
        raise ValueError("tree runner provenance changed")
    if sha(tree_dir / "oof.parquet") != terminal["files"]["oof.parquet"] or sha(tree_dir / "oof.parquet") != terminal["oof_sha256"]:
        raise ValueError("tree OOF hash mismatch")
    if sha(proposal_dir / "qa.json") != inputs["proposal_qa_sha256"] or sha(proposal_dir / "proposals.parquet") != inputs["proposal_sha256"]:
        raise ValueError("frozen proposal provenance mismatch")
    qa = read_json(proposal_dir / "qa.json")
    if qa["status"] != "PASS" or not all(qa["checks"].values()) or qa["new_fits"] != 0 or qa["reused_historical_fits"] != 6:
        raise ValueError("historical proposal semantic/lineage QA required")
    tree_keys = pd.read_parquet(tree_dir / "oof.parquet", columns=KEYS + ["fold"])
    tree_keys = tree_keys.loc[tree_keys.fold.isin([f["id"] for f in cfg["folds"]])].reset_index(drop=True)
    proposal_keys = pd.read_parquet(proposal_dir / "proposals.parquet", columns=KEYS)
    coverage, permutation = key_coverage(tree_keys, proposal_keys, cfg)
    temporal = chronology(proposal_keys, qa, cfg)
    output = ROOT / "artifacts" / cfg["id"]
    output.mkdir(parents=True, exist_ok=False)
    base = {"id": cfg["id"], "role": cfg["role"], "new_fits": 0, "tree_historical_fits": 24,
            "mstcn_reused_historical_fits": 6, "tree_terminal_sha256": sha(terminal_path),
            "union_seal_sha256": sha(SEAL), "coverage": coverage, "chronology": temporal,
            "official_rows": 0, "hidden_rows": 0, "csv_outputs": 0, "upload": 0,
            "tree_feature_caveat": cfg["tree_feature_caveat"], "input_pins": inputs}
    if permutation is None:
        base.update(status="BLOCKED_KEY_MISMATCH", performance_rows_read=0,
                    runtime_seconds=time.monotonic() - started)
        save_json(output / "result.json", base)
        print(json.dumps({"status": base["status"], "coverage": coverage}))
        return
    tree = pd.read_parquet(tree_dir / "oof.parquet", columns=KEYS + ["fold", "label", *ARMS])
    tree = tree.loc[tree.fold.isin([f["id"] for f in cfg["folds"]])].reset_index(drop=True)
    proposal = pd.read_parquet(proposal_dir / "proposals.parquet", columns=KEYS + ["proposal"])
    if composition.ordered_key_sha256(tree) != composition.ordered_key_sha256(tree_keys):
        raise ValueError("tree keys changed between projected reads")
    frame, result = evaluate_frames(tree, proposal, permutation, cfg)
    arithmetic = independent_arithmetic(frame, result)
    frame.to_parquet(output / "paired-evaluation.parquet", index=False)
    base.update(result, status="COMPLETE_RETROSPECTIVE_EVALUATION", rows=len(frame),
                ordered_keys_sha256=composition.ordered_key_sha256(frame),
                paired_artifact_sha256=sha(output / "paired-evaluation.parquet"),
                runtime_seconds=time.monotonic() - started)
    checked_contract()
    if sha(terminal_path) != base["tree_terminal_sha256"] or sha(tree_dir / "oof.parquet") != terminal["oof_sha256"] or sha(proposal_dir / "proposals.parquet") != inputs["proposal_sha256"]:
        raise ValueError("evaluation inputs changed while running")
    save_json(output / "result.json", base)
    save_json(output / "independent-qa.json", arithmetic | {"result_sha256": sha(output / "result.json"),
        "coverage_exact": True, "chronology_pass": True, "new_fits": 0, "official_rows": 0})
    print(json.dumps({"status": base["status"], "rows": len(frame), "arithmetic_checks": arithmetic["checks"],
                      "output": output.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    seal() if args.seal else run()
