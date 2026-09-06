"""Zero-fit exact-Q3/Q4 fixed MS proposal bridge, after CPU2 tuning QA only."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("original_union_v2", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/evaluate_union_v2.py")
u = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(u)
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

e = u.engine
ID = "p1_core_tuning_ms_bridge_20260906_v1"
TREE = ROOT / "artifacts/p1_core_tuning_20260906_v1"
REPORT = ROOT / "reports" / ID
SEAL = REPORT / "seal.json"
ARMS = ("control", "candidate", "control_plus_ms", "candidate_plus_ms")


def exact_alignment(tree_keys, proposal_keys, cfg):
    left = e.composition.canonical_keys(tree_keys)
    left["fold"] = tree_keys.fold.to_numpy()
    right = u.original_fold_ownership(proposal_keys, cfg)
    li = pd.MultiIndex.from_frame(left[e.KEYS + ["fold"]])
    ri = pd.MultiIndex.from_frame(right[e.KEYS + ["fold"]])
    if len(li) != len(ri) or not li.isin(ri).all() or not ri.isin(li).all():
        raise ValueError("exact full key/fold parity required; intersection forbidden")
    for fold in cfg["folds"]:
        if left.fold.eq(fold["id"]).sum() != fold["source_rows"]:
            raise ValueError("fixed Q3/Q4 population mismatch")
    permutation = ri.get_indexer(li)
    if (permutation < 0).any() or len(np.unique(permutation)) != len(permutation):
        raise ValueError("non-bijective proposal alignment")
    return left, right, permutation


def bootstrap(frame, cfg):
    spec = cfg["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    by_fold, clusters = [], {}
    for fold, part in frame.groupby("fold", sort=True):
        entries = [np.stack([e.counts(block.label, block.control_plus_ms), e.counts(block.label, block.candidate_plus_ms)])
                   for _, block in part.groupby("block", sort=True)]
        if len(entries) < spec["minimum_clusters_per_fold"]:
            raise ValueError("too few predeclared blocks")
        values = np.stack(entries)
        clusters[fold] = len(values)
        indexes = rng.integers(0, len(values), (spec["resamples"], len(values)))
        by_fold.append(values[indexes].sum(axis=1))
    totals = np.stack(by_fold).sum(axis=0)
    deltas = e.f1_from_counts(totals[:, 1]) - e.f1_from_counts(totals[:, 0])
    return {"ci90": np.quantile(deltas, spec["ci_quantiles"]).tolist(),
            "p_improve": float((deltas > 0).mean()), "clusters": clusters,
            "resamples": spec["resamples"], "seed": spec["seed"],
            "method": "fold-stratified KST seven-day paired pooled-F1 bootstrap; descriptive only"}


def sources():
    paths = {str(Path(__file__).relative_to(ROOT).as_posix()): e.sha(__file__),
             "tests/test_p1_core_tuning_ms_bridge_20260906_v1.py": e.sha(ROOT / "tests/test_p1_core_tuning_ms_bridge_20260906_v1.py"),
             "reports/p1_core_tuning_20260906_v1/seal.json": e.sha(ROOT / "reports/p1_core_tuning_20260906_v1/seal.json"),
             "reports/p1_champion_reconstruction_20260906_v1/union-seal-v2.json": e.sha(u.SEAL)}
    old = e.read_json(u.SEAL)
    for path, digest in old["source_pins"].items():
        if e.sha(ROOT / path) != digest:
            raise ValueError("original union contract/source changed")
        paths[path] = digest
    return paths


def seal():
    REPORT.mkdir(parents=True, exist_ok=True)
    cfg = e.read_json(u.CONTRACT)
    e.save_json(SEAL, {"id": ID, "status": "SEALED_BEFORE_BRIDGE_METRICS", "source_pins": sources(),
        "rows": 287862, "fit_budget": 0, "cpu_threads": 2, "gpu": False,
        "scope": "separate retrospective diagnostic; does not replace new tree overall primary",
        "comparison": "same frozen MS proposal OR new CPU2 control vs OR new CPU2 inner-selected candidate",
        "proposal": cfg["proposal"], "folds": cfg["folds"], "bootstrap": cfg["bootstrap"],
        "selection": "none, no recipe/threshold/rule/row changes", "old_CPU4_OOF_read": False,
        "official_rows": 0, "csv_outputs": 0, "upload": 0})


def run():
    started = time.monotonic()
    frozen = e.read_json(SEAL)
    if frozen["source_pins"] != sources():
        raise ValueError("bridge source seal changed")
    terminal = e.read_json(TREE / "terminal_result.json")
    qa = e.read_json(TREE / "independent-qa.json")
    if terminal["status"] != "COMPLETE" or qa["status"] != "PASS" or qa["terminal_result_sha256"] != e.sha(TREE / "terminal_result.json"):
        raise ValueError("completed new tuning and linked independent QA required")
    if terminal["seal_sha256"] != e.sha(ROOT / "reports/p1_core_tuning_20260906_v1/seal.json") or terminal["oof_sha256"] != e.sha(TREE / "oof.parquet"):
        raise ValueError("tuning OOF/seal lineage mismatch")
    cfg = e.read_json(u.CONTRACT)
    proposal_dir = ROOT / cfg["inputs"]["proposal_directory"]
    proposal_path, qa_path = proposal_dir / "proposals.parquet", proposal_dir / "qa.json"
    if e.sha(proposal_path) != cfg["inputs"]["proposal_sha256"] or e.sha(qa_path) != cfg["inputs"]["proposal_qa_sha256"]:
        raise ValueError("fixed MS proposal source changed")
    proposal_qa = e.read_json(qa_path)
    if proposal_qa["status"] != "PASS" or not all(proposal_qa["checks"].values()) or proposal_qa["new_fits"] != 0 or proposal_qa["reused_historical_fits"] != 6:
        raise ValueError("MS semantic QA required")
    tree_keys = pd.read_parquet(TREE / "oof.parquet", columns=e.KEYS + ["fold"])
    tree_keys = tree_keys.loc[tree_keys.fold.isin([f["id"] for f in cfg["folds"]])].reset_index(drop=True)
    proposal_keys = pd.read_parquet(proposal_path, columns=e.KEYS)
    left, owned, permutation = exact_alignment(tree_keys, proposal_keys, cfg)
    chronology = []
    for fold in cfg["folds"]:
        meta = proposal_qa["folds"][fold["phase"]]
        maximum = pd.Timestamp(meta["training_max_time_utc"])
        minimum = owned.loc[owned.fold.eq(fold["id"]), "time"].min()
        gap = (minimum - maximum).total_seconds() / 3600
        if maximum != pd.Timestamp(fold["training_max_utc"]) or meta["rows"] != fold["source_rows"] or gap < cfg["registered_minimum_holdout_gap_hours"] or gap < cfg["required_feature_dependency_hours"]:
            raise ValueError("original MS chronology mismatch")
        chronology.append({"fold": fold["id"], "rows": fold["source_rows"], "gap_hours": gap})
    # Only after full key coverage has passed, read training-derived labels/bits.
    rows = pd.read_parquet(TREE / "oof.parquet")
    rows = rows.loc[rows.fold.isin([f["id"] for f in cfg["folds"]])].reset_index(drop=True)
    proposal = pd.read_parquet(proposal_path, columns=e.KEYS + ["proposal"])
    p = e.composition.binary_column(proposal, "proposal")[permutation]
    frame = left.copy()
    for name in ("label", "control", "candidate"):
        frame[name] = e.composition.binary_column(rows, name)
    frame["proposal"] = p
    frame["control_plus_ms"] = frame.control.to_numpy() | p
    frame["candidate_plus_ms"] = frame.candidate.to_numpy() | p
    anchor = pd.Timestamp(cfg["bootstrap"]["anchor_kst"])
    frame["block"] = ((frame.time.dt.tz_convert("Asia/Seoul").dt.normalize() - anchor).dt.days // cfg["bootstrap"]["block_days"]).astype(int)
    checks = []
    def metrics(part):
        result = {arm: e.metric(part.label, part[arm]) for arm in ARMS}
        for arm in ARMS:
            tn, fp, fn, tp = confusion_matrix(part.label, part[arm], labels=[0, 1]).ravel()
            reference = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "rows": len(part)}
            if any(result[arm][k] != v for k, v in reference.items()) or abs(result[arm]["f1"] - f1_score(part.label, part[arm], zero_division=0)) > 1e-12:
                raise ValueError("independent bridge arithmetic mismatch")
            checks.append("independent_confusion_and_F1")
        return {"rows": len(part), "positive": int(part.label.sum()), "metrics": result,
                "tree_delta_f1": result["candidate"]["f1"] - result["control"]["f1"],
                "same_MS_delta_f1": result["candidate_plus_ms"]["f1"] - result["control_plus_ms"]["f1"]}
    primary = metrics(frame)
    primary["paired_seven_day_bootstrap"] = bootstrap(frame, cfg)
    # Independent sampling implementation: scalar per-replicate block aggregation.
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    samples = []
    for _, fold in frame.groupby("fold", sort=True):
        counts = np.stack([np.stack([e.counts(part.label, part.control_plus_ms), e.counts(part.label, part.candidate_plus_ms)]) for _, part in fold.groupby("block", sort=True)])
        samples.append(np.stack([counts[rng.integers(0, len(counts), len(counts))].sum(axis=0) for _ in range(2000)]))
    totals = np.stack(samples).sum(axis=0)
    d = e.f1_from_counts(totals[:, 1]) - e.f1_from_counts(totals[:, 0])
    if not np.array_equal(np.quantile(d, [.05, .95]), primary["paired_seven_day_bootstrap"]["ci90"]) or float((d > 0).mean()) != primary["paired_seven_day_bootstrap"]["p_improve"]:
        raise ValueError("independent paired bootstrap mismatch")
    checks.append("independent_stratified_bootstrap_exact")
    slices = []
    for fields in (["fold"], ["station", "layer"], ["fold", "block"]):
        for key, part in frame.groupby(fields, sort=True):
            slices.append({"by": fields, "key": str(key), **metrics(part)})
    output = ROOT / "artifacts" / ID
    output.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output / "paired.parquet", index=False)
    pins = {"tree_terminal": e.sha(TREE / "terminal_result.json"), "tree_QA": e.sha(TREE / "independent-qa.json"),
            "tree_OOF": e.sha(TREE / "oof.parquet"), "fixed_MS_proposal": e.sha(proposal_path),
            "fixed_MS_QA": e.sha(qa_path), "bridge_seal": e.sha(SEAL)}
    if pins["tree_OOF"] != terminal["oof_sha256"] or pins["fixed_MS_proposal"] != cfg["inputs"]["proposal_sha256"] or frozen["source_pins"] != sources():
        raise ValueError("bridge source changed while evaluating")
    result = {"id": ID, "status": "COMPLETE_RETROSPECTIVE_DIAGNOSTIC", "rows": len(frame),
        "coverage": {"full_keys_and_original_fold_owner": True, "missing_rows": 0, "extra_rows": 0, "intersection_rows_dropped": 0},
        "primary_Q3_Q4": primary, "slices": slices, "chronology": chronology,
        "worst_station_layer": min((s for s in slices if s["by"] == ["station", "layer"]), key=lambda s: s["same_MS_delta_f1"]),
        "worst_block": min((s for s in slices if s["by"] == ["fold", "block"]), key=lambda s: s["same_MS_delta_f1"]),
        "input_sha256": pins, "paired_sha256": e.sha(output / "paired.parquet"), "seal_sha256": e.sha(SEAL),
        "runtime_seconds": time.monotonic() - started, "fits": 0, "official_rows": 0, "hidden_rows": 0, "csv_outputs": 0, "upload": 0,
        "caveats": ["new tree overall primary unchanged; this is a separate fixed-MS bridge", "same CPU2 comparator, no old CPU4 OOF/model reuse", "original proposal and all 287862 keys including run-boundary ownership retained", "no policy/threshold tuning or full/official materialization", "not fresh confirmation or official score prediction"]}
    e.save_json(output / "result.json", result)
    e.save_json(output / "independent-qa.json", {"status": "PASS", "pid": os.getpid(), "check_count": len(checks),
        "checks": checks, "result_sha256": e.sha(output / "result.json"), "exact_coverage": True,
        "fits": 0, "official_rows": 0, "qa": "sklearn confusion/F1 plus independent scalar paired resampling"})
    print({"status": result["status"], "rows": len(frame), "checks": len(checks), "fits": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        seal() if args.seal else run()
