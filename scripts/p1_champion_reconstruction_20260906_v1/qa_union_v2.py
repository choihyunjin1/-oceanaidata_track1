"""Independent QA of the original-fold-preserving technical amendment."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    q = module("union_independent_v1_immutable", "qa_union.py")
    a = module("union_v2_ownership", "evaluate_union_v2.py")
    engine = a.configure_engine()
    cfg = engine.checked_contract()
    output = q.ROOT / "artifacts" / cfg["id"]
    target = output / "independent-bootstrap-qa.json"
    if target.exists():
        raise FileExistsError("preserve existing QA")
    result = q.read(output / "result.json")
    if result["status"] != "COMPLETE_RETROSPECTIVE_EVALUATION":
        raise ValueError("complete exact-key result required; no intersection scoring")
    if result["union_seal_sha256"] != q.sha(a.SEAL) or result["paired_artifact_sha256"] != q.sha(output / "paired-evaluation.parquet"):
        raise ValueError("result/seal/paired artifact mismatch")
    frame = q.pd.read_parquet(output / "paired-evaluation.parquet")
    if len(frame) != 287862 or result["rows"] != len(frame) or result["ordered_keys_sha256"] != q.composition.ordered_key_sha256(frame):
        raise ValueError("complete original evaluation population required")
    keys = list(q.composition.KEYS)
    inputs = cfg["inputs"]
    original = q.pd.read_parquet(q.ROOT / inputs["proposal_directory"] / "proposals.parquet", columns=keys)
    original_owned = a.original_fold_ownership(original, cfg)
    left = q.composition.canonical_keys(frame)
    left["fold"] = frame.fold.to_numpy()
    li = q.pd.MultiIndex.from_frame(left[keys + ["fold"]])
    ri = q.pd.MultiIndex.from_frame(original_owned[keys + ["fold"]])
    if len(li) != len(ri) or not li.isin(ri).all():
        raise ValueError("independent original fold/key coverage mismatch")
    if any(result[k] != 0 for k in ("new_fits", "official_rows", "hidden_rows", "csv_outputs", "upload")):
        raise ValueError("scope counters violated")
    checks = q.verify_statistics(frame, result, cfg)
    after_boundary = int((original_owned.fold.eq("2025_q3") & original_owned.time.ge(q.pd.Timestamp("2025-10-01T00:00:00+09:00"))).sum())
    if after_boundary != 119:
        raise ValueError("original 119-row cross-boundary ownership changed")
    payload = {"status": "PASS", "pid": os.getpid(), "checks": len(checks) + 3,
        "check_names": checks + ["original_full_key_and_fold_coverage", "original_fold_SHA_exact", "119_run_owned_October_rows_retained_in_Q3"],
        "method": "independent sklearn counts and replicate-loop stratified paired bootstrap; exact original key-fold lineage",
        "result_sha256": q.sha(output / "result.json"), "paired_sha256": result["paired_artifact_sha256"],
        "qa_source_sha256": q.sha(__file__), "qa_engine_sha256": q.sha(HERE / "qa_union.py"),
        "new_fits": 0, "official_rows": 0, "hidden_rows": 0, "rtol": 0, "atol": 1e-12,
        "confidence": "share_with_retrospective_and_block_dependence_caveats"}
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "checks": payload["checks"], "fits": 0}))


if __name__ == "__main__":
    main()
