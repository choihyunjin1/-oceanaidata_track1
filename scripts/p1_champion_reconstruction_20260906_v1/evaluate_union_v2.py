"""New zero-fit technical amendment; preserve original whole-run fold ownership.

The v1 evaluator/contract/failure are immutable. Reuse its numerical functions,
not its incorrect calendar fold assignment. No prediction or metric change.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("p1_union_v1_frozen_engine", HERE / "evaluate_union.py")
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)
ROOT = engine.ROOT
REPORT = engine.REPORT
CONTRACT = REPORT / "union-contract-v2.json"
SEAL = REPORT / "union-seal-v2.json"


def original_ordered_key_sha(frame):
    digest = engine.hashlib.sha256()
    for column in engine.KEYS:
        digest.update(column.encode("ascii") + b"\0")
        for value in frame[column].tolist():
            raw = str(value).encode("utf-8")
            digest.update(len(raw).to_bytes(4, "little"))
            digest.update(raw)
    return digest.hexdigest()


def original_fold_ownership(frame, cfg):
    """Extractor loops q3 then q4; prove each complete ordered key slice."""
    if len(frame) != sum(f["source_rows"] for f in cfg["folds"]):
        raise ValueError("original proposal whole-fold population mismatch")
    out = engine.composition.canonical_keys(frame)
    out["fold"] = ""
    offset = 0
    for fold in cfg["folds"]:
        end = offset + fold["source_rows"]
        source_part = frame.iloc[offset:end]
        if original_ordered_key_sha(source_part) != fold["ordered_source_key_sha256"]:
            raise ValueError("original fold ordered key SHA mismatch; no calendar fallback")
        out.loc[offset:end - 1, "fold"] = fold["id"]
        offset = end
    return out


def source_pins():
    files = [Path(__file__), HERE / "evaluate_union.py", HERE / "qa_union.py", HERE / "qa_union_v2.py",
             HERE / "composition.py", CONTRACT,
             ROOT / "tests/test_p1_champion_union_v2_20260906_v1.py",
             REPORT / "union-contract.json", REPORT / "union-seal.json",
             REPORT / "union-evaluation-v1-technical-failure.json"]
    return {path.relative_to(ROOT).as_posix(): engine.sha(path) for path in files}


def configure_engine():
    engine.CONTRACT = CONTRACT
    engine.SEAL = SEAL
    engine.source_pins = source_pins
    engine.attach_proposal_folds = original_fold_ownership
    return engine


def seal():
    # Validate original immutable v1 lineage before the approved amendment.
    old = engine.read_json(REPORT / "union-seal.json")
    if any(engine.sha(ROOT / path) != value for path, value in old["source_pins"].items()):
        raise ValueError("original v1 source/contract/test no longer preserved")
    engine.save_json(SEAL, {"status": "SEALED_TECHNICAL_CORRECTION_BEFORE_ANY_UNION_METRICS",
        "created_unix": time.time(), "source_pins": source_pins(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "scikit-learn")},
        "new_fits": 0, "predictions_or_thresholds_changed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    else:
        configure_engine().run()
