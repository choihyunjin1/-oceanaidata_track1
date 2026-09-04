"""Zero-fit semantic gate and unused graph-architecture audit for P1 v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v3_subspace_supervised_semantic_gate_20260901_v1"
CONFIG_PATH = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK_PATH = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON object required")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode() + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def preflight() -> dict[str, Any]:
    if ARTIFACT_DIR.exists() or LOCK_PATH.exists():
        raise FileExistsError("semantic-gate namespace consumed")
    config = _read(CONFIG_PATH)
    if config["experiment_id"] != EXPERIMENT_ID or config["decision"] != "CLOSE_ZERO_FIT_SEMANTIC_DUPLICATE":
        raise RuntimeError("semantic decision changed")
    hashes = {}
    for group in (config["semantic_predecessor"], config["registry"]):
        for role, item in group.items():
            if not isinstance(item, dict) or "path" not in item:
                continue
            path = ROOT / item["path"]
            actual = _sha(path)
            if actual != item["sha256"]:
                raise RuntimeError(f"authority changed: {role}")
            hashes[role] = actual
    if config["zero_fit_contract"] != {"model_fits": 0, "target_rows": 0, "official_test_sample_submission_hidden_reads": 0, "csv": 0, "uploads": 0, "retry_or_retune": 0}:
        raise RuntimeError("zero-fit contract changed")
    return {"schema_version": "p1.v3_semantic_gate.preflight.v1", "experiment_id": EXPERIMENT_ID, "status": "PASS_ZERO_OPERATION_SEMANTIC_AUDIT", "config_sha256": _sha(CONFIG_PATH), "runner_sha256": _sha(Path(__file__)), "authority_hashes": hashes, "decision": config["decision"], "counters": {"claims": 0, "fits": 0, "targets": 0, "official": 0, "csv": 0, "uploads": 0}}


def qa() -> dict[str, Any]:
    ready, config = preflight(), _read(CONFIG_PATH)
    alternative = config["unused_graph_spatiotemporal_audit"]
    checks = {"zero_operation": all(value == 0 for value in ready["counters"].values()), "semantic_predecessor_bound": len(config["semantic_predecessor"]["same_roles"]) == 6, "predecessor_outer_reversal_preserved": config["semantic_predecessor"]["observed_q3_delta_f1"] > 0 > config["semantic_predecessor"]["observed_q4_delta_f1"], "alternative_audit_only": alternative["status"] == "AUDIT_ONLY_NOVEL_REPRESENTATION_REQUIRES_PREREGISTRATION", "graph_exact_hits_zero": alternative["exact_repository_name_hits"] == 0}
    return {"experiment_id": EXPERIMENT_ID, "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def execute() -> dict[str, Any]:
    ready, config = preflight(), _read(CONFIG_PATH)
    _write(LOCK_PATH, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_EXACTLY_ONCE_ZERO_FIT", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT_DIR.mkdir(exist_ok=False)
    result = {"schema_version": "p1.v3_semantic_gate.result.v1", "experiment_id": EXPERIMENT_ID, "decision": config["decision"], "reason": "The proposed supervised head repeats the prior anchor-negative regularized logistic partial-pooling and precision-LCB router; changing only the representation does not create a new nested architecture on the same proxy.", "predecessor_evidence": config["semantic_predecessor"], "unused_graph_spatiotemporal_audit": config["unused_graph_spatiotemporal_audit"], "counters": {"executions": 1, "fits": 0, "targets": 0, "official": 0, "csv": 0, "uploads": 0}, "hashes": {"config": ready["config_sha256"], "runner": ready["runner_sha256"], "lock": _sha(LOCK_PATH)}}
    _write(ARTIFACT_DIR / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = preflight() if args.preflight else qa() if args.qa else execute()
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False), end="")


if __name__ == "__main__":
    main()
