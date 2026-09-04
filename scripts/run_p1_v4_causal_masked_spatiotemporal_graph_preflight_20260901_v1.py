"""Zero-fit novelty, GPU, and horizontal-graph data-contract gate for P1 v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v4_causal_masked_spatiotemporal_graph_preflight_20260901_v1"
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
        raise FileExistsError("v4 namespace consumed")
    config = _read(CONFIG_PATH)
    readme = Path(config["data_contract"]["source_readme"]["path"]).resolve(strict=True)
    if _sha(readme) != config["data_contract"]["source_readme"]["sha256"]:
        raise RuntimeError("source README changed")
    resource = config["resource_contract"]
    observed_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if observed_devices < resource["required_cuda_devices_gte"]:
        raise RuntimeError("CUDA resource contract failed")
    properties = torch.cuda.get_device_properties(0)
    observed_memory = properties.total_memory // (1024 * 1024)
    if observed_memory < resource["required_gpu_memory_mib_gte"]:
        raise RuntimeError("GPU memory contract failed")
    if not config["semantic_audit"]["novel"] or config["semantic_audit"]["semantic_duplicate"]:
        raise RuntimeError("semantic novelty decision changed")
    if config["data_contract"]["status"] != "FAIL_MISSING_PHYSICAL_HORIZONTAL_ADJACENCY_AUTHORITY":
        raise RuntimeError("data-contract blocker changed")
    return {"schema_version": "p1.v4_graph.preflight.v1", "experiment_id": EXPERIMENT_ID, "status": "PASS_ZERO_OPERATION_AUDIT_WITH_DATA_CONTRACT_BLOCK", "config_sha256": _sha(CONFIG_PATH), "runner_sha256": _sha(Path(__file__)), "readme_sha256": _sha(readme), "gpu": {"name": properties.name, "memory_mib": observed_memory, "torch": torch.__version__, "devices": observed_devices, "resource_pass": True}, "semantic_novelty": config["semantic_audit"], "data_contract": config["data_contract"], "decision": config["decision"], "counters": {"claims": 0, "fits": 0, "targets": 0, "official": 0, "csv": 0, "uploads": 0}}


def qa() -> dict[str, Any]:
    ready, config = preflight(), _read(CONFIG_PATH)
    checks = {"zero_operation": all(value == 0 for value in ready["counters"].values()), "gpu_pass": ready["gpu"]["resource_pass"], "novel": ready["semantic_novelty"]["novel"], "no_horizontal_authority": not config["data_contract"]["signed_horizontal_edge_manifest_present"], "zero_fit_decision": config["decision"] == "CLOSE_ZERO_FIT_MISSING_HORIZONTAL_GRAPH_CONTRACT", "simpler_candidate_audit_only": config["simpler_unused_candidate"]["status"] == "AUDIT_ONLY_REQUIRES_NEW_PREREGISTRATION"}
    return {"experiment_id": EXPERIMENT_ID, "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def execute() -> dict[str, Any]:
    ready, config = preflight(), _read(CONFIG_PATH)
    _write(LOCK_PATH, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_EXACTLY_ONCE_ZERO_FIT", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT_DIR.mkdir(exist_ok=False)
    result = {"schema_version": "p1.v4_graph.result.v1", "experiment_id": EXPERIMENT_ID, "decision": config["decision"], "resource_audit": ready["gpu"], "semantic_audit": config["semantic_audit"], "data_contract_failure": config["data_contract"], "simpler_unused_candidate": config["simpler_unused_candidate"], "counters": {"executions": 1, "fits": 0, "targets": 0, "official": 0, "csv": 0, "uploads": 0}, "hashes": {"config": ready["config_sha256"], "runner": ready["runner_sha256"], "lock": _sha(LOCK_PATH)}}
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
