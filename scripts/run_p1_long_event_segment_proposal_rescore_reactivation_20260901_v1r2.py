"""Pinned three-alias compatibility repair for the frozen P1 screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPAIR_RUNNER = ROOT / "scripts/run_p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r1.py"
CONFIG_PATH = ROOT / "configs/experiments/p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r2.json"
ARTIFACT_DIR = ROOT / "artifacts/p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r2"
EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r2"
REQUIRED_NUMERICAL_ATTRIBUTES = {
    "TabularEncoder",
    "apply_postprocess",
    "detect_plateaus",
    "detect_singleton_spikes",
}
BRIDGE_BINDINGS = {
    "apply_postprocess": ("p1_qc.pipeline", "apply_postprocess", "src/p1_qc/pipeline.py", "389a905abbaf4b62e7d862c44fa25bba2e58dae7b7a7f5bcb4e1e8438d914669"),
    "detect_plateaus": ("p1_qc.rules", "detect_plateaus", "src/p1_qc/rules.py", "ec921139f210f3b264c519346547fa0e17b094f54ce83780f21cf48f5287069c"),
    "detect_singleton_spikes": ("p1_qc.rules", "detect_singleton_spikes", "src/p1_qc/rules.py", "ec921139f210f3b264c519346547fa0e17b094f54ce83780f21cf48f5287069c"),
}


def _load_repair() -> ModuleType:
    specification = importlib.util.spec_from_file_location("p1_segment_reactivation_v1r1_base", REPAIR_RUNNER)
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load v1r1 state-reuse repair")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.CONFIG_PATH = CONFIG_PATH
    module.ARTIFACT_DIR = ARTIFACT_DIR
    module.base.EXPERIMENT_ID = EXPERIMENT_ID
    module.base.CONFIG_PATH = CONFIG_PATH
    module.base.ARTIFACT_DIR = ARTIFACT_DIR
    module.base.__file__ = str(Path(__file__).resolve())
    return module


repair = _load_repair()
_one_pass_prepare = repair.prepare


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bridge(snapshot: Path, captured: dict[str, Any]) -> dict[str, Any]:
    numerical = captured["numerical"]
    execution_source = Path(captured["execution"].__file__).read_text(encoding="utf-8")
    enumerated = set(re.findall(r"\bnumerical\.([A-Za-z_][A-Za-z0-9_]*)", execution_source))
    if enumerated != REQUIRED_NUMERICAL_ATTRIBUTES:
        raise RuntimeError(f"frozen execution required-attribute set changed: {sorted(enumerated)}")
    preexisting = {name for name in enumerated if hasattr(numerical, name)}
    if preexisting != {"TabularEncoder"}:
        raise RuntimeError(f"pre-bridge numerical namespace changed: {sorted(preexisting)}")
    if set(BRIDGE_BINDINGS) != enumerated - preexisting:
        raise RuntimeError("bridge is not the exact missing-attribute set")

    receipts: dict[str, Any] = {}
    for alias, (module_name, callable_name, relative, expected_sha) in BRIDGE_BINDINGS.items():
        module = sys.modules.get(module_name)
        if module is None:
            raise RuntimeError(f"authenticated snapshot module not loaded: {module_name}")
        module_path = Path(module.__file__).resolve(strict=True)
        expected_path = (snapshot / relative).resolve(strict=True)
        if module_path != expected_path or _sha256(module_path) != expected_sha:
            raise RuntimeError(f"bridge module binding changed: {module_name}")
        function = getattr(module, callable_name)
        if not callable(function):
            raise RuntimeError(f"bridge target is not callable: {module_name}.{callable_name}")
        setattr(numerical, alias, function)
        if getattr(numerical, alias) is not function:
            raise RuntimeError(f"bridge callable identity failed: {alias}")
        source_sha = hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()
        receipts[alias] = {
            "qualified_name": f"{module_name}.{callable_name}",
            "module_relative_path": relative,
            "module_sha256": expected_sha,
            "callable_source_sha256": source_sha,
            "callable_identity_exact": True,
            "wrapper_used": False,
        }
    postexisting = {name for name in enumerated if hasattr(numerical, name)}
    if postexisting != enumerated:
        raise RuntimeError("post-bridge numerical namespace is incomplete")
    return {
        "required_attributes": sorted(enumerated),
        "preexisting_attributes": sorted(preexisting),
        "exact_added_attributes": sorted(BRIDGE_BINDINGS),
        "other_aliases_added": 0,
        "callable_receipts": receipts,
    }


def prepare(*, retain_snapshot: bool):
    public, v6, snapshot, result, captured = _one_pass_prepare(retain_snapshot=True)
    if snapshot is None:
        raise RuntimeError("v1r2 bridge requires a retained snapshot during readiness")
    try:
        bridge = _bridge(snapshot, captured)
        public["pinned_compatibility_bridge"] = bridge
        public["verification_sha256"] = repair.base._canonical_sha(
            {key: value for key, value in public.items() if key != "verification_sha256"}
        )
    except BaseException:
        v6._remove_snapshot_modules(snapshot)
        v6._cleanup_snapshot(snapshot)
        raise
    if not retain_snapshot:
        v6._remove_snapshot_modules(snapshot)
        v6._cleanup_snapshot(snapshot)
        snapshot = None
    return public, v6, snapshot, result, captured


def _retained_prepare(*, retain_snapshot: bool):
    return prepare(retain_snapshot=retain_snapshot)


repair.prepare = _retained_prepare


def qa() -> dict[str, Any]:
    public, *_ = prepare(retain_snapshot=False)
    bridge = public["pinned_compatibility_bridge"]
    checks = {
        "zero_operation": all(value == 0 for value in public["operation_counters"].values()),
        "fresh_namespace": public["namespace"]["fresh_artifact_namespace"] is True,
        "exact_required_set": set(bridge["required_attributes"]) == REQUIRED_NUMERICAL_ATTRIBUTES,
        "exact_three_alias_bridge": set(bridge["exact_added_attributes"]) == set(BRIDGE_BINDINGS),
        "callable_identity_exact": all(
            receipt["callable_identity_exact"] and not receipt["wrapper_used"]
            for receipt in bridge["callable_receipts"].values()
        ),
        "worker_state_reload_absent": public["same_process_state_reuse"]["load_worker_state_calls"] == 0,
    }
    return {
        "schema_version": "p1_long_event_segment_proposal_rescore.reactivation_independent_qa.v1r2",
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "preflight_verification_sha256": public["verification_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--qa", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        public, *_ = prepare(retain_snapshot=False)
        print(repair.base._canonical_bytes({"output": public}).decode("utf-8"), end="")
    elif args.qa:
        print(repair.base._canonical_bytes(qa()).decode("utf-8"), end="")
    else:
        path, result = repair.execute()
        print(
            repair.base._canonical_bytes(
                {"status": "complete", "result_path": str(path), "decision": result["decision"], "f1_delta": result["metrics"]["pooled"]["f1_delta"]}
            ).decode("utf-8"),
            end="",
        )


if __name__ == "__main__":
    main()
