"""Fresh science-neutral provenance recovery for the frozen P1 v36 GCE cycle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1r1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SCIENCE_MODULE = ROOT / "scripts/run_p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1.py"
SHARED_ENGINE = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v36r1_science", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen science module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


science = _module(SCIENCE_MODULE)
shared = science.shared
base = shared.base
GCEClassifier = science.GCEClassifier
CAUSAL_FEATURES = science.CAUSAL_FEATURES
OBJECTIVE_GUARDS = science._objective_guards
_BASE_WRITE = base._write
_SHARED_PREFLIGHT = shared.preflight


def _write_with_provenance(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        payload = dict(value)
        hashes = dict(payload["hashes"])
        hashes.update(
            {
                "wrapper": base._sha(Path(__file__)),
                "science_module": base._sha(SCIENCE_MODULE),
                "shared_engine": base._sha(SHARED_ENGINE),
            }
        )
        payload["hashes"] = hashes
        payload["provenance_recovery"] = {
            "invalid_parent_artifact_reads": 0,
            "invalid_parent_scientific_metrics_used": 0,
            "wrapper_identity_preserved": True,
        }
    _BASE_WRITE(path, payload)


def _configure() -> None:
    shared.EXPERIMENT_ID = EXPERIMENT_ID
    shared.CONFIG = CONFIG
    shared.ARTIFACT = ARTIFACT
    shared.LOCK = LOCK
    shared.dfa_features = CAUSAL_FEATURES
    shared._synthetic_guards = OBJECTIVE_GUARDS
    shared.shared.LinearProbeClassifier = GCEClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = CAUSAL_FEATURES
    base.VIBClassifier = GCEClassifier
    base._write = _write_with_provenance


def _provenance_preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    ready = _SHARED_PREFLIGHT(data_dir)
    config = base._read(CONFIG)
    recovery = config["provenance_recovery"]
    hashes = {
        "wrapper": base._sha(Path(__file__)),
        "science_module": base._sha(SCIENCE_MODULE),
        "shared_engine": base._sha(SHARED_ENGINE),
    }
    if hashes["science_module"] != recovery["science_module_sha256"]:
        raise RuntimeError("frozen v36 science module drifted")
    if hashes["shared_engine"] != recovery["shared_engine_sha256"]:
        raise RuntimeError("shared v34 execution engine drifted")
    if ready["runner_sha256"] != hashes["wrapper"]:
        raise RuntimeError("wrapper identity was overwritten")
    ready["provenance"] = {
        "hashes": hashes,
        "wrapper_identity_preserved": True,
        "invalid_parent_artifact_reads": 0,
        "invalid_parent_scientific_metrics_used": 0,
    }
    return ready


def _install_hooks() -> None:
    shared._configure = _configure
    shared.preflight = _provenance_preflight
    _configure()


def preflight(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return shared.preflight(data_dir)


def execute(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return shared.execute(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    value = shared.qa(data_dir)
    checks = value["checks"]
    result_path = ARTIFACT / "result.json"
    if result_path.exists():
        result = base._read(result_path)
        hashes = result["hashes"]
        checks["wrapper_hash"] = hashes["wrapper"] == hashes["runner"] == base._sha(Path(__file__))
        checks["science_module_hash"] = hashes["science_module"] == base._sha(SCIENCE_MODULE)
        checks["shared_engine_hash"] = hashes["shared_engine"] == base._sha(SHARED_ENGINE)
        checks["invalid_parent_unused"] = result["provenance_recovery"]["invalid_parent_artifact_reads"] == result["provenance_recovery"]["invalid_parent_scientific_metrics_used"] == 0
    else:
        ready = _provenance_preflight(data_dir)
        checks["wrapper_hash"] = ready["provenance"]["hashes"]["wrapper"] == base._sha(Path(__file__))
        checks["science_module_hash"] = ready["provenance"]["hashes"]["science_module"] == base._sha(SCIENCE_MODULE)
        checks["shared_engine_hash"] = ready["provenance"]["hashes"]["shared_engine"] == base._sha(SHARED_ENGINE)
        checks["invalid_parent_unused"] = ready["provenance"]["invalid_parent_artifact_reads"] == ready["provenance"]["invalid_parent_scientific_metrics_used"] == 0
    value["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    value = preflight(args.data_dir) if args.preflight else execute(args.data_dir) if args.execute else qa(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
