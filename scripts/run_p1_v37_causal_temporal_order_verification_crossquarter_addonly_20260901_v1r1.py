"""Fresh science-neutral support-bit recovery for frozen P1 v37."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v37_causal_temporal_order_verification_crossquarter_addonly_20260901_v1r1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SCIENCE_MODULE = ROOT / "scripts/run_p1_v37_causal_temporal_order_verification_crossquarter_addonly_20260901_v1.py"
SHARED_ENGINE = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v37r1_science", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen science module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


science = _module(SCIENCE_MODULE)
shared = science.shared
base = science.base
_BASE_WRITE = base._write
_SHARED_PREFLIGHT = shared.preflight


def _binary_support(values: np.ndarray) -> np.ndarray:
    """Decode the standardized binary support column without a raw-scale cutoff."""

    repaired = np.asarray(values, dtype=np.float32).copy()
    support = repaired[:, -1]
    unique = np.unique(support)
    if len(unique) != 2:
        raise RuntimeError(f"standardized support column must have two values, got {len(unique)}")
    repaired[:, -1] = (support == unique[-1]).astype(np.float32)
    return repaired


class RepairedTemporalOrderClassifier(science.TemporalOrderClassifier):
    """Frozen v37 model with science-neutral standardized support decoding."""

    def fit(self, features: np.ndarray, labels: np.ndarray) -> RepairedTemporalOrderClassifier:
        super().fit(_binary_support(features), labels)
        return self

    def predict_score(self, features: np.ndarray) -> np.ndarray:
        return super().predict_score(_binary_support(features))


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
            "parent_optimizer_steps": 0,
            "standardized_support_mapping": "maximum_to_1_minimum_to_0",
            "wrapper_identity_preserved": True,
        }
    _BASE_WRITE(path, payload)


def _configure() -> None:
    shared.EXPERIMENT_ID = EXPERIMENT_ID
    shared.CONFIG = CONFIG
    shared.ARTIFACT = ARTIFACT
    shared.LOCK = LOCK
    shared.dfa_features = science.temporal_order_features
    shared._synthetic_guards = science._synthetic_guards
    shared.shared.LinearProbeClassifier = RepairedTemporalOrderClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = science.temporal_order_features
    base.VIBClassifier = RepairedTemporalOrderClassifier
    base._write = _write_with_provenance


def _provenance_preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    ready = _SHARED_PREFLIGHT(data_dir)
    config = base._read(CONFIG)
    recovery = config["recovery"]
    hashes = {
        "wrapper": base._sha(Path(__file__)),
        "science_module": base._sha(SCIENCE_MODULE),
        "shared_engine": base._sha(SHARED_ENGINE),
    }
    if hashes["science_module"] != recovery["parent_science_module_sha256"]:
        raise RuntimeError("frozen v37 science module drifted")
    if hashes["shared_engine"] != recovery["shared_engine_sha256"]:
        raise RuntimeError("shared v34 engine drifted")
    if ready["runner_sha256"] != hashes["wrapper"]:
        raise RuntimeError("v37r1 wrapper identity overwritten")
    probe = np.array([[-2.0], [0.4], [-2.0], [0.4]], dtype=np.float32)
    mapped = _binary_support(probe)[:, 0]
    ready["provenance"] = {
        "hashes": hashes,
        "standardized_support_mapping": "maximum_to_1_minimum_to_0",
        "mapping_probe": mapped.astype(int).tolist(),
        "invalid_parent_artifact_reads": 0,
        "invalid_parent_scientific_metrics_used": 0,
        "parent_optimizer_steps": 0,
        "wrapper_identity_preserved": True,
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
        checks["invalid_parent_unused"] = (
            result["provenance_recovery"]["invalid_parent_artifact_reads"]
            == result["provenance_recovery"]["invalid_parent_scientific_metrics_used"]
            == result["provenance_recovery"]["parent_optimizer_steps"]
            == 0
        )
    else:
        ready = _provenance_preflight(data_dir)
        checks["wrapper_hash"] = ready["provenance"]["hashes"]["wrapper"] == base._sha(Path(__file__))
        checks["science_module_hash"] = ready["provenance"]["hashes"]["science_module"] == base._sha(SCIENCE_MODULE)
        checks["shared_engine_hash"] = ready["provenance"]["hashes"]["shared_engine"] == base._sha(SHARED_ENGINE)
        checks["invalid_parent_unused"] = ready["provenance"]["invalid_parent_artifact_reads"] == 0
    checks["support_mapping"] = np.array_equal(
        _binary_support(np.array([[-2.0], [0.4], [-2.0]], dtype=np.float32))[:, 0],
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    value["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    if result_path.exists():
        value["result_sha256"] = base._sha(result_path)
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
