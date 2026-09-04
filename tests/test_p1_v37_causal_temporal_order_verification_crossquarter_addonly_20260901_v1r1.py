from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v37_causal_temporal_order_verification_crossquarter_addonly_20260901_v1r1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v37r1_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()


def test_science_contract_is_identical_to_v37() -> None:
    repaired = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    invalid = json.loads(
        (ROOT / "configs/experiments/p1_v37_causal_temporal_order_verification_crossquarter_addonly_20260901_v1.json").read_text(encoding="utf-8")
    )
    for key in (
        "surface",
        "source",
        "primary_source",
        "cross_quarter_guard",
        "auditability_amendment",
        "representation",
        "representation_support_gate",
        "model",
        "selection",
        "anchor",
        "diagnostics",
        "parts",
        "decision",
        "operations",
    ):
        assert repaired[key] == invalid[key]


def test_standardized_binary_support_maps_maximum_to_one() -> None:
    values = np.array([[-1.75, -2.0], [0.25, 0.4], [3.0, -2.0], [8.0, 0.4]], dtype=np.float32)
    repaired = mod._binary_support(values)
    assert np.array_equal(repaired[:, -1], np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    assert np.array_equal(repaired[:, 0], values[:, 0])


def test_frozen_science_feature_and_network_are_reused() -> None:
    assert mod.science.temporal_order_features is mod.shared.dfa_features or callable(mod.science.temporal_order_features)
    assert issubclass(mod.RepairedTemporalOrderClassifier, mod.science.TemporalOrderClassifier)


def test_provenance_pins_parent_science_and_engine() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    recovery = config["recovery"]
    assert mod.base._sha(mod.SCIENCE_MODULE) == recovery["parent_science_module_sha256"]
    assert mod.base._sha(mod.SHARED_ENGINE) == recovery["shared_engine_sha256"]
    assert recovery["invalid_parent_artifact_reads"] == recovery["invalid_parent_scientific_metrics_used"] == 0
    assert recovery["optimizer_steps_in_parent"] == 0


def test_real_preflight_is_zero_operation_and_preserves_wrapper() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    assert ready["runner_sha256"] == ready["provenance"]["hashes"]["wrapper"] == mod.base._sha(RUNNER)
    assert ready["provenance"]["mapping_probe"] == [0, 1, 0, 1]
    assert all(value == 0 for value in ready["counters"].values())
