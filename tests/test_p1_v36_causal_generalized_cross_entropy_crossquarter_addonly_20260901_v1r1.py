from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1r1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v36r1_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()


def test_science_contract_is_identical_to_v36() -> None:
    repaired = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    invalid = json.loads(mod.SCIENCE_MODULE.with_name("p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1.py").parent.parent.joinpath("configs/experiments/p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1.json").read_text(encoding="utf-8"))
    for key in ("representation", "representation_support_gate", "model", "selection", "anchor", "diagnostics", "parts", "decision"):
        assert repaired[key] == invalid[key]


def test_frozen_science_objects_are_reused_without_redefinition() -> None:
    assert mod.GCEClassifier is mod.science.GCEClassifier
    assert mod.CAUSAL_FEATURES is mod.science.CAUSAL_FEATURES
    assert mod.OBJECTIVE_GUARDS is mod.science._objective_guards


def test_provenance_contract_pins_science_and_engine_not_invalid_artifacts() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    recovery = config["provenance_recovery"]
    assert recovery["invalid_parent_artifact_reads"] == recovery["invalid_parent_scientific_metrics_used"] == 0
    assert mod.base._sha(mod.SCIENCE_MODULE) == recovery["science_module_sha256"]
    assert mod.base._sha(mod.SHARED_ENGINE) == recovery["shared_engine_sha256"]


def test_imported_configure_is_replaced_and_wrapper_identity_survives() -> None:
    mod._install_hooks()
    assert mod.shared._configure is mod._configure
    assert mod.shared.preflight is mod._provenance_preflight
    assert Path(mod.base.__file__).resolve() == RUNNER.resolve()


def test_real_preflight_records_three_distinct_provenance_hashes() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    hashes = ready["provenance"]["hashes"]
    assert ready["runner_sha256"] == hashes["wrapper"] == mod.base._sha(RUNNER)
    assert hashes["science_module"] == mod.base._sha(mod.SCIENCE_MODULE)
    assert hashes["shared_engine"] == mod.base._sha(mod.SHARED_ENGINE)
    assert ready["provenance"]["wrapper_identity_preserved"]
    assert all(value == 0 for value in ready["counters"].values())
