from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p3_era5_context_transfer_dependency_recovery_20260828_v2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_p3_era5_recovery_v2_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_environment_preflight_supports_frozen_continuation() -> None:
    module = _load_module()
    result = module._environment_preflight()
    assert result["passed"] is True
    assert result["feature_shape"] == [32, 286]
    assert result["stage2_tree_count"] > result["stage1_tree_count"]
    assert result["finite_predictions"] is True


def test_recovery_changes_only_id_and_output() -> None:
    module = _load_module()
    frozen = module._load_frozen_runner(ROOT)
    receipt = module._install_dependency_recovery_contract(frozen, ROOT)
    config, _scope, paths = frozen._load_contract(ROOT)
    assert config["experiment_id"] == module.EXPECTED_ID
    assert paths.output == (ROOT / module.EXPECTED_OUTPUT_REL).resolve()
    assert receipt["scientific_overrides"] == []
    assert receipt["operational_overrides"] == ["experiment_id", "artifact_dir"]
