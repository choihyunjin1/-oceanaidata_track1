from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p1_mstcn_deployable_type_veto_stability_20260829_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("p1_deployable_type_veto", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_deployment_features_exclude_incumbent_probability() -> None:
    module = _module()
    assert "incumbent_probability_mean" not in module.FEATURES
    assert "incumbent_probability_max" not in module.FEATURES
    assert set(module.base.TYPE_NUMERIC).issubset(module.FEATURES)


def test_candidate_consensus_preserves_incumbent_when_all_vetoed() -> None:
    module = _module()
    bundles = module.base.load_bundles()
    q4 = bundles["2025_q4"]
    prediction, additions = module.candidate_from_acceptance(
        q4, module.np.zeros(len(q4.segments), dtype=bool)
    )
    assert module.np.array_equal(prediction, q4.incumbent)
    assert additions["accepted_segments"] == 0
    assert additions["accepted_rows"] == 0
