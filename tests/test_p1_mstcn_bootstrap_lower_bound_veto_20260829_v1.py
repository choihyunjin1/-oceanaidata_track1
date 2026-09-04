from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p1_mstcn_bootstrap_lower_bound_veto_20260829_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("p1_bootstrap_lower_bound", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lower_bound_contract_is_conservative_and_deployable() -> None:
    module = _module()
    config = module.json.loads(module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["acceptance_frequency"] == 0.9
    assert "incumbent_probability_mean" not in module.prior.FEATURES
    assert "incumbent_probability_max" not in module.prior.FEATURES


def test_no_acceptance_is_exact_fallback() -> None:
    module = _module()
    q4 = module.prior.base.load_bundles()["2025_q4"]
    prediction, additions = module.prior.candidate_from_acceptance(
        q4, module.np.zeros(len(q4.segments), dtype=bool)
    )
    assert module.np.array_equal(prediction, q4.incumbent)
    assert additions["accepted_rows"] == 0
