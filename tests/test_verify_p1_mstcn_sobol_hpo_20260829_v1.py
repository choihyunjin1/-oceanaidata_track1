from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "verify_p1_mstcn_sobol_hpo_20260829_v1.py"


def _load_verifier():
    name = "verify_p1_mstcn_sobol_hpo_20260829_v1_tested"
    spec = importlib.util.spec_from_file_location(name, VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_verifier_is_artifact_only_and_has_no_external_interface() -> None:
    verifier = _load_verifier()
    source = VERIFIER_PATH.read_text(encoding="utf-8").casefold()
    assert verifier.EXPERIMENT_ID == "p1_mstcn_sobol_hpo_20260829_v1"
    assert "requests." not in source
    assert "selenium" not in source
    assert "pandas" not in source
    assert "pyarrow" not in source
    assert "import torch" not in source
    assert "from torch" not in source
    assert "*.csv" in source
    assert 'discovery_receipt["score_sha256"]' in source
    assert 'top2_receipt["score_sha256"]' in source
    assert 'receipt["epochs"] == 150' in source
    assert 'receipt["nonfinite_count_total"] == 0' in source
