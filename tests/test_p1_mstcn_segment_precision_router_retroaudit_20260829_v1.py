from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("p1_segment_router", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_add_only_f1_condition() -> None:
    module = _module()
    assert module.segment_is_beneficial(
        true_positive_rows=5, false_positive_rows=5, incumbent_f1=0.90
    )
    assert not module.segment_is_beneficial(
        true_positive_rows=4, false_positive_rows=6, incumbent_f1=0.90
    )
    assert not module.segment_is_beneficial(
        true_positive_rows=0, false_positive_rows=0, incumbent_f1=0.90
    )


def test_feature_contract_is_small_and_type_features_are_separate() -> None:
    module = _module()
    assert len(module.CORE_NUMERIC) == 14
    assert module.CATEGORICAL == ["station", "layer"]
    assert module.TYPE_NUMERIC == [
        "type_spike_mean",
        "type_noise_mean",
        "type_flatline_mean",
        "type_offset_mean",
        "type_drift_mean",
        "type_entropy_mean",
    ]
