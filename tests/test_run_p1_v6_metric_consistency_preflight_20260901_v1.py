from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v6_metric_consistency_preflight_20260901_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v6_metric_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_add_only_f1_half_condition_is_exact_over_integer_grid() -> None:
    module = _module()
    for tp in range(1, 12):
        for fp in range(0, 8):
            for fn in range(0, 8):
                for true_add in range(0, fn + 1):
                    for false_add in range(0, 8):
                        assert module.exact_break_even(tp, fp, fn, true_add, false_add)


def test_equality_is_break_even_not_improvement() -> None:
    module = _module()
    anchor, delta = module.add_only_delta(2, 1, 1, 1, 2)
    assert anchor / 2 == 1 / 3
    assert delta == 0


def test_zero_additions_do_not_count_as_improvement() -> None:
    module = _module()
    _anchor, delta = module.add_only_delta(10, 2, 8, 0, 0)
    assert delta == 0
    assert module.exact_break_even(10, 2, 8, 0, 0)
