"""Continuation guards only; no model fit/source data/official input."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_p1_tuning_depth_stress_20260906_v2 as m  # noqa: E402


@pytest.mark.parametrize("values", [np.array([]), np.array([np.nan, 0]), np.array([np.inf, 0]),
                                    np.array([-.1, .5]), np.array([.5, 1.1]), np.array([[.5, .5]])])
def test_invalid_partial_probabilities_rejected(values):
    assert not m.valid_probability(values, 2)


def test_probability_guard_accepts_exact_boundary_values():
    assert m.valid_probability(np.array([0., 1.]), 2)


def test_synthetic_differential_contract_still_passes():
    assert m.legacy.synthetic_audit()["status"] == "PASS"
    assert m.OUT != m.legacy.OUT and m.CONFIG != m.legacy.CONFIG
    assert m.REUSED == ("H2_2024", "H1_2025")


def test_no_fit_or_official_or_selection_code():
    code = Path(m.__file__).read_text(encoding="utf-8")
    for prohibited in (' / "test.csv"', "sample_submission", ".to_csv(", "model.fit(", "_fit_model(", "select_inner("):
        assert prohibited not in code
