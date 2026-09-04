from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.matched_budget_local_compare import (
    apply_fixed_long_lead_shrink,
    complete_case_bootstrap_delta,
    metric_summary,
    residual_correlation,
    validate_surface,
)


ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    rows = []
    for anchor_id, station, offset in ((1, "G-ORS", 0.0), (2, "I-ORS", 0.2)):
        for lead in (3, 6, 9, 12, 18, 24):
            target = 1.0 + offset + lead / 100.0
            prediction = target + 0.05 + lead / 1000.0
            rows.append(
                {
                    "fold": "fixed",
                    "station": station,
                    "anchor_id": anchor_id,
                    "lead_h": lead,
                    "target_hs": target,
                    "reference": prediction,
                    "candidate": prediction,
                }
            )
    return pd.DataFrame(rows)


def test_fixed_long_lead_shrink_is_exact_noop_on_short_leads() -> None:
    base = np.arange(6, dtype=np.float64)
    persistence = base + 2.0
    leads = np.array([3, 6, 9, 12, 18, 24])
    result = apply_fixed_long_lead_shrink(
        base, persistence, leads, weight=0.25, active_leads=(12, 18, 24)
    )
    np.testing.assert_array_equal(result[:3], base[:3])
    np.testing.assert_allclose(result[3:], base[3:] + 0.5, rtol=0.0, atol=0.0)


def test_complete_case_surface_and_zero_delta_bootstrap() -> None:
    frame = _frame()
    audit = validate_surface(
        frame, expected_cases=2, expected_rows=12, expected_leads=(3, 6, 9, 12, 18, 24)
    )
    assert audit["complete_case_surface"] is True
    assert metric_summary(frame, "candidate")["rmse_m"] == metric_summary(frame, "reference")["rmse_m"]
    bootstrap = complete_case_bootstrap_delta(
        frame,
        candidate="candidate",
        reference="reference",
        replicates=100,
        seed=20260825,
    )
    assert bootstrap["delta_candidate_minus_reference_ci90_m"] == [0.0, 0.0]
    assert bootstrap["median_delta_m"] == 0.0
    assert residual_correlation(frame, "candidate", "reference") == pytest.approx(1.0)


def test_preregistered_budget_and_quarantine_contract() -> None:
    config = json.loads(
        (ROOT / "configs/experiments/p3_matched_budget_local_compare_20260825_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(config["coefficient_family"]["settings"]) == 3
    assert [setting["persistence_weight"] for setting in config["coefficient_family"]["settings"]] == [0.2, 0.225, 0.25]
    assert len(config["structural_families"]) == 3
    assert all(family["seed_count_each_side"] == 3 for family in config["structural_families"])
    assert all(family["fit_cells_each_side"] == 45 for family in config["structural_families"])
    assert config["data_boundary"]["official_evaluation_value_reads"] == 0
    assert config["data_boundary"]["submission_files_generated"] == 0
    assert config["data_boundary"]["era5_path_or_process_accesses"] == 0
    density = next(
        item
        for item in config["excluded_existing_candidates"]
        if item["id"] == "target_mix_density_reweighted_catboost"
    )
    assert density["override_allowed"] is False
