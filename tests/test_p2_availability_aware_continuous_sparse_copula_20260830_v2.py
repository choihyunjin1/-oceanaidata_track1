from __future__ import annotations

import numpy as np
import pytest

from p2_restore import (
    p2_availability_aware_continuous_sparse_copula_20260830_v1 as v1,
)
from p2_restore import (
    p2_availability_aware_continuous_sparse_copula_20260830_v2 as v2,
)


def test_overlay_is_sealed_and_inherits_science_contract_exactly() -> None:
    merged, overlay = v2.load_config()
    base = v1.load_config()
    assert overlay["technical_overlay"]["changed_component"] == (
        "post_prediction_physical_domain_guard_only"
    )
    assert overlay["technical_overlay"]["prediction_values_changed_by_overlay"] is False
    assert overlay["execution_policy"]["v1_reexecution_allowed"] is False
    assert overlay["inherited_scientific_contract"]["tail_or_cvar_role"].startswith(
        "DIAGNOSTIC_SENSITIVITY_ONLY"
    )
    for key in (
        "source_contract",
        "frozen_historical_windows",
        "state",
        "dependence",
        "correction",
        "primary_decision",
        "resource_contract",
        "immutable_training_inputs",
    ):
        assert merged[key] == base[key]
    assert merged["artifact_path"].endswith("_v2/result.json")


def test_relative_guard_preserves_preexisting_inactive_extrema_without_mutation() -> None:
    reference = np.array([12.0, 72.5, -8.25, 21.0])
    candidate = reference.copy()
    correction = np.zeros_like(reference)
    active = np.array([True, False, False, True])
    before = candidate.copy()
    receipt = v2._relative_physical_domain_guard(
        reference=reference,
        candidate=candidate,
        correction=correction,
        active_rows=active,
    )
    assert np.array_equal(candidate, before)
    assert receipt["reference_outside_count"] == 2
    assert receipt["new_candidate_outside_count"] == 0
    assert receipt["active_candidate_outside_count"] == 0
    assert receipt["prediction_values_changed_by_overlay"] is False


def test_relative_guard_allows_bounded_in_domain_active_correction() -> None:
    reference = np.array([10.0, 20.0, 30.0])
    correction = np.array([0.2, -0.2, 0.0])
    candidate = reference + correction
    receipt = v2._relative_physical_domain_guard(
        reference=reference,
        candidate=candidate,
        correction=correction,
        active_rows=np.array([True, True, False]),
    )
    assert receipt["candidate_all_finite"] is True
    assert receipt["candidate_outside_count"] == 0


@pytest.mark.parametrize(
    ("reference", "candidate", "active", "message"),
    [
        ([44.9], [45.1], [True], "new physical-domain violation"),
        ([80.0], [79.9], [False], "pre-existing reference extreme"),
        ([80.0], [80.0], [True], "marked active"),
        ([20.0], [20.1], [False], "inactive candidate changed"),
        ([20.0], [np.nan], [True], "became nonfinite"),
    ],
)
def test_relative_guard_rejects_every_forbidden_transition(
    reference: list[float],
    candidate: list[float],
    active: list[bool],
    message: str,
) -> None:
    reference_array = np.asarray(reference, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    with pytest.raises(v2.OverlayContractError, match=message):
        v2._relative_physical_domain_guard(
            reference=reference_array,
            candidate=candidate_array,
            correction=candidate_array - reference_array,
            active_rows=np.asarray(active, dtype=bool),
        )


def test_overlay_config_keeps_primary_metric_and_diagnostics_roles() -> None:
    merged, overlay = v2.load_config()
    inherited = overlay["inherited_scientific_contract"]
    assert inherited["primary_rows"] == 69_850
    assert merged["primary_decision"]["metric"] == inherited["primary_metric"]
    assert all(
        merged["primary_decision"][name] is False
        for name in (
            "minimum_improved_windows_is_hard_veto",
            "worst_season_cap_is_hard_veto",
            "all_layers_nonworse_is_hard_veto",
            "support_is_hard_veto_after_level0_validity",
            "correction_magnitude_is_hard_veto",
        )
    )
    assert inherited["structural_correction_bound_c"] == [-0.2, 0.2]
    assert inherited["outer_dependence_model_fits"] == 3
    assert inherited["inner_selection_fits"] == 0
    assert inherited["hpo_trials"] == 0
