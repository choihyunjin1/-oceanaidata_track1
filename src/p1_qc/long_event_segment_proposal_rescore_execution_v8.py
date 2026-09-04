"""r8 infrastructure wrapper for the frozen P1 Cycle-1 science.

All numerical behavior remains in the byte-pinned r6 implementation.  This
module only changes the append-only infrastructure identity after the r8
isolated-Python and true-parent capability firewall has authenticated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v2_infrastructure_v8"
SCIENTIFIC_EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v1"
R6_EXECUTION_MODULE_SHA256 = (
    "8b91db9234bde301730728dcffa4fd2c014b099d0c518571ddaa733608b81636"
)
R8_AMENDMENT_SHA256 = (
    "003672cac416b5a691357ef108c0401b5bfb9661d4daa3adbe3c6efce68f4b61"
)
MAXIMUM_LIFETIME_PHYSICAL_FITS = 72
MAXIMUM_SCIENTIFIC_MATERIALIZATIONS = 21
MAXIMUM_OUTER_SCORES = 1


def run_authorized_screen(
    state: Mapping[str, Any],
    numerical: Any,
    closure: Mapping[str, Any],
    journal: Any,
    deadline_epoch: float,
) -> dict[str, Any]:
    """Delegate once to frozen r6 after r8 trust activation."""

    from p1_qc import long_event_segment_proposal_rescore_execution_v6 as frozen

    if frozen.EXPERIMENT_ID != (
        "p1_long_event_segment_proposal_rescore_20260826_v2_infrastructure_v6"
    ):
        raise RuntimeError("frozen r6 execution identity changed")
    if frozen.SCIENTIFIC_EXPERIMENT_ID != SCIENTIFIC_EXPERIMENT_ID:
        raise RuntimeError("frozen scientific identity changed")
    if tuple(frozen.ROUND_B_SEEDS) != (20260813, 20260829, 20260847):
        raise RuntimeError("Round-B seeds changed")
    if tuple(frozen.SEGMENT_SEEDS) != (20260826, 20260843, 20260871):
        raise RuntimeError("segment seeds changed")

    previous = frozen.EXPERIMENT_ID
    frozen.EXPERIMENT_ID = EXPERIMENT_ID
    try:
        result = dict(
            frozen.run_authorized_screen(
                state,
                numerical,
                closure,
                journal,
                deadline_epoch,
            )
        )
    finally:
        frozen.EXPERIMENT_ID = previous

    expected = {
        "claims": 1,
        "inner_anchor_physical_fits": 9,
        "inner_segment_physical_fits": 54,
        "outer_segment_physical_fits": 9,
        "physical_fits": MAXIMUM_LIFETIME_PHYSICAL_FITS,
        "scientific_materializations": MAXIMUM_SCIENTIFIC_MATERIALIZATIONS,
        "outer_scores": MAXIMUM_OUTER_SCORES,
        "candidate_files": 0,
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
        "uploads": 0,
    }
    if result.get("operation_counters") != expected:
        raise RuntimeError("frozen r6 operation graph changed")
    if result.get("scientific_experiment_id") != SCIENTIFIC_EXPERIMENT_ID:
        raise RuntimeError("frozen result scientific identity changed")
    result["experiment_id"] = EXPERIMENT_ID
    result["infrastructure_revision"] = "v8_isolated_python_parent_capability"
    result["r8_amendment_sha256"] = R8_AMENDMENT_SHA256
    return result
