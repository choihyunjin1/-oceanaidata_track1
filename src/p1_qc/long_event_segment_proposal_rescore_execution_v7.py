"""r7 infrastructure wrapper for the frozen P1 segment-rescore science.

The numerical implementation remains the byte-pinned r6 module.  This wrapper
changes only the infrastructure experiment identity and refuses any drift in
the frozen 9 + 54 + 9 fit / 21 materialization result contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v2_infrastructure_v7"
SCIENTIFIC_EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v1"
R6_EXECUTION_MODULE_SHA256 = (
    "8b91db9234bde301730728dcffa4fd2c014b099d0c518571ddaa733608b81636"
)
R7_AMENDMENT_SHA256 = (
    "71563c954a5c529044d82c63af0e44ddf313dcc55b787c784afd153fc14434ff"
)
MAXIMUM_LIFETIME_PHYSICAL_FITS = 72
MAXIMUM_SCIENTIFIC_MATERIALIZATIONS = 21


def run_authorized_screen(
    state: Mapping[str, Any],
    numerical: Any,
    closure: Mapping[str, Any],
    journal: Any,
    deadline_epoch: float,
) -> dict[str, Any]:
    """Delegate to the frozen r6 science after the r7 trust boundary.

    Importing this module is permitted only from the sealed private project
    snapshot.  The r7 runner installs an origin guard before this import and
    holds both this file and the complete r6 dependency tree for the worker
    lifetime.
    """

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
        "outer_scores": 1,
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
    result["infrastructure_revision"] = "v7_full_runtime_replay_firewall"
    result["r7_amendment_sha256"] = R7_AMENDMENT_SHA256
    return result
