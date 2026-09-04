from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p3_v5_extratrees_competition_adjudication_20260831_v7 as module  # noqa: E402


def test_frozen_lineage_and_seed() -> None:
    spec = module.SPECS[module.TARGET_SPEC_INDEX]
    assert spec.name == module.TARGET_NAME
    assert spec.family == "extra_trees"
    assert spec.policy == "hard_0p50"
    assert module.FULL_FIT_SEED == module.BOOTSTRAP_SEED + 1001


def test_adjudication_separates_scientific_and_competition() -> None:
    adjudication, features, cases, profile = module.adjudicate()
    assert adjudication["scientific"]["status"].startswith("SCIENTIFIC_INCONCLUSIVE")
    assert adjudication["competition"]["status"].startswith("COMPETITION_EXPECTED_VALUE_PASS")
    assert adjudication["competition"]["central_projected_point_delta"] > 0.0
    assert adjudication["competition"]["conservative_projected_point_delta"] < 0.0
    assert len(features) == 608
    assert len(cases) == 182
    assert profile["official_feature_rows_read_before_internal_gate"] == 0
