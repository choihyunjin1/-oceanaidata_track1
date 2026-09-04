from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v11 as cycle  # noqa: E402


def test_family_gate_is_hard_router_v2() -> None:
    config = cycle.load_contract()
    assert config["transport_family"]["tier_id"] == "HARD_CONDITIONAL_ROUTER"
    assert np.isclose(config["decision_policy"]["transport_penalty_points"], 0.3219056897594759)
    assert np.isclose(config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"], 0.33190568975947593)


def test_guarded_removal_respects_budget_and_station_share() -> None:
    frame = pd.DataFrame({"station": ["A"] * 8 + ["B"] * 2})
    scope = np.ones(10, dtype=bool)
    score = np.arange(10, dtype=float)[::-1]
    config = {"decision_policy": {"maximum_station_intervention_share": 0.8}}
    mask, receipt = cycle.guarded_removals(frame, scope, score, 0.5, 10, config)
    assert mask.sum() == 5
    assert receipt["maximum_station_share"] <= 0.8


def test_guard_abstains_when_only_one_station_can_be_touched() -> None:
    frame = pd.DataFrame({"station": ["A"] * 10})
    mask, receipt = cycle.guarded_removals(frame, np.ones(10, dtype=bool), np.arange(10, dtype=float), 0.5, 10, {"decision_policy": {"maximum_station_intervention_share": 0.8}})
    assert mask.sum() == 0
    assert receipt["status"] == "STATION_CONCENTRATION_ABSTAIN"


def test_native_handles_numpy_values(tmp_path: Path) -> None:
    assert cycle.native({"ok": np.bool_(True), "n": np.int64(1)}) == {"ok": True, "n": 1}
    assert cycle.native({"threshold": float("inf")}) == {"threshold": None}
    path = tmp_path / "abstain.json"
    cycle.write_json(path, {"status": "ABSTAIN", "threshold": float("inf")})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "ABSTAIN",
        "threshold": None,
    }
