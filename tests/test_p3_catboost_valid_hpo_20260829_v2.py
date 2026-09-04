from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from p3_wave.catboost_valid_hpo_v2 import (
    VALID_STRUCTURES,
    control_candidate,
    materialize_grid,
    validate_schedule,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p3_catboost_valid_hpo_20260829_v2.json"
GRID_PATH = ROOT / "configs/experiments/p3_catboost_valid_hpo_20260829_v2.grid.json"
RUNNER_PATH = ROOT / "scripts/run_p3_catboost_valid_hpo_20260829_v2.py"
SPEC = importlib.util.spec_from_file_location("p3_catboost_valid_hpo_v2_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_grid_contains_only_36_valid_unique_challengers() -> None:
    grid = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    challengers = materialize_grid(grid)
    assert len(challengers) == 36
    structures = {
        (row["parameters"]["boosting_type"], row["parameters"]["grow_policy"])
        for row in challengers
    }
    assert structures == set(VALID_STRUCTURES)
    assert ("Ordered", "Depthwise") not in structures
    assert control_candidate(grid)["candidate_id"] == "control_incumbent"


def test_schedule_is_exact_138_historical_selection_fits() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    assert validate_schedule(config) == 138
    assert config["confirmation"]["maximum_confirmation_fit_count"] == 3
    assert config["confirmation"]["maximum_full_refit_fit_count_after_gate_pass"] == 1


def test_wrong_token_stops_before_smoke_or_attempt_lock(tmp_path: Path) -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    lock = ROOT / config["outputs"]["attempt_lock"]
    before = lock.read_bytes() if lock.exists() else None
    try:
        RUNNER.ENGINE.execute_hpo(CONFIG_PATH, tmp_path, "WRONG_TOKEN")
    except RUNNER.HPOContractError as exc:
        assert "authorization token differs" in str(exc)
    else:
        raise AssertionError("wrong token unexpectedly accepted")
    after = lock.read_bytes() if lock.exists() else None
    assert after == before


def test_config_extends_exact_frozen_v1_and_uses_new_outputs() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    assert config["experiment_id"] == "p3_catboost_valid_hpo_20260829_v2"
    assert "p3_catboost_valid_hpo_20260829_v2" in config["outputs"]["attempt_lock"]
    assert config["frozen_pipeline"]["kma"]["deployment_alpha"] == 0.4
    assert config["selection"]["expected_feature_count"] == 591
