from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from p2_restore.authoritative_nested_surrogate_contract import validate_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_recipe_20260825_v1.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _recipe_and_evidence() -> tuple[dict, dict[str, dict]]:
    recipe = _read(CONFIG_PATH)
    evidence = {
        name: _read(PROJECT_ROOT / spec["path"])
        for name, spec in recipe["evidence"].items()
    }
    return recipe, evidence


def _runner() -> ModuleType:
    path = (
        PROJECT_ROOT
        / "scripts/validate_p2_authoritative_nested_surrogate_recipe_v1.py"
    )
    spec = importlib.util.spec_from_file_location("p2_nested_contract_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sealed_actual_contract_passes_and_builds_45_cells() -> None:
    recipe, evidence = _recipe_and_evidence()
    result = validate_contract(recipe, evidence)
    assert result.decision["verdict"] == "NEW_AUTHORITATIVE_SURROGATE_REQUIRED"
    assert result.decision["exact_comparison_reproducible"] is False
    assert result.decision["matched_evidence"]["direction_conflict_confirmed"] is True
    assert result.qa["exactness_dimension_count"] == 8
    assert result.qa["seeded_pipeline_cell_count"] == 45
    cells = result.comparison_preregistration["seeded_execution_plan"]
    assert len({cell["cell_id"] for cell in cells}) == 45
    assert not any(cell["fit_authorized"] for cell in cells)


def test_complete_pipeline_seed_change_is_rejected() -> None:
    recipe, evidence = _recipe_and_evidence()
    changed = copy.deepcopy(recipe)
    changed["authoritative_nested_surrogate_recipe"][
        "complete_pipeline_seed_contract"
    ]["seeds"][-1] = 99
    with pytest.raises(ValueError, match="complete seeds changed"):
        validate_contract(changed, evidence)


def test_historical_blocker_change_is_rejected() -> None:
    recipe, evidence = _recipe_and_evidence()
    changed_evidence = copy.deepcopy(evidence)
    changed_evidence["exact_audit"]["blocking_recipe_gaps"].pop()
    with pytest.raises(ValueError, match="blocking exact-recipe gaps changed"):
        validate_contract(recipe, changed_evidence)


def test_runner_check_only_verifies_all_aggregate_hashes() -> None:
    result = _runner().run(CONFIG_PATH, execute=False)
    assert result["status"] == "PASS"
    assert result["aggregate_json_evidence_count"] == 10
    assert result["new_model_fits"] == 0
    assert result["new_score_reads"] == 0
    assert result["verdict"] == "NEW_AUTHORITATIVE_SURROGATE_REQUIRED"
