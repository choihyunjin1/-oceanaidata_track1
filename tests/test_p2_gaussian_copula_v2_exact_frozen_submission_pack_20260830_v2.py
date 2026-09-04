from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from p2_restore.p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1 import (
    load_json,
    sha256_file,
)
from scripts import (
    run_p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v2 as runner,
)


def test_completion_runner_help_proves_scripts_loader_contract() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(runner.__file__).resolve()), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr
    assert "--execute" in completed.stdout


def test_completion_overlay_canonical_hash_and_pins() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    overlay_path = (
        repo_root
        / "configs"
        / "experiments"
        / "p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v2.json"
    )
    overlay = load_json(overlay_path)
    assert sha256_file(overlay_path) == runner.EXPECTED_OVERLAY_CONFIG_SHA256
    for record in overlay["base_attempt"].values():
        assert sha256_file(repo_root / record["path"]) == record["sha256"]


def test_completion_overlay_changes_only_loader_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    overlay = load_json(runner.DEFAULT_CONFIG)
    base_config = load_json(runner.DEFAULT_BASE_CONFIG)
    receipts = runner.validate_overlay(repo_root, overlay, base_config)
    assert set(receipts) == {
        "config",
        "runner",
        "scientific_module",
        "failure_receipt",
    }
    assert overlay["repair"]["loader_contract_lines_added"] == 1
    assert overlay["repair"]["model_or_prediction_logic_changed"] is False
    assert overlay["repair"]["data_or_query_mapping_changed"] is False
    assert overlay["repair"]["gates_or_qa_changed"] is False


def test_frozen_model_selection_and_resource_contract_are_unchanged() -> None:
    overlay = load_json(runner.DEFAULT_CONFIG)
    base_config = load_json(runner.DEFAULT_BASE_CONFIG)
    frozen = base_config["frozen_recipe"]
    assert len(frozen["training_blocks"]) == 7
    assert frozen["selected_shrinkage"] == 0.5
    assert frozen["inner_search_or_hpo"] == 0
    assert frozen["expected_refit_model_receipt_sha256"] == overlay[
        "frozen_scientific_invariants"
    ]["expected_refit_model_receipt_sha256"]
    assert base_config["execution_policy"] == overlay["execution_policy"]
