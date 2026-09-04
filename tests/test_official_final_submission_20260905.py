from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "final_submission_20260905.json"


def test_frozen_contract_is_clean_and_complete() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    policy = config["policy"]
    assert policy["organizer_distributed_data_only"] is True
    assert policy["source_data_redistribution_allowed"] is False
    assert policy["maximum_upload_file_bytes"] == 50_000_000
    assert all(
        policy[name] == 0
        for name in (
            "external_observation_rows",
            "external_reanalysis_rows",
            "external_forecast_rows",
            "pretrained_weight_files_loaded",
            "hidden_truth_rows_read",
        )
    )
    assert {config[p]["expected_rows"] for p in ("P1", "P2", "P3")} == {
        169_011,
        26_061,
        1_200,
    }


def test_notebooks_are_problem_local_and_top_to_bottom() -> None:
    for problem in ("P1", "P2", "P3"):
        path = (
            ROOT
            / "notebooks"
            / "final_submission_20260905"
            / problem
            / f"{problem}_final_submission.ipynb"
        )
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert f"{problem}_DATA_DIR" in source
        assert "run_submission.preflight" in source
        assert "run_submission.materialize" in source
        assert "READY_EXACT_NOT_UPLOADED" in source
        for other in {"P1", "P2", "P3"} - {problem}:
            assert f"{other}_DATA_DIR" not in source


def test_runners_have_no_network_or_upload_surface() -> None:
    forbidden = ("requests.", "http://", "https://oceanaidata", "selenium", "playwright", "upload(")
    for problem in ("P1", "P2", "P3"):
        path = ROOT / "scripts" / "final_submission_20260905" / problem / "run_submission.py"
        source = path.read_text(encoding="utf-8").lower()
        assert all(token.lower() not in source for token in forbidden)


def test_local_package_receipts_when_present() -> None:
    package = ROOT / "artifacts" / "official_final_submission_20260905"
    if not package.is_dir():
        return
    master = json.loads((package / "MASTER_MANIFEST.json").read_text(encoding="utf-8"))
    assert master["status"] == "LOCAL_READY_EXACT_NOT_UPLOADED"
    assert master["atomic_problem_directories"] is True
    assert master["notebooks_executed"] is True
    for problem in ("P1", "P2", "P3"):
        receipt = master["receipts"][problem]
        assert receipt["status"] == "READY_EXACT_NOT_UPLOADED"
        assert receipt["candidate_hash_exact"] is True
        assert receipt["package_atomic"] is True
        assert (package / problem / "outputs" / f"{problem}_submission.csv").is_file()
