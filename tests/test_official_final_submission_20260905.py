from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "final_submission_20260905.json"
PROBLEMS = ("P1", "P2", "P3")


def test_model_driven_contract_is_clean_and_complete() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    policy = config["policy"]
    assert config["status"] == "MODEL_DRIVEN_CLEAN_LINEAGE_FINAL_PACKAGES"
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
    assert {config[p]["expected_rows"] for p in PROBLEMS} == {169_011, 26_061, 1_200}
    for problem in PROBLEMS:
        assert config[problem]["training_notebook"].endswith(f"/{problem}/TRAIN.ipynb")
        assert config[problem]["prediction_notebook"].endswith(
            f"/{problem}/PREDICT.ipynb"
        )


def test_notebooks_separate_training_from_model_inference() -> None:
    for problem in PROBLEMS:
        root = ROOT / "notebooks" / "final_submission_20260905" / problem
        train = nbformat.read(root / "TRAIN.ipynb", as_version=4)
        predict = nbformat.read(root / "PREDICT.ipynb", as_version=4)
        train_source = "\n".join(cell.source for cell in train.cells)
        predict_source = "\n".join(cell.source for cell in predict.cells)
        assert "train_model.train" in train_source
        assert "RUN_FULL_SCRATCH_RETRAIN" in train_source
        assert "run_submission.preflight" in predict_source
        assert "run_submission.materialize" in predict_source
        assert "READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED" in predict_source
        assert "frozen_candidate_csv_copy" in predict_source
        for other in set(PROBLEMS) - {problem}:
            assert f"{other}_DATA_DIR" not in train_source
            assert f"{other}_DATA_DIR" not in predict_source


def test_problem_runtime_has_train_model_predict_and_no_network_surface() -> None:
    forbidden = (
        "requests.",
        "http://",
        "https://oceanaidata",
        "selenium",
        "playwright",
        "upload(",
    )
    for problem in PROBLEMS:
        root = ROOT / "scripts" / "final_submission_20260905" / problem
        for name in ("train_model.py", "predict_submission.py", "run_submission.py"):
            path = root / name
            assert path.is_file()
            source = path.read_text(encoding="utf-8").lower()
            assert all(token.lower() not in source for token in forbidden)
        predictor = (root / "predict_submission.py").read_text(encoding="utf-8")
        assert "03_model" in predictor
        assert "frozen_candidate_csv" not in predictor


def test_p3_recreates_historical_csv_numeric_boundaries() -> None:
    source = (
        ROOT
        / "scripts"
        / "final_submission_20260905"
        / "P3"
        / "predict_submission.py"
    ).read_text(encoding="utf-8")
    assert source.count("csv_float_roundtrip(") >= 3
    assert "two_saved_catboost_router_chains_then_frozen_affine_combination" in source


def test_local_package_structure_and_receipts_when_present() -> None:
    package = ROOT / "artifacts" / "official_final_submission_20260905"
    if not package.is_dir():
        return
    master = json.loads((package / "MASTER_MANIFEST.json").read_text(encoding="utf-8"))
    assert master["status"] == "LOCAL_READY_MODEL_INFERENCE_NOT_UPLOADED"
    assert master["atomic_problem_directories"] is True
    assert master["training_notebooks_present"] is True
    assert master["prediction_notebooks_executed"] is True
    assert master["historical_champion_hash_exact"] == {
        "P1": True,
        "P2": False,
        "P3": True,
    }
    required = {
        "01_data",
        "02_train",
        "03_model",
        "04_predict",
        "05_answer",
        "06_submission",
        "07_source",
    }
    for problem in PROBLEMS:
        problem_root = package / problem
        assert required.issubset({path.name for path in problem_root.iterdir() if path.is_dir()})
        receipt = master["receipts"][problem]
        assert receipt["status"] == "READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED"
        assert receipt["candidate_hash_exact"] is True
        assert receipt["package_atomic"] is True
        answer = problem_root / "05_answer" / f"{problem}_submission.csv"
        assert answer.is_file()
        form = json.loads((problem_root / "06_submission/FORM.json").read_text("utf-8"))
        assert form["data_directory"] == "01_data/organizer_dataset"
        assert form["training_entrypoint"] == "02_train/train_model.py"
        assert form["training_notebook"] == "02_train/TRAIN.ipynb"
        assert form["model_directory"] == "03_model"
        assert form["model_manifest"] == "03_model/MODEL_MANIFEST.json"
        assert form["prediction_entrypoint"] == "04_predict/predict_submission.py"
        assert form["prediction_notebook"] == "04_predict/PREDICT.ipynb"
        assert form["answer_file"] == f"05_answer/{problem}_submission.csv"
        assert form["answer_sha256"] == receipt["sha256"]
        assert form["rows"] == receipt["rows"]
        assert (problem_root / "RUN_TRAINING.ps1").is_file()
        assert (problem_root / "RUN_INFERENCE.ps1").is_file()
        assert (problem_root / "02_train/TRAINING_LINEAGE.md").is_file()
        assert not any(path.name == "__pycache__" for path in problem_root.rglob("__pycache__"))
    p3_source = package / "P3/07_source/src/p3_wave"
    assert not list(p3_source.glob("kma_*.py"))
    assert not list(p3_source.glob("era5_*.py"))
    assert not (p3_source / "chronos2_transfer.py").exists()
    assert not list(p3_source.glob("champion_matched_era5_*.py"))
