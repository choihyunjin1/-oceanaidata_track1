from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "final_submission_20260905.json"
PROBLEMS = ("P1", "P2", "P3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_script(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_core_archives_contain_only_dependency_closed_source_modules() -> None:
    builder = _load_script(
        "scripts/build_official_final_submission_20260905.py", "final_builder_test"
    )
    assert set(builder.PACKAGED_SOURCE_MODULES) == {"P1", "P2"}
    assert builder.excluded_from_core(
        "P1", Path("07_source/src/p1_qc/external.py")
    )
    assert builder.excluded_from_core(
        "P2", Path("07_source/src/p2_restore/era5_full.py")
    )
    assert builder.excluded_from_core("P1", Path("upload_parts/large.part00"))
    assert not builder.excluded_from_core(
        "P1", Path("07_source/src/p1_qc/ms_tcn_asrf.py")
    )
    assert not builder.excluded_from_core(
        "P2", Path("07_source/src/p2_restore/features.py")
    )
    package = ROOT / "artifacts/official_final_submission_20260905"
    if not package.is_dir():
        return
    module_names = {"P1": "p1_qc", "P2": "p2_restore"}
    for problem, module_name in module_names.items():
        archive_path = package / "upload" / f"{problem}_official_final_core.zip"
        with zipfile.ZipFile(archive_path) as archive:
            assert not any(
                name.startswith(f"{problem}/upload_parts/")
                for name in archive.namelist()
            )
            prefix = f"{problem}/07_source/src/{module_name}/"
            actual = {
                Path(name).name
                for name in archive.namelist()
                if name.startswith(prefix) and name.endswith(".py")
            }
        assert actual == set(builder.PACKAGED_SOURCE_MODULES[problem])


def test_p1_split_reassembler_round_trip(tmp_path: Path) -> None:
    reassembler = _load_script(
        "scripts/reassemble_p1_upload_20260905.py", "p1_reassembler_test"
    )
    upload = tmp_path / "upload"
    package = tmp_path / "P1"
    upload.mkdir()
    source = b"model-bytes-without-observation-values"
    parts = [source[:13], source[13:]]
    part_records = []
    for index, payload in enumerate(parts):
        name = f"03_model__weights__demo.pt.part{index:02d}"
        archive_path = upload / f"P1_{name}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(name, payload)
        part_records.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "files": [
            {
                "source_relative_path": "03_model/weights/demo.pt",
                "source_bytes": len(source),
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "parts": part_records,
            }
        ]
    }
    (upload / "P1_REASSEMBLY_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    receipt = reassembler.reassemble(upload, package)
    assert receipt["status"] == "P1_REASSEMBLY_PASS"
    assert (package / "03_model/weights/demo.pt").read_bytes() == source
    with pytest.raises(reassembler.ReassemblyError):
        reassembler.safe_target(package, "../outside.bin")
    with pytest.raises(reassembler.ReassemblyError):
        reassembler.safe_archive(upload, "nested/part00")


def test_portal_contract_matches_forms_answers_and_upload_inventory() -> None:
    portal_path = ROOT / "configs/final_submission_portal_20260905.json"
    portal_text = portal_path.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in portal_text
    portal = json.loads(portal_text)
    assert portal["status"] == "LOCAL_GUIDE_ONLY_NOT_UPLOADED"
    assert portal["safety"]["organizer_distributed_data_only"] is True
    assert portal["safety"]["hidden_truth_access_allowed"] is False
    package = ROOT / "artifacts/official_final_submission_20260905"
    if not package.is_dir():
        return
    upload = package / "upload"
    master = json.loads((package / "MASTER_MANIFEST.json").read_text(encoding="utf-8"))
    actual_uploads = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in upload.iterdir()
        if path.is_file()
    }
    recorded_uploads = {
        item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]}
        for item in master["upload_files"]
    }
    assert actual_uploads == recorded_uploads
    assert len(actual_uploads) == 26
    p1_files = [
        path
        for path in upload.iterdir()
        if path.name in {"P1_official_final_core.zip", "P1_REASSEMBLY_MANIFEST.json"}
        or (path.name.startswith("P1_") and ".part" in path.name and path.suffix == ".zip")
    ]
    assert len(p1_files) == portal["final_model_upload"]["problems"]["P1"][
        "expected_file_count"
    ]
    assert len([path for path in p1_files if ".part" in path.name]) == 22
    assert max(path.stat().st_size for path in upload.iterdir() if path.is_file()) < 50_000_000
    forbidden_core_names = (
        "/01_data/organizer_dataset/",
        "external.py",
        "iors_external_point_residual.py",
        "external_meteorology.py",
        "era5_",
        "kma_",
        "chronos2_transfer.py",
        "champion_matched_era5_",
    )
    for problem in PROBLEMS:
        with zipfile.ZipFile(upload / f"{problem}_official_final_core.zip") as archive:
            lowered = [name.lower() for name in archive.namelist()]
        assert not any(
            token.lower() in name for name in lowered for token in forbidden_core_names
        )
    with zipfile.ZipFile(upload / "P1_official_final_core.zip") as archive:
        assert "P1/REASSEMBLE_UPLOAD.py" in archive.namelist()
        readme = archive.read("P1/README.md").decode("utf-8")
        assert "REASSEMBLE_UPLOAD.py" in readme

    answers = portal["answer_upload"]["problems"]
    expected_answer_sha = {
        "P1": answers["P1"]["sha256"],
        "P2": answers["P2"]["reproducibility_model_replay"]["sha256"],
        "P3": answers["P3"]["sha256"],
    }
    for problem in PROBLEMS:
        answer = package / problem / "05_answer" / f"{problem}_submission.csv"
        assert _sha256(answer) == expected_answer_sha[problem]
        form = json.loads(
            (package / problem / "06_submission/FORM.json").read_text(encoding="utf-8")
        )
        portal_form = portal["final_model_upload"]["problems"][problem]
        assert form["website_form"]["title"] == portal_form["title"]
        assert form["website_form"]["one_line_summary"] == portal_form[
            "one_line_summary"
        ]
        assert form["website_form"]["notes"] == portal_form["notes"]
