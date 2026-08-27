from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import ocean_reproduce as orchestration
from p2_restore.final_inference import csv_float_roundtrip as p2_roundtrip
from p3_wave.final_inference import csv_float_roundtrip as p3_roundtrip


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_csv_float_boundaries_are_deterministic() -> None:
    values = np.array([0.1 + 0.2, 1.7360191091338169, 30.07068456090405])
    assert np.array_equal(p2_roundtrip(values), p2_roundtrip(values))
    assert np.array_equal(p3_roundtrip(values), p3_roundtrip(values))


def test_stage_resume_requires_all_artifact_hashes(tmp_path: Path) -> None:
    output = tmp_path / "output"
    artifact = output / "saved_weight" / "P1_submission.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("safe", encoding="utf-8")
    orchestration._write_stage_receipt(output, "saved_P1", {"ok": True}, [artifact])
    assert orchestration._resume_stage(output, "saved_P1") == {"ok": True}
    artifact.write_text("changed", encoding="utf-8")
    assert orchestration._resume_stage(output, "saved_P1") is None


def test_raw_file_guard_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "ready").mkdir()
    (tmp_path / "ready" / "P1_submission.csv").write_text("submission", encoding="utf-8")
    orchestration._assert_no_raw_files(tmp_path)
    (tmp_path / "retrain" / "P3").mkdir(parents=True)
    (tmp_path / "retrain" / "P3" / "test_context.parquet").write_bytes(b"raw")
    with pytest.raises(RuntimeError, match="leaked"):
        orchestration._assert_no_raw_files(tmp_path)


def test_existing_output_refuses_without_resume(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError, match="--resume"):
        orchestration.run_all(
            project_root=tmp_path,
            p1_data_dir=tmp_path / "p1",
            p2_data_dir=tmp_path / "p2",
            p3_data_dir=tmp_path / "p3",
            output_dir=output,
            mode="both",
            resume=False,
        )


def test_retrain_only_orchestration_writes_portable_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dirs = {problem: tmp_path / problem.lower() for problem in ("P1", "P2", "P3")}
    required = {
        "P1": ("train.csv", "test.csv", "sample_submission.csv", "baseline_rule.csv"),
        "P2": (
            "observations.csv",
            "test_index.csv",
            "sample_submission.csv",
            "baseline_interp.csv",
        ),
        "P3": (
            "train_wave.csv",
            "train_atmos.csv",
            "test_context.parquet",
            "test_index.csv",
            "sample_submission.csv",
            "baseline_persistence.csv",
        ),
    }
    for problem, directory in data_dirs.items():
        directory.mkdir()
        for name in required[problem]:
            (directory / name).write_bytes(f"{problem}:{name}".encode())

    def fake_p1(_: Path, target: Path) -> dict[str, object]:
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.joblib").write_bytes(b"model")
        (target / "submission.csv").write_text("candidate", encoding="utf-8")
        return {"ok": True}

    def fake_other(_: Path, target: Path, update) -> dict[str, object]:
        update(0.5, "synthetic test stage")
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(b"model")
        (target / "submission.csv").write_text("candidate", encoding="utf-8")
        return {"ok": True}

    monkeypatch.setattr(orchestration, "_retrain_p1", fake_p1)
    monkeypatch.setattr(orchestration, "_retrain_p2", fake_other)
    monkeypatch.setattr(orchestration, "_retrain_p3", fake_other)
    original = Path.cwd()
    output = tmp_path / "portable"
    manifest = orchestration.run_all(
        project_root=tmp_path,
        p1_data_dir=data_dirs["P1"],
        p2_data_dir=data_dirs["P2"],
        p3_data_dir=data_dirs["P3"],
        output_dir=output,
        mode="retrain",
    )
    assert Path.cwd() == original
    assert manifest["status"] == "complete_not_uploaded"
    assert manifest["uploaded"] is False
    assert (output / "status.json").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "logs" / "run.log").is_file()
    assert _sha(output / "manifest.json")
