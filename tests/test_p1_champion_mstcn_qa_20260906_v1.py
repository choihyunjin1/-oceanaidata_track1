"""QA-tool synthetic fixtures only. No candidate checkpoint or organizer IO."""
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/p1_champion_reconstruction_20260906_v1/qa_mstcn.py"
SPEC = importlib.util.spec_from_file_location("independent_mstcn_qa", PATH)
QA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QA)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def save_synthetic_state(path, seed):
    # One scalar-backed expanded tensor per expected shape. Real canonical shape
    # metadata, but no candidate weights, 210MB allocation, network or training.
    storage = torch.zeros((), dtype=torch.float32)
    tensors = {name: storage.expand(shape) for name, shape in QA.expected_tensor_shapes().items()}
    torch.save({"seed": seed, "epoch": 150, "input_features": 165, "state_dict": tensors}, path)


@pytest.fixture
def bundle(tmp_path):
    output = tmp_path / "synthetic_only"
    output.mkdir()
    for relative in QA.SOURCE_PINS:
        target = output / "02_code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copyfile(ROOT / "scripts/p1_champion_reconstruction_20260906_v1/mstcn.py",
                    output / "02_code/mstcn.py")
    (output / "03_model").mkdir()
    validation = output / "04_validation"
    validation.mkdir()
    names = [f"synthetic_{i}" for i in range(74)]
    encoder = {"center": [0.] * 74, "scale": [1.] * 74,
               "station_vocab": ["G-ORS", "I-ORS", "S-ORS"],
               "layer_vocab": [str(i) for i in range(1, 9)],
               "depth_regime_vocab": ["deep", "mid", "missing", "shallow"],
               "numeric_names": names, "depth_thresholds": [1., 2.],
               "uses_supplied_depth_regime": False, "preprocessing_fit_uses_holdout_rows": False,
               "fit_ids_sha256": hashlib.sha256(np.arange(776706, dtype="<i8").tobytes()).hexdigest()}
    write(output / "03_model/encoder.json", encoder)
    values = np.zeros((2, 2048, 165), dtype=np.float32)
    valid = np.zeros((2, 2048), dtype=np.float32)
    valid[:, :16] = 1
    np.savez_compressed(validation / "own_train_probe.npz", values=values, valid=valid)
    fits = []
    for seed in QA.SEEDS:
        state_path = output / f"03_model/seed_{seed}.pt"
        save_synthetic_state(state_path, seed)
        fit = {"seed": seed, "epoch": 150, "optimizer_steps": 4050, "windows": 1707,
               "model_sha256": QA.sha(state_path), "wall_seconds": 1., "nonfinite_count": 0}
        fits.append(fit)
        write(validation / f"fit_{seed}.json", fit)
        np.savez_compressed(validation / f"probe_expected_{seed}.npz",
                            row=np.full((2, 2048), .5, np.float32),
                            boundary=np.full((2, 2048, 2), .5, np.float32),
                            type=np.full((2, 2048, 5), .5, np.float32))
    started = time.time() - 10
    training_pid, replay_pid = 11111111, 22222222
    runtime = {"pid": training_pid, "python": "synthetic", "torch": "synthetic",
               "cuda": "synthetic", "gpu": "no physical GPU used",
               "cpu_threads": 2, "precision": "bfloat16"}
    training = {"status": "TRAINING_COMPLETE_REPLAY_PENDING", "id": QA.ID,
                "started_at": started, "training_pid": training_pid, "contract": QA.EXPECTED,
                "completed_fits": 3, "fit_receipts": fits, "source_pins": QA.SOURCE_PINS,
                "files_sha256": {relative: QA.sha(output / relative) for relative in QA.expected_files()},
                "wall_seconds": 5., "runtime": runtime, "official_access_rows": 0,
                "old_models_cache_answer_reads": 0}
    write(output / "training-result.json", training)
    checks = {f"{seed}_{head}_exact": True for seed in QA.SEEDS for head in ("row", "boundary", "type")}
    checks.update({"fresh_pid": True, "within_original_6h": True})
    checks.update({f"runtime_{key}": True for key in runtime if key != "pid"})
    replay = {"status": "PASS", "checks": checks, "pid": replay_pid,
              "training_pid": training_pid, "training_result_sha256": QA.sha(output / "training-result.json"),
              "runtime": {**runtime, "pid": replay_pid}, "official_access_rows": 0,
              "scratch_retraining_determinism": "NOT_TESTED", "source_to_replay_seconds": 6.}
    write(output / "fresh-process-replay.json", replay)
    write(output / "terminal.json", replay)
    write(output / "ATTEMPT_LOCK.json", {"pid": training_pid, "started_at": started, "contract": QA.EXPECTED})
    write(output / "REPLAY_ATTEMPT_LOCK.json", {"pid": replay_pid})
    write(output / "prepare.json", {
        "train_sha256": QA.TRAIN_SHA, "rows": 776706, "input_shape": [776706, 165],
        "feature_sha256": "a" * 64, "ordered_key_sha256": "b" * 64, "numeric_names": names,
        "wall_seconds": 1., "old_cache_reads": 0, "official_access_rows": 0,
        "dependency": {"cache_feature_count": 80, "model_numeric_feature_count": 74,
                       "unbounded_features_projected": 0, "all_cache_features_classified": True,
                       "excluded_unbounded": ["nominal_depth_m", "plateau_full_length", "plateau_count", "depth_regime"],
                       "bounded_future_support_hours_max": 168}})
    return output


def relink(output, training):
    write(output / "training-result.json", training)
    replay = QA.read(output / "fresh-process-replay.json")
    replay["training_result_sha256"] = QA.sha(output / "training-result.json")
    write(output / "fresh-process-replay.json", replay)
    write(output / "terminal.json", replay)


def test_independent_architecture_arithmetic():
    shapes = QA.expected_tensor_shapes()
    assert len(shapes) == 200
    assert sum(np.prod(shape) for shape in shapes.values()) == 52568587
    assert shapes["prediction_generator.stem.weight"] == (512, 165, 1)


def test_full_synthetic_metadata_qa_pass(bundle):
    result = QA.verify(bundle)
    assert result["status"] == "PASS", result["failed_checks"]
    assert result["failed_check_count"] == 0 and result["check_count"] > 140
    assert result["fit_wall_seconds_sum"] == 3.
    assert result["fresh_replay_sha256"] == QA.sha(bundle / "fresh-process-replay.json")
    assert all(result[key] == 0 for key in ("new_model_fits", "new_forward_calls", "gpu_calls", "raw_csv_rows_read"))


@pytest.mark.parametrize("status", ["RUNNING", "TERMINAL_TECHNICAL_FAILURE"])
def test_not_complete_rejects_before_binary_reads(bundle, monkeypatch, status):
    training = QA.read(bundle / "training-result.json")
    training["status"] = status
    relink(bundle, training)
    monkeypatch.setattr(QA, "state_metadata", lambda *_: pytest.fail("early model read"))
    with pytest.raises(RuntimeError, match="completion"):
        QA.verify(bundle)


@pytest.mark.parametrize("change", ["status", "empty_checks", "same_pid", "wrong_link"])
def test_replay_gate_fail_closed(bundle, monkeypatch, change):
    replay = QA.read(bundle / "fresh-process-replay.json")
    if change == "status":
        replay["status"] = "FAIL"
    elif change == "empty_checks":
        replay["checks"] = {}
    elif change == "same_pid":
        replay["pid"] = replay["training_pid"]
    else:
        replay["training_result_sha256"] = "0" * 64
    write(bundle / "fresh-process-replay.json", replay)
    write(bundle / "terminal.json", replay)
    monkeypatch.setattr(QA, "state_metadata", lambda *_: pytest.fail("early model read"))
    with pytest.raises(RuntimeError):
        QA.verify(bundle)


def test_model_hash_changed_no_deserialization(bundle, monkeypatch):
    (bundle / f"03_model/seed_{QA.SEEDS[0]}.pt").write_bytes(b"synthetic corruption")
    monkeypatch.setattr(QA, "state_metadata", lambda *_: pytest.fail("unhashed model read"))
    with pytest.raises(ValueError, match="hashes failed"):
        QA.verify(bundle)


def test_extra_manifest_file_denied(bundle):
    training = QA.read(bundle / "training-result.json")
    training["files_sha256"]["../old.pt"] = "0" * 64
    relink(bundle, training)
    with pytest.raises(RuntimeError, match="manifest"):
        QA.verify(bundle)


@pytest.mark.parametrize("relative", ["../old.pt", "C:/old.pt", "C:\\old.pt", "/old.pt"])
def test_owned_paths_reject_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        QA.owned_path(tmp_path, relative)


def test_fit_arithmetic_and_zeros_independent(bundle):
    training = QA.read(bundle / "training-result.json")
    training["official_access_rows"] = 1
    training["wall_seconds"] = 2.
    relink(bundle, training)
    result = QA.verify(bundle)
    assert result["status"] == "FAIL"
    assert "training_official_access_rows_zero" in result["failed_checks"]
    assert "fit_sum_fits_inside_training_wall" in result["failed_checks"]


def test_six_hour_clock_not_reset(bundle, monkeypatch):
    training = QA.read(bundle / "training-result.json")
    monkeypatch.setattr(QA.time, "time", lambda: training["started_at"] + 21601)
    result = QA.verify(bundle)
    assert "independent_qa_inside_original_six_hours" in result["failed_checks"]


def test_state_seed_metadata_rejected(tmp_path):
    path = tmp_path / "synthetic.pt"
    save_synthetic_state(path, QA.SEEDS[1])
    with pytest.raises(ValueError, match="seed/epoch"):
        QA.state_metadata(path, QA.SEEDS[0])


def test_wrong_tensor_shape_rejected(tmp_path):
    path = tmp_path / "synthetic.pt"
    save_synthetic_state(path, QA.SEEDS[0])
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["state_dict"]["type_head.bias"] = torch.zeros(4)
    torch.save(state, path)
    with pytest.raises(ValueError, match="shape"):
        QA.state_metadata(path, QA.SEEDS[0])


def test_no_producer_import_or_forward():
    import ast

    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("mstcn" in name or "p1_qc" in name for name in imports)
    names = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not names.intersection({"fit", "forward", "predict_encoded", "predict_proposal", "read_csv"})
