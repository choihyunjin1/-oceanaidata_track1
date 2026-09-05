"""Regression tests for self-artifact checksum permission, not model loading."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_clean_v5", ROOT / "scripts/run_p2_clean_regeneration_20260905_v5.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_current_attempt_new_artifact_hash_allowed_after_write_only():
    source = ROOT / "synthetic_source/observations.csv"
    path = M.original.MODELS / "test_new_seed.pt"
    M.CREATED_MODELS.clear()
    assert not M.path_allowed(path, "RUN_TRAINING", False, source)
    assert M.path_allowed(path, "RUN_TRAINING", True, source)
    assert M.path_allowed(path, "RUN_TRAINING", False, source)
    assert not M.path_allowed(M.original.MODELS / "uncreated_seed.pt", "RUN_TRAINING", False, source)


def test_previous_attempt_and_old_weights_reads_remain_forbidden():
    source = ROOT / "synthetic_source/observations.csv"
    for path in [ROOT / "artifacts/p2_clean_regeneration_20260905_v4/03_model/model_seed20260901.pt", ROOT / "artifacts/p2_score_repair_deploy_20260905_v1/model_seed20260901.pt"]:
        assert not M.path_allowed(path, "RUN_TRAINING", False, source)
        assert not M.path_allowed(path, "RUN_TRAINING", True, source)


def test_torch_load_prohibited_even_self_generated_model():
    with pytest.raises(PermissionError, match="torch.load"):
        M.deny_training_model_load(M.original.MODELS / "test_new_seed.pt")


def test_official_rows_training_denied_but_inference_keys_path_allowed():
    source = (ROOT / "synthetic_source/observations.csv").resolve()
    for filename in ["test_index.csv", "sample_submission.csv"]:
        assert not M.path_allowed(source.parent / filename, "RUN_TRAINING", False, source)
        assert M.path_allowed(source.parent / filename, "RUN_INFERENCE", False, source)
    assert not M.path_allowed(ROOT / "old/answer.csv", "RUN_INFERENCE", False, source)


def test_frozen_training_functions_have_no_answer_or_model_load():
    assert M.original.lineage_source_contract()
