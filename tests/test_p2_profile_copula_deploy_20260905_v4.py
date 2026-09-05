"""Synthetic deployment boundary checks; no real model/data/official inputs."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_copula_deploy", ROOT / "scripts/run_p2_profile_copula_deploy_20260905_v4.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_fullfit_no_official_or_old_predictions():
    source = (ROOT / "synthetic_source/observations.csv").resolve()
    assert M.allowed_path(source, "RUN_TRAINING", False, source)
    assert not M.allowed_path(source, "RUN_TRAINING", True, source)
    for path in [source.parent / "test_index.csv", source.parent / "sample_submission.csv", ROOT / "old/answer.csv", ROOT / "old/raw_oof.npz", ROOT / "old/model.pt"]:
        assert not M.allowed_path(path, "RUN_TRAINING", False, source)
    assert M.allowed_path(M.REGEN / "03_model/model_seed.pt", "RUN_TRAINING", False, source)
    assert not M.allowed_path(M.REGEN / "03_model/model_seed.pt", "RUN_TRAINING", True, source)


def test_new_model_and_answer_file_boundaries():
    source = (ROOT / "synthetic_source/observations.csv").resolve()
    assert M.allowed_path(M.MODELS / "copula_full.npz", "RUN_TRAINING", True, source)
    assert not M.allowed_path(M.ANSWERS / "answer.csv", "RUN_TRAINING", True, source)
    assert M.allowed_path(M.ANSWERS / "answer.csv", "INFERENCE", True, source)


def test_policy_exact_full_no_posthoc_shrink_or_routing():
    cfg = M.config()
    assert cfg["correction_strength"] == 1.0 and cfg["new_full_copula_fits"] == 1
    assert cfg["new_backbone_fits"] == 0 and cfg["sample_value_rows_read"] == 0
    raw = np.array([1., 3., -4.])
    residual = np.array([.1, -.2, 2.])
    np.testing.assert_array_equal(raw + residual, [1.1, 2.8, -2.])


def test_training_uses_released_labels_regenerated_C_not_OOF():
    import inspect
    source = inspect.getsource(M.train)
    assert "previous.load_data" in source and "truth - control" in source
    assert "OLD_RAW" not in source and "sample_submission" not in source
    assert "profile.fit_copula" in source
