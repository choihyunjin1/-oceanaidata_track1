from __future__ import annotations

import inspect
from pathlib import Path

import scripts.resume_p3_corrected_fixed_long_shrink_v5 as resume


def test_resume_protocol_and_preserved_model_hashes_are_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    assert resume.v4.sha256_file(root / resume.PROTOCOL_RELATIVE) == resume.PROTOCOL_SHA256
    for name, path in resume._model_paths(root).items():
        assert resume.v4.sha256_file(path) == resume.MODEL_SHA256[name]


def test_resume_path_has_no_fit_call() -> None:
    source = inspect.getsource(resume.resume_experiment)
    assert ".fit(" not in source
    assert "_fit_full_models" not in source
    assert source.count("_infer_fresh_models") == 2


def test_resume_api_has_no_config_output_or_model_override() -> None:
    assert set(inspect.signature(resume.resume_experiment).parameters) == {
        "root",
        "data_dir",
    }
