from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p2_copula_support_audit_20260829_v1.py"
SPEC = importlib.util.spec_from_file_location("p2_copula_support_audit", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_rank_gaussian_is_finite_monotone_and_tie_preserving() -> None:
    values = np.asarray([1.0, 2.0, 2.0, 4.0])
    transformed = RUNNER._rank_gaussian(values)
    assert np.isfinite(transformed).all()
    assert transformed[0] < transformed[1] == transformed[2] < transformed[3]


def test_config_forbids_every_official_interface_file() -> None:
    config = RUNNER.json.loads(RUNNER.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["allowed_source_file"] == "observations.csv"
    assert set(config["forbidden_files"]) == {
        "test_index.csv",
        "sample_submission.csv",
        "baseline_interp.csv",
        "score.py",
    }
    assert config["model_fit_count"] == 0
