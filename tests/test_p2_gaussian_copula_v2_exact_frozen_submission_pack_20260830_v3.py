from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_restore import (
    p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1 as sealed,
)
from p2_restore import (
    p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v3 as adapter,
)
from p2_restore.p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1 import (
    load_json,
    sha256_file,
)
from p2_restore.submission import build_submission, validate_submission
from scripts import (
    run_p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v3 as runner,
)


def _synthetic_test_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": pd.Series(["S", "S"], dtype="string"),
            "layer": [2, 3],
            "time": pd.Series(["t0", "t0"], dtype="string"),
            "nominal_depth": [5.0, 25.0],
        }
    )


def test_exact_input_superset_is_preserved_and_output_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "test_index.csv"
    _synthetic_test_index().to_csv(path, index=False)
    test, receipt = adapter.read_test_index_superset(path)
    assert list(test.columns) == adapter.INPUT_COLUMNS
    assert receipt["nominal_depth_preserved"] is True
    assert receipt["nominal_depth_used_as_model_feature"] is False
    submission = build_submission(test, np.asarray([1.0, 2.0]))
    assert list(submission.columns) == adapter.OUTPUT_COLUMNS
    assert validate_submission(submission, test)["rows"] == 2


@pytest.mark.parametrize(
    "columns",
    [
        ["station", "layer", "time"],
        ["station", "layer", "nominal_depth", "time"],
        ["station", "layer", "time", "nominal_depth", "extra"],
    ],
)
def test_input_superset_rejects_missing_reordered_or_extra_columns(
    tmp_path: Path, columns: list[str]
) -> None:
    path = tmp_path / "test_index.csv"
    frame = _synthetic_test_index()
    if "extra" in columns:
        frame["extra"] = 0
    frame.loc[:, columns].to_csv(path, index=False)
    with pytest.raises(sealed.SubmissionPackError, match="schema mismatch"):
        adapter.read_test_index_superset(path)


def test_v3_runner_import_and_help_contract() -> None:
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


def test_v3_overlay_hash_pins_and_frozen_invariants() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    overlay = load_json(runner.DEFAULT_CONFIG)
    base_config = load_json(runner.DEFAULT_BASE_CONFIG)
    assert sha256_file(runner.DEFAULT_CONFIG) == runner.EXPECTED_OVERLAY_CONFIG_SHA256
    receipts = runner.validate_overlay(repo_root, overlay, base_config)
    assert receipts["query_adapter"]["sha256"] == (
        runner.EXPECTED_QUERY_ADAPTER_SHA256
    )
    frozen = base_config["frozen_recipe"]
    assert len(frozen["training_blocks"]) == 7
    assert frozen["selected_shrinkage"] == 0.5
    assert frozen["inner_search_or_hpo"] == 0
    assert base_config["execution_policy"] == overlay["execution_policy"]
