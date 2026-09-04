from __future__ import annotations

import pandas as pd
import pytest

from p1_qc.submission import (
    SubmissionValidationError,
    build_submission,
    canonical_type,
    validate_submission,
    write_submission,
)
from scripts.validate_submission import P1_REQUIRED_FILES, default_test_path


def test_canonical_type_deduplicates_and_orders() -> None:
    assert canonical_type("drift+spike+drift") == "spike+drift"
    with pytest.raises(SubmissionValidationError):
        canonical_type("unknown")


def test_submission_round_trip(tmp_path) -> None:
    test = pd.DataFrame(
        {
            "station": ["S-ORS", "S-ORS"],
            "year": [2026, 2026],
            "layer": [1, 1],
            "time": ["2026-01-01T00:00:00+09:00", "2026-01-01T00:10:00+09:00"],
            "temp": [10.0, 11.0],
            "psal": [30.0, 30.1],
            "depth": [5.0, 5.0],
        }
    )
    submission = build_submission(test, [0, 1], ["", "drift+spike"])
    path = write_submission(submission, tmp_path / "candidate.csv")
    report = validate_submission(path, test)
    assert report["positive"] == 1
    assert submission.loc[1, "anomaly_type"] == "spike+drift"


def test_submission_rejects_reordered_keys() -> None:
    test = pd.DataFrame(
        {
            "station": ["S-ORS", "I-ORS"],
            "year": [2026, 2026],
            "layer": [1, 1],
            "time": ["a", "b"],
        }
    )
    submission = test.iloc[::-1].copy()
    submission["label"] = [0, 1]
    with pytest.raises(SubmissionValidationError, match="order"):
        validate_submission(submission, test)


def _write_required_p1_files(path) -> None:
    path.mkdir(parents=True)
    for name in P1_REQUIRED_FILES:
        (path / name).write_text("placeholder\n", encoding="utf-8")


def test_validator_resolves_p1_data_dir(tmp_path) -> None:
    data_dir = tmp_path / "input"
    _write_required_p1_files(data_dir)
    assert (
        default_test_path(env={"P1_DATA_DIR": str(data_dir)}, search_root=tmp_path)
        == (data_dir / "test.csv").resolve()
    )


def test_validator_fallback_rejects_ambiguous_file_sets(tmp_path) -> None:
    _write_required_p1_files(tmp_path / "first")
    _write_required_p1_files(tmp_path / "second")
    with pytest.raises(FileNotFoundError, match="found 2"):
        default_test_path(env={}, search_root=tmp_path)
