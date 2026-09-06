"""Build and strictly validate P1 submission files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p1_qc.experiment import sha256_file

KEY_COLUMNS = ["station", "year", "layer", "time"]
REQUIRED_COLUMNS = KEY_COLUMNS + ["label"]
ANOMALY_TYPES = ("spike", "noise", "flatline", "offset", "drift")
TYPE_ORDER = {name: index for index, name in enumerate(ANOMALY_TYPES)}


class SubmissionValidationError(ValueError):
    pass


def canonical_type(value: str | float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    parts = [part.strip() for part in str(value).split("+") if part.strip()]
    unknown = sorted(set(parts).difference(ANOMALY_TYPES))
    if unknown:
        raise SubmissionValidationError(f"unknown anomaly type(s): {unknown}")
    return "+".join(sorted(set(parts), key=TYPE_ORDER.__getitem__))


def build_submission(
    test: pd.DataFrame,
    labels: Iterable[int],
    anomaly_types: Iterable[str] | None = None,
) -> pd.DataFrame:
    result = test.loc[:, KEY_COLUMNS].copy()
    label_array = np.asarray(list(labels))
    if len(label_array) != len(result):
        raise SubmissionValidationError(
            f"label length {len(label_array):,} does not match test rows {len(result):,}"
        )
    if not np.isin(label_array, [0, 1]).all():
        raise SubmissionValidationError("labels must be binary 0/1")
    result["label"] = label_array.astype("int8")
    if anomaly_types is not None:
        raw_types = list(anomaly_types)
        if len(raw_types) != len(result):
            raise SubmissionValidationError("anomaly_type length does not match test")
        canonical = [canonical_type(value) for value in raw_types]
        result["anomaly_type"] = [
            value if label else "" for value, label in zip(canonical, label_array, strict=True)
        ]
    return result


def write_submission(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def validate_submission(
    submission: str | Path | pd.DataFrame,
    test: str | Path | pd.DataFrame,
) -> dict[str, Any]:
    submission_path = Path(submission) if not isinstance(submission, pd.DataFrame) else None
    test_path = Path(test) if not isinstance(test, pd.DataFrame) else None
    if submission_path is not None:
        try:
            raw = submission_path.read_bytes()
            raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SubmissionValidationError("submission is not valid UTF-8") from exc
        submission_frame = pd.read_csv(submission_path, keep_default_na=False)
    else:
        submission_frame = submission.copy()
    test_frame = pd.read_csv(test_path) if test_path is not None else test.copy()

    expected = [REQUIRED_COLUMNS, REQUIRED_COLUMNS + ["anomaly_type"]]
    if list(submission_frame.columns) not in expected:
        raise SubmissionValidationError(
            f"invalid columns/order {list(submission_frame.columns)}; expected one of {expected}"
        )
    if len(submission_frame) != len(test_frame):
        raise SubmissionValidationError(
            f"row count {len(submission_frame):,} does not match test {len(test_frame):,}"
        )
    if submission_frame[KEY_COLUMNS].isna().any().any():
        raise SubmissionValidationError("keys contain missing values")
    if submission_frame.duplicated(KEY_COLUMNS).any():
        raise SubmissionValidationError("submission keys are not unique")
    if test_frame.duplicated(KEY_COLUMNS).any():
        raise SubmissionValidationError("test keys are not unique")
    if not submission_frame[KEY_COLUMNS].equals(test_frame[KEY_COLUMNS]):
        merged = test_frame[KEY_COLUMNS].merge(
            submission_frame[KEY_COLUMNS], how="outer", on=KEY_COLUMNS, indicator=True
        )
        if merged["_merge"].ne("both").any():
            raise SubmissionValidationError("submission key set differs from test")
        raise SubmissionValidationError("submission key order differs from test")

    labels = pd.to_numeric(submission_frame["label"], errors="coerce")
    if labels.isna().any() or not np.isfinite(labels.to_numpy(dtype=float)).all():
        raise SubmissionValidationError("labels contain missing or non-finite values")
    if not labels.isin([0, 1]).all() or not np.equal(labels, labels.astype(int)).all():
        raise SubmissionValidationError("labels must be integer 0/1")

    if "anomaly_type" in submission_frame:
        canonical = submission_frame["anomaly_type"].map(canonical_type)
        noncanonical = canonical.ne(submission_frame["anomaly_type"].astype(str))
        if noncanonical.any():
            example = submission_frame.loc[noncanonical, "anomaly_type"].iloc[0]
            raise SubmissionValidationError(f"anomaly_type is not canonical: {example!r}")
        if canonical[labels.eq(0)].ne("").any():
            raise SubmissionValidationError("normal rows must have blank anomaly_type")

    report: dict[str, Any] = {
        "rows": len(submission_frame),
        "positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "test_order_match": True,
        "columns": list(submission_frame.columns),
    }
    if submission_path is not None:
        report.update(
            {
                "path": str(submission_path.resolve()),
                "bytes": submission_path.stat().st_size,
                "sha256": sha256_file(submission_path),
            }
        )
    return report
