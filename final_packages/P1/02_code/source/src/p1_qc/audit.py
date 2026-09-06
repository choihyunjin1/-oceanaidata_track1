"""Read-only structural and semantic audits for P1 frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .data import ANOMALY_TYPES, BASE_COLUMNS, KEY_COLUMNS, TRAIN_COLUMNS, parse_anomaly_types


@dataclass(frozen=True)
class AuditIssue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    count: int = 0
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditReport:
    kind: str
    rows: int
    columns: tuple[str, ...]
    issues: tuple[AuditIssue, ...] = ()
    stats: Mapping[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            detail = "; ".join(f"{issue.code}: {issue.message}" for issue in self.errors)
            raise ValueError(f"{self.kind} audit failed: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rows": self.rows,
            "columns": list(self.columns),
            "ok": self.ok,
            "issues": [asdict(issue) for issue in self.issues],
            "stats": dict(self.stats),
        }


@dataclass(frozen=True)
class TrainTestAudit:
    train: AuditReport
    test: AuditReport
    issues: tuple[AuditIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return (
            self.train.ok
            and self.test.ok
            and not any(issue.severity == "error" for issue in self.issues)
        )

    def raise_for_errors(self) -> None:
        self.train.raise_for_errors()
        self.test.raise_for_errors()
        errors = [issue for issue in self.issues if issue.severity == "error"]
        if errors:
            raise ValueError("train/test audit failed: " + "; ".join(x.message for x in errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "train": self.train.to_dict(),
            "test": self.test.to_dict(),
            "issues": [asdict(issue) for issue in self.issues],
        }


def _examples(values: pd.Series, limit: int = 3) -> tuple[str, ...]:
    return tuple(str(value) for value in values.drop_duplicates().head(limit).tolist())


def _expected_columns(kind: str) -> tuple[str, ...]:
    if kind == "train":
        return TRAIN_COLUMNS
    if kind == "test":
        return BASE_COLUMNS
    if kind in {"submission", "baseline"}:
        return KEY_COLUMNS + ("label",)
    raise ValueError(f"unsupported audit kind: {kind}")


def audit_frame(
    frame: pd.DataFrame,
    *,
    kind: Literal["train", "test", "submission", "baseline", "auto"] = "auto",
    cadence_minutes: int = 10,
    known_anomaly_types: Sequence[str] = ANOMALY_TYPES,
) -> AuditReport:
    """Audit a frame without changing it.

    Exact source row counts are recorded but not enforced, allowing this
    function to validate small fixtures and legitimate scoped subsets.
    """

    if kind == "auto":
        if "temp" in frame and "label" in frame:
            kind = "train"
        elif "temp" in frame:
            kind = "test"
        else:
            kind = "submission"
    expected = _expected_columns(kind)
    issues: list[AuditIssue] = []
    stats: dict[str, Any] = {"missing_by_column": frame.isna().sum().astype(int).to_dict()}

    missing_columns = [column for column in expected if column not in frame]
    if kind in {"submission", "baseline"} and "anomaly_type" in frame:
        accepted_order = [*expected, "anomaly_type"]
    else:
        accepted_order = list(expected)
    if missing_columns:
        issues.append(
            AuditIssue(
                "error",
                "missing_columns",
                f"required columns are missing: {missing_columns}",
                len(missing_columns),
            )
        )
    elif list(frame.columns) != accepted_order:
        extra = [column for column in frame.columns if column not in accepted_order]
        if extra:
            issues.append(
                AuditIssue(
                    "warning",
                    "extra_columns",
                    f"unexpected columns are present: {extra}",
                    len(extra),
                )
            )
        if list(frame.columns[: len(accepted_order)]) != accepted_order:
            issues.append(
                AuditIssue(
                    "warning",
                    "column_order",
                    f"source column order differs from {accepted_order}",
                )
            )

    if all(column in frame for column in KEY_COLUMNS):
        key_missing = frame[list(KEY_COLUMNS)].isna().any(axis=1)
        if key_missing.any():
            issues.append(
                AuditIssue(
                    "error",
                    "missing_keys",
                    "key columns contain missing values",
                    int(key_missing.sum()),
                )
            )
        duplicate = frame.duplicated(list(KEY_COLUMNS), keep=False)
        if duplicate.any():
            examples = frame.loc[duplicate, list(KEY_COLUMNS)].astype(str).agg("|".join, axis=1)
            issues.append(
                AuditIssue(
                    "error",
                    "duplicate_keys",
                    "key columns are not unique",
                    int(duplicate.sum()),
                    _examples(examples),
                )
            )

    parsed_time: pd.Series | None = None
    if "time" in frame:
        parsed_time = pd.to_datetime(frame["time"], errors="coerce", utc=True)
        failed = parsed_time.isna()
        stats["time_parse_failures"] = int(failed.sum())
        if failed.any():
            issues.append(
                AuditIssue(
                    "error",
                    "invalid_time",
                    "timestamps could not be parsed",
                    int(failed.sum()),
                    _examples(frame.loc[failed, "time"].astype(str)),
                )
            )
        offset_ok = frame["time"].astype("string").str.endswith("+09:00", na=False)
        stats["kst_offset_rows"] = int(offset_ok.sum())
        if not offset_ok.all():
            issues.append(
                AuditIssue(
                    "warning",
                    "timezone_suffix",
                    "timestamps do not all end with the official +09:00 offset",
                    int((~offset_ok).sum()),
                )
            )
        if "year" in frame and not failed.all():
            local_year = parsed_time.dt.tz_convert("Asia/Seoul").dt.year
            given_year = pd.to_numeric(frame["year"], errors="coerce")
            mismatch = local_year.notna() & given_year.notna() & local_year.ne(given_year)
            if mismatch.any():
                issues.append(
                    AuditIssue(
                        "error",
                        "year_mismatch",
                        "year is not the KST calendar year of time",
                        int(mismatch.sum()),
                    )
                )

    for column in ("temp", "psal", "depth"):
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        nonfinite = frame[column].notna() & ~np.isfinite(
            numeric.to_numpy(dtype=float, na_value=np.nan)
        )
        if nonfinite.any():
            issues.append(
                AuditIssue(
                    "error",
                    f"nonfinite_{column}",
                    f"{column} contains non-finite numeric values",
                    int(nonfinite.sum()),
                )
            )
    if "temp" in frame and frame["temp"].isna().any():
        issues.append(
            AuditIssue(
                "error",
                "missing_temp",
                "the required target signal temp contains missing values",
                int(frame["temp"].isna().sum()),
            )
        )

    if "label" in frame:
        label = pd.to_numeric(frame["label"], errors="coerce")
        invalid = label.isna() | ~label.isin([0, 1])
        if invalid.any():
            issues.append(
                AuditIssue(
                    "error",
                    "invalid_label",
                    "label must contain only finite integer 0/1 values",
                    int(invalid.sum()),
                    _examples(frame.loc[invalid, "label"].astype(str)),
                )
            )
        else:
            stats["positive_rows"] = int(label.sum())
            stats["positive_rate"] = float(label.mean()) if len(label) else 0.0

    if kind == "train" and "anomaly_type" in frame and "label" in frame:
        label = pd.to_numeric(frame["label"], errors="coerce")
        text = frame["anomaly_type"].astype("string").fillna("").str.strip()
        missing_positive_type = label.eq(1) & text.eq("")
        normal_with_type = label.eq(0) & text.ne("")
        if missing_positive_type.any():
            issues.append(
                AuditIssue(
                    "error",
                    "missing_positive_type",
                    "positive rows must have anomaly_type",
                    int(missing_positive_type.sum()),
                )
            )
        if normal_with_type.any():
            issues.append(
                AuditIssue(
                    "error",
                    "normal_with_type",
                    "normal rows must not have anomaly_type",
                    int(normal_with_type.sum()),
                )
            )
        try:
            membership = parse_anomaly_types(
                frame["anomaly_type"], known_types=known_anomaly_types, strict=True
            )
            stats["anomaly_membership_rows"] = membership.sum().astype(int).to_dict()
        except ValueError as exc:
            issues.append(AuditIssue("error", "unknown_anomaly_type", str(exc)))

    if parsed_time is not None and all(column in frame for column in ("station", "layer")):
        cadence_frame = pd.DataFrame(
            {
                "station": frame["station"].to_numpy(),
                "layer": frame["layer"].to_numpy(),
                "time": parsed_time.to_numpy(),
            }
        ).sort_values(["station", "layer", "time"], kind="mergesort")
        delta = cadence_frame.groupby(["station", "layer"], observed=True)["time"].diff()
        delta_minutes = delta.dt.total_seconds().div(60.0)
        observed_delta = delta_minutes.notna()
        stats["cadence_deltas"] = int(observed_delta.sum())
        stats["exact_cadence_deltas"] = int(delta_minutes.eq(cadence_minutes).sum())
        stats["non_cadence_deltas"] = int(
            (observed_delta & delta_minutes.ne(cadence_minutes)).sum()
        )
        backwards = observed_delta & delta_minutes.le(0)
        if backwards.any():
            issues.append(
                AuditIssue(
                    "error",
                    "non_increasing_time",
                    "time is not strictly increasing within station/layer",
                    int(backwards.sum()),
                )
            )

    return AuditReport(
        kind=str(kind),
        rows=len(frame),
        columns=tuple(str(column) for column in frame.columns),
        issues=tuple(issues),
        stats=stats,
    )


def audit_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    cadence_minutes: int = 10,
) -> TrainTestAudit:
    train_report = audit_frame(train, kind="train", cadence_minutes=cadence_minutes)
    test_report = audit_frame(test, kind="test", cadence_minutes=cadence_minutes)
    issues: list[AuditIssue] = []
    if all(column in train for column in KEY_COLUMNS) and all(
        column in test for column in KEY_COLUMNS
    ):
        train_keys = pd.MultiIndex.from_frame(train[list(KEY_COLUMNS)])
        test_keys = pd.MultiIndex.from_frame(test[list(KEY_COLUMNS)])
        overlap = train_keys.intersection(test_keys)
        if len(overlap):
            issues.append(
                AuditIssue(
                    "error",
                    "train_test_key_overlap",
                    "train and test keys overlap",
                    len(overlap),
                    tuple("|".join(map(str, value)) for value in overlap[:3]),
                )
            )
        train_groups = set(zip(train["station"], train["layer"], strict=False))
        test_groups = set(zip(test["station"], test["layer"], strict=False))
        unseen = sorted(test_groups.difference(train_groups))
        if unseen:
            issues.append(
                AuditIssue(
                    "warning",
                    "unseen_test_groups",
                    f"test contains station/layer groups absent from train: {unseen}",
                    len(unseen),
                )
            )
    return TrainTestAudit(train_report, test_report, tuple(issues))


__all__ = [
    "AuditIssue",
    "AuditReport",
    "TrainTestAudit",
    "audit_frame",
    "audit_train_test",
]
