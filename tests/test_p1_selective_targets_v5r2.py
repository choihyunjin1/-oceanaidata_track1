from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p1_qc.data import BASE_COLUMNS, TRAIN_COLUMNS
from p1_qc.selective_targets_v5r2 import (
    SelectiveTargetAccessor,
    csv_field_spans,
    decode_csv_field,
    load_frozen_oof_keys_only,
    load_input_only_train,
)


class _Commitment:
    def __init__(self) -> None:
        self.folds: set[str] = set()
        self.global_commitment = False

    def is_fold_committed(self, fold: str) -> bool:
        return fold in self.folds

    def is_global_committed(self) -> bool:
        return self.global_commitment


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_bytes(*, poison_label: bool = False, poison_anomaly: bool = False) -> bytes:
    rows = [",".join(TRAIN_COLUMNS).encode("utf-8") + b"\n"]
    for index in range(4):
        base = [
            f"S{index}",
            "2025",
            "1",
            f"2025-01-0{index + 1}T00:00:00+09:00",
            str(10.0 + index),
            "34.0",
            "5.0",
        ]
        label = b"\xff" if poison_label and index == 1 else str(index % 2).encode("ascii")
        anomaly = b"\xff" if poison_anomaly and index == 1 else b"normal"
        rows.append(
            ",".join(base).encode("utf-8") + b"," + label + b"," + anomaly + b"\n"
        )
    return b"".join(rows)


def _accessor(path: Path) -> SelectiveTargetAccessor:
    return SelectiveTargetAccessor(
        path,
        expected_sha256=_sha(path),
        expected_rows=4,
        validation_rows_by_fold={
            "2025_q2": np.asarray([1], dtype=np.int64),
            "2025_q3": np.asarray([2], dtype=np.int64),
            "2025_q4": np.asarray([3], dtype=np.int64),
        },
        fold_order=("2025_q2", "2025_q3", "2025_q4"),
    )


def test_csv_span_parser_decodes_only_selected_quoted_field() -> None:
    line, spans = csv_field_spans(b'a,"b,b",c\n', expected_fields=3)
    assert decode_csv_field(line, spans[0]) == "a"
    assert decode_csv_field(line, spans[1]) == "b,b"
    assert decode_csv_field(line, spans[2]) == "c"


def test_opaque_index_does_not_decode_poisoned_active_validation_label(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.csv"
    path.write_bytes(_csv_bytes(poison_label=True))
    accessor = _accessor(path)
    commitment = _Commitment()
    assert accessor.decoded_target_scalars == 0
    assert accessor.labels_for(
        np.asarray([0]),
        commitment=commitment,
        purpose="q2_train",
        active_fold="2025_q2",
    ).tolist() == [0]
    before = accessor.decoded_target_scalars
    with pytest.raises(PermissionError, match="active outer-validation"):
        accessor.labels_for(
            np.asarray([1]),
            commitment=commitment,
            purpose="q2_validation_forbidden",
            active_fold="2025_q2",
        )
    assert accessor.decoded_target_scalars == before
    commitment.folds.add("2025_q2")
    with pytest.raises(UnicodeDecodeError):
        accessor.labels_for(
            np.asarray([1]),
            commitment=commitment,
            purpose="q3_rolling_training",
            active_fold="2025_q3",
        )


def test_rolling_origin_reuse_requires_all_prior_fold_commitments(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_bytes(_csv_bytes())
    accessor = _accessor(path)
    commitment = _Commitment()
    with pytest.raises(PermissionError, match="5/5 commitments"):
        accessor.labels_for(
            np.asarray([1]),
            commitment=commitment,
            purpose="q3_rolling_training",
            active_fold="2025_q3",
        )
    commitment.folds.add("2025_q2")
    assert accessor.labels_for(
        np.asarray([1]),
        commitment=commitment,
        purpose="q3_rolling_training",
        active_fold="2025_q3",
    ).tolist() == [1]
    assert accessor.validation_target_decode_counts("2025_q2") == {
        "label": 1,
        "anomaly_type": 0,
    }
    assert accessor.validation_target_decode_counts("2025_q3") == {
        "label": 0,
        "anomaly_type": 0,
    }
    with pytest.raises(PermissionError, match="active outer-validation"):
        accessor.labels_for(
            np.asarray([2]),
            commitment=commitment,
            purpose="q3_validation_forbidden",
            active_fold="2025_q3",
        )


def test_anomaly_type_is_global_commitment_only_and_poison_proves_nondecode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.csv"
    path.write_bytes(_csv_bytes(poison_anomaly=True))
    accessor = _accessor(path)
    commitment = _Commitment()
    with pytest.raises(PermissionError, match="global predictions commitment"):
        accessor.anomaly_types_for(
            np.asarray([1]), commitment=commitment, purpose="forbidden_anomaly"
        )
    assert accessor.decoded_anomaly_rows == accessor.decoded_target_scalars == 0
    commitment.global_commitment = True
    with pytest.raises(UnicodeDecodeError):
        accessor.anomaly_types_for(
            np.asarray([1]), commitment=commitment, purpose="postcommit_poison_probe"
        )


def test_input_only_loader_never_decodes_poisoned_target_bytes(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_bytes(_csv_bytes(poison_label=True, poison_anomaly=True))
    result = load_input_only_train(path)
    assert list(result.columns) == list(BASE_COLUMNS)
    assert result.shape == (4, len(BASE_COLUMNS))
    assert result.loc[1, "station"] == "S1"
    assert result.loc[1, "temp"] == 11.0
    assert result.attrs["input_fields_decoded"] == 4 * len(BASE_COLUMNS)
    assert result.attrs["target_fields_decoded"] == 0


def test_frozen_oof_loader_projects_without_target_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ["station", "year", "layer", "time", "fold", "prediction"]
    observed: dict[str, object] = {}

    def fake_read_parquet(path: Path, **kwargs: object) -> pd.DataFrame:
        observed.update(kwargs)
        return pd.DataFrame({column: [] for column in expected})

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    result = load_frozen_oof_keys_only(Path("ignored.parquet"))
    assert list(result.columns) == expected
    assert observed["columns"] == expected
    assert "label" not in observed["columns"]
    assert "anomaly_type" not in observed["columns"]
