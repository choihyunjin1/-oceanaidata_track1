from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from p2_restore.joint_hydrographic_multitask_layer4_checkpoint_v1_recovery import (
    RECOVERY_FILES,
    key_alignment_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def _frame(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["fold", "station", "layer", "time"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame


def test_truth_strict_superset_is_audited_not_rejected() -> None:
    common = [
        ("outer_a", "S-ORS", 2, "2024-01-01T00:00:00Z"),
        ("outer_a", "S-ORS", 3, "2024-01-01T00:00:00Z"),
        ("outer_a", "S-ORS", 4, "2024-01-01T00:00:00Z"),
    ]
    reference = _frame(common)
    comparator = _frame(common)
    truth = _frame(
        common + [("outer_b", "S-ORS", 2, "2025-01-01T00:00:00Z")]
    )
    audit = key_alignment_audit(reference, truth, comparator, pd_module=pd)
    assert audit["common_metric_rows"] == 3
    assert audit["truth_rows"] == 4
    assert audit["truth_only_rows_excluded"] == 1
    assert audit["reference_minus_truth_rows"] == 0
    assert audit["truth_only_by_fold_layer"] == {"outer_b|layer_2": 1}


def test_committed_prediction_key_missing_truth_fails_closed() -> None:
    reference = _frame(
        [
            ("outer_a", "S-ORS", 2, "2024-01-01T00:00:00Z"),
            ("outer_a", "S-ORS", 3, "2024-01-01T00:00:00Z"),
        ]
    )
    comparator = reference.copy()
    truth = reference.iloc[:1].copy()
    with pytest.raises(ValueError, match="missing truth"):
        key_alignment_audit(reference, truth, comparator, pd_module=pd)


def test_reference_and_r3_key_drift_fails_closed() -> None:
    reference = _frame(
        [("outer_a", "S-ORS", 2, "2024-01-01T00:00:00Z")]
    )
    comparator = _frame(
        [("outer_a", "S-ORS", 3, "2024-01-01T00:00:00Z")]
    )
    truth = pd.concat([reference, comparator], ignore_index=True)
    with pytest.raises(ValueError, match="reference/r3"):
        key_alignment_audit(reference, truth, comparator, pd_module=pd)


def test_duplicate_key_surface_fails_closed() -> None:
    reference = _frame(
        [
            ("outer_a", "S-ORS", 2, "2024-01-01T00:00:00Z"),
            ("outer_a", "S-ORS", 2, "2024-01-01T00:00:00Z"),
        ]
    )
    with pytest.raises(ValueError, match="duplicated"):
        key_alignment_audit(reference, reference.iloc[:1], reference, pd_module=pd)


def test_recovery_is_append_only_and_has_no_training_or_official_file_entrypoint() -> None:
    output = ROOT / "artifacts/p2_joint_hydrographic_multitask_layer4_checkpoint_v1"
    assert output.joinpath("prediction_commitment.json").is_file()
    assert set(RECOVERY_FILES) == {
        "recovery.lock",
        "recovery_receipt.json",
        "metrics.json",
        "checkpoint_oof.csv",
        "training_receipt.json",
        "manifest.json",
        "manifest.sha256",
        "seal.json",
    }
    source = (
        ROOT
        / "src/p2_restore/joint_hydrographic_multitask_layer4_checkpoint_v1_recovery.py"
    ).read_text(encoding="utf-8")
    assert "optimizer.step(" not in source
    assert 'data_dir / "test_index.csv"' not in source
    assert 'data_dir / "sample_submission.csv"' not in source
    assert 'data_dir / "baseline_interp.csv"' not in source
