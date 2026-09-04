from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p1_qc.data import sha256_file
from scripts.validate_r1_oof import (
    R1OOFValidationError,
    build_r1_oof_report,
    main,
    write_r1_oof_report,
)


def _oof(rows: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    time = pd.date_range("2025-04-01", periods=rows, freq="10min", tz="Asia/Seoul")
    label = np.ones(rows, dtype=np.int8)
    label[:6] = 0
    label[294:] = 0
    baseline_prediction = np.zeros(rows, dtype=np.int8)
    baseline_prediction[10:150] = 1
    candidate_prediction = baseline_prediction.copy()
    candidate_prediction[6:294] = 1
    probability = np.linspace(0.01, 0.99, rows, dtype=np.float32)
    common = pd.DataFrame(
        {
            "station": ["A"] * rows,
            "year": [2025] * rows,
            "layer": [1] * rows,
            "time": time.astype(str),
            "fold": ["fold_1"] * rows,
            "label": label,
            "probability": probability,
            "anomaly_type": np.where(label == 1, "offset", ""),
        }
    )
    baseline = common.assign(prediction=baseline_prediction)
    candidate = common.assign(
        base_prediction=baseline_prediction,
        prediction=candidate_prediction,
    )
    return candidate, baseline


def _test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["A"] * 2 + ["B"] * 8,
            "layer": [1] * 10,
            "time": pd.date_range("2026-01-01", periods=10, freq="10min", tz="Asia/Seoul").astype(
                str
            ),
        }
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate, baseline = _oof()
    candidate_path = tmp_path / "candidate.parquet"
    baseline_path = tmp_path / "baseline.parquet"
    test_path = tmp_path / "test.csv"
    candidate.to_parquet(candidate_path, index=False)
    baseline.to_parquet(baseline_path, index=False)
    _test_frame().to_csv(test_path, index=False)
    return candidate_path, baseline_path, test_path


def test_report_validates_alignment_and_all_required_aggregate_metrics(tmp_path: Path) -> None:
    candidate_path, baseline_path, test_path = _write_inputs(tmp_path)
    report = build_r1_oof_report(
        candidate_path,
        baseline_path,
        test_path,
        bootstrap_replicates=32,
    )
    assert report["status"] == "passed"
    assert report["alignment"]["candidate_base_prediction_exact_baseline_prediction"]
    assert report["alignment"]["candidate_vs_baseline_probability_max_abs_difference"] == 0
    assert (
        report["official_row_metrics"]["candidate"]["f1"]
        > report["official_row_metrics"]["baseline"]["f1"]
    )
    assert report["test_share_weighted_metrics"]["covered_test_row_share"] == pytest.approx(0.2)
    assert report["long_positive_events"]["events"] == 1
    assert report["long_positive_events"]["positive_rows"] == 288
    assert report["long_positive_events"]["candidate_row_recall"] == 1.0
    assert report["normal_station_layer_day_fp"]["normal_station_layer_days"] == 2
    assert report["paired_block_bootstrap"]["replicates"] == 32
    assert len(report["by_fold"]) == 1
    assert len(report["by_station_layer"]) == 1
    assert {row["anomaly_type"] for row in report["by_anomaly_type_membership_non_additive"]} == {
        "spike",
        "noise",
        "flatline",
        "offset",
        "drift",
    }


def test_key_fold_or_label_reordering_fails_without_emitting_raw_values(tmp_path: Path) -> None:
    candidate, baseline = _oof()
    candidate.loc[[0, 1], "time"] = candidate.loc[[1, 0], "time"].to_numpy()
    candidate_path = tmp_path / "candidate.parquet"
    baseline_path = tmp_path / "baseline.parquet"
    test_path = tmp_path / "test.csv"
    candidate.to_parquet(candidate_path, index=False)
    baseline.to_parquet(baseline_path, index=False)
    _test_frame().to_csv(test_path, index=False)
    with pytest.raises(R1OOFValidationError, match=r"time order differs in 2 rows") as caught:
        build_r1_oof_report(candidate_path, baseline_path, test_path, bootstrap_replicates=4)
    assert "2025-" not in str(caught.value)


def test_candidate_base_prediction_mismatch_fails(tmp_path: Path) -> None:
    candidate, baseline = _oof()
    candidate.loc[0, "base_prediction"] = 1
    candidate_path = tmp_path / "candidate.parquet"
    baseline_path = tmp_path / "baseline.parquet"
    test_path = tmp_path / "test.csv"
    candidate.to_parquet(candidate_path, index=False)
    baseline.to_parquet(baseline_path, index=False)
    _test_frame().to_csv(test_path, index=False)
    with pytest.raises(R1OOFValidationError, match="base_prediction differs"):
        build_r1_oof_report(candidate_path, baseline_path, test_path, bootstrap_replicates=4)


def test_probability_difference_is_reported_but_not_used_as_prediction_identity(
    tmp_path: Path,
) -> None:
    candidate, baseline = _oof()
    candidate["probability"] = np.clip(candidate["probability"] + 0.001, 0, 1)
    candidate_path = tmp_path / "candidate.parquet"
    baseline_path = tmp_path / "baseline.parquet"
    test_path = tmp_path / "test.csv"
    candidate.to_parquet(candidate_path, index=False)
    baseline.to_parquet(baseline_path, index=False)
    _test_frame().to_csv(test_path, index=False)
    report = build_r1_oof_report(
        candidate_path,
        baseline_path,
        test_path,
        bootstrap_replicates=4,
    )
    assert report["alignment"]["candidate_vs_baseline_probability_max_abs_difference"] > 0
    assert not report["alignment"]["candidate_vs_baseline_probability_exact_match"]


def test_json_and_sha_sidecar_are_deterministic_and_aggregate_only(tmp_path: Path) -> None:
    candidate_path, baseline_path, test_path = _write_inputs(tmp_path)
    report = build_r1_oof_report(
        candidate_path,
        baseline_path,
        test_path,
        bootstrap_replicates=8,
    )
    first = write_r1_oof_report(report, tmp_path / "first.json")
    second = write_r1_oof_report(report, tmp_path / "second.json")
    assert first["output_sha256"] == second["output_sha256"]
    assert first["output_sha256"] == sha256_file(first["output"])
    assert first["sha256_sidecar"].read_text(encoding="ascii").startswith(first["output_sha256"])
    payload = json.loads(first["output"].read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "T00:00:00" not in serialized
    assert str(tmp_path) not in serialized
    assert "time" not in payload["sources"]["candidate_oof"]


def test_cli_writes_default_2000_replicate_report(tmp_path: Path) -> None:
    candidate_path, baseline_path, test_path = _write_inputs(tmp_path)
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--candidate-oof",
                str(candidate_path),
                "--baseline-oof",
                str(baseline_path),
                "--test-csv",
                str(test_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["paired_block_bootstrap"]["replicates"] == 2000
    assert output.with_suffix(".json.sha256").is_file()
