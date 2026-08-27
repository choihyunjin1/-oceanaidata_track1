from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from p2_restore.dynamic_sigmoid_key_alignment import (
    key_aligned_gate_1,
    load_key_only_population,
)
from p2_restore.dynamic_sigmoid_profile import TimeBlock

ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs/experiments/p2_dynamic_sigmoid_profile_v1.json"
V2_CONFIG = ROOT / "configs/experiments/p2_dynamic_sigmoid_profile_v2_key_aligned.json"


def test_v2_changes_only_gate_1_population_contract() -> None:
    v1 = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    v2 = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    unchanged = (
        "objective",
        "problem_contract",
        "validation",
        "sigmoid",
        "public_features",
        "gates",
        "incumbent",
        "provenance",
    )
    assert all(v2[section] == v1[section] for section in unchanged)
    assert v2["adaptive_after_prior_outer_exposure"] is True
    assert v2["fresh_holdout_claimed"] is False
    population = v2["gate_1_population"]
    assert population["columns_allowed"] == ["time", "layer"]
    assert population["truth_read"] is False
    assert population["prediction_read"] is False
    assert population["unique_time_denominator"] is True


def test_v2_pins_parent_and_denominator_diagnostic() -> None:
    v2 = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    generation = v2["generation"]
    parent = ROOT / generation["parent_config_path"]
    diagnostic = ROOT / generation["denominator_diagnostic_path"]
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == generation["parent_config_sha256"]
    assert (
        hashlib.sha256(diagnostic.read_bytes()).hexdigest()
        == generation["denominator_diagnostic_sha256"]
    )


def test_key_only_loader_requests_only_time_and_layer(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "oof.parquet"
    source.write_bytes(b"pinned")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    calls: list[list[str]] = []

    def fake_read_parquet(path: Path, *, columns: list[str]) -> pd.DataFrame:
        assert path == source
        calls.append(columns)
        return pd.DataFrame(
            {
                "time": ["2024-09-01T00:00:00+09:00"],
                "layer": [2],
            }
        )

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    result = load_key_only_population(source, expected_sha256=expected)
    assert calls == [["time", "layer"]]
    assert list(result.columns) == ["time", "layer"]


def test_key_aligned_denominator_excludes_unscored_unsupported_times() -> None:
    index = pd.date_range("2025-11-01T00:00:00+09:00", periods=5, freq="10min").tz_convert("UTC")
    features = pd.DataFrame(
        {
            "public_temp_count": [5, 5, 2, 2, 2],
            "public_depth_span": [45.0, 45.0, 10.0, 10.0, 10.0],
        },
        index=index,
    )
    keys = pd.DataFrame(
        {
            "time": [index[0], index[0], index[1], index[1]],
            "layer": [2, 3, 2, 3],
        }
    )
    block = TimeBlock.from_strings(
        "block", ("2025-11-01T00:00:00+09:00", "2025-11-02T00:00:00+09:00")
    )
    result = key_aligned_gate_1(
        features,
        keys,
        block,
        minimum_public_points=4,
        minimum_depth_span_m=30.0,
        threshold=0.8,
    )
    assert result["validation_times"] == 2
    assert result["supported_times"] == 2
    assert result["support_share"] == 1.0
    assert result["pass"] is True
    assert result["full_grid_diagnostic"]["support_share"] == 0.4
    assert result["full_grid_diagnostic"]["denominator_removed_times"] == 3


def test_v2_runner_has_no_submission_or_test_reader() -> None:
    source = (ROOT / "scripts/run_p2_dynamic_sigmoid_profile_v2_key_aligned.py").read_text(
        encoding="utf-8"
    )
    assert "to_csv(" not in source
    assert "test_index.csv" not in source
    assert "sample_submission.csv" not in source
    assert "requests." not in source
