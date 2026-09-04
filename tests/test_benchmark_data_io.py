from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_data_io.py"
SPEC = importlib.util.spec_from_file_location("benchmark_data_io", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _source_frame(kind: str, rows: int = 48) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * rows,
            "year": [2025] * rows,
            "layer": [1] * rows,
            "time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="Asia/Seoul").astype(
                str
            ),
            "temp": [15.0 + index / 100 for index in range(rows)],
            "psal": [32.0] * rows,
            "depth": [5.0] * rows,
        }
    )
    if kind == "train":
        frame["label"] = [1 if 12 <= index < 18 else 0 for index in range(rows)]
        frame["anomaly_type"] = ["flatline" if 12 <= index < 18 else "" for index in range(rows)]
    return frame


@pytest.mark.parametrize("kind", ["train", "test"])
def test_all_loaders_preserve_shape_and_column_contract(tmp_path: Path, kind: str) -> None:
    source = tmp_path / f"{kind}.csv"
    parquet = tmp_path / f"{kind}.zstd.parquet"
    expected = _source_frame(kind)
    expected.to_csv(source, index=False)
    builder = benchmark._prepare_cache_builder(source, parquet, kind)
    built, metadata = builder()
    expected_quality = benchmark._quality_contract(expected)

    assert built.num_rows == len(expected)
    assert metadata["compression"] == "zstd"
    assert parquet.is_file()
    for method in benchmark.CSV_METHODS[1:] + benchmark.PARQUET_METHODS:
        loader = benchmark._prepare_loader(method, source, parquet, kind, None)
        value, _ = loader()
        rows, columns, names, object_bytes = benchmark._object_contract(value)
        assert rows == len(expected), method
        assert columns == len(expected.columns), method
        assert names == list(expected.columns), method
        assert object_bytes > 0, method
        observed_quality = benchmark._quality_contract(value)
        assert (
            observed_quality["normalized_nullish_counts"]
            == expected_quality["normalized_nullish_counts"]
        ), method
        assert observed_quality["label_positive"] == expected_quality["label_positive"], method


def test_zstd_cache_keeps_time_as_string_key(tmp_path: Path) -> None:
    import pyarrow as pa

    source = tmp_path / "test.csv"
    destination = tmp_path / "test.zstd.parquet"
    expected = _source_frame("test")
    expected.to_csv(source, index=False)
    table, _ = benchmark._prepare_cache_builder(source, destination, "test")()

    assert table.schema.field("time").type == pa.string()
    assert table.column("time")[0].as_py() == expected.loc[0, "time"]


def test_feature_profile_does_not_require_parquet(tmp_path: Path) -> None:
    source = tmp_path / "test.csv"
    _source_frame("test").to_csv(source, index=False)
    config = Path(__file__).resolve().parents[1] / "configs" / "p1.toml"
    loader = benchmark._prepare_loader("feature_build", source, None, "test", config)
    features, metadata = loader()

    assert len(features) == 48
    assert features.shape[1] > len(_source_frame("test").columns)
    assert metadata["feature_mode"] == "offline"
    quality = benchmark._quality_contract(features)
    assert quality["station_distinct"] == 1
    assert quality["layer_distinct"] is None


def test_summary_p95_and_break_even_are_deterministic() -> None:
    runs = [
        {
            "dataset": "train",
            "method": "example",
            "seconds": seconds,
            "peak_rss_delta_bytes": peak,
            "object_bytes": 1024,
            "rows": 10,
            "columns": 2,
        }
        for seconds, peak in ((1.0, 100), (2.0, 200), (4.0, 400))
    ]
    summary = benchmark.summarize_runs(runs, source_sizes={"train": 1 << 20})[0]

    assert summary["median_seconds"] == 2.0
    assert summary["p95_seconds"] == pytest.approx(3.8)
    assert benchmark.break_even_loads(2.0, 1.0, 0.5) == 4
    assert benchmark.break_even_loads(2.0, 0.5, 1.0) is None


def test_recommendations_include_arrow_csv_and_combined_break_even() -> None:
    summaries = []
    for dataset in ("train", "test"):
        for method, seconds in (
            ("pandas_csv_default", 1.0),
            ("pandas_csv_typed_usecols", 0.8),
            ("pandas_csv_pyarrow", 0.2),
            ("pandas_parquet_numpy", 0.1),
            ("pandas_parquet_arrow", 0.15),
        ):
            summaries.append({"dataset": dataset, "method": method, "median_seconds": seconds})
    builds = [
        {
            "dataset": dataset,
            "method": "build_parquet_zstd",
            "median_seconds": 0.4,
        }
        for dataset in ("train", "test")
    ]
    recommendations = benchmark.build_recommendations(
        summaries,
        builds,
        {"train": 100, "test": 50},
        {"train": 20, "test": 10},
    )

    assert recommendations["train"]["fastest_pipeline_compatible_csv"] == ("pandas_csv_pyarrow")
    assert recommendations["combined_train_test"]["break_even_repeated_loads"] == 4


def test_contract_validation_rejects_loader_shape_mismatch() -> None:
    base = {
        "dataset": "train",
        "method": "one",
        "rows": 10,
        "columns": 2,
        "column_names": ["a", "b"],
        "quality": {
            "normalized_nullish_counts": {"a": 0, "b": 0},
            "label_positive": None,
            "station_distinct": 1,
            "layer_distinct": 1,
        },
    }
    mismatch = {**base, "method": "two", "rows": 9}
    with pytest.raises(RuntimeError, match="schema/shape mismatch"):
        benchmark._validate_contracts([base, mismatch])


def test_output_directory_must_stay_below_ignored_artifacts(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    accepted = benchmark._safe_output_dir("artifacts/cache/io", tmp_path)
    assert accepted == (tmp_path / "artifacts/cache/io").resolve()
    with pytest.raises(ValueError, match="artifacts"):
        benchmark._safe_output_dir(tmp_path / "reports", tmp_path)
