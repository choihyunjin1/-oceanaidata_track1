"""Benchmark P1 CSV and generated Parquet ingestion without exposing raw rows.

Each measured load runs in a fresh child process.  This isolates framework
allocators and makes peak-RSS deltas comparable, while the OS file cache stays
warm to represent repeated local model-development runs.  Source files are
opened read-only; generated ZSTD Parquet files and JSON results live under an
ignored artifacts directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

CSV_METHODS = (
    "p1_audited_csv",
    "pandas_csv_default",
    "pandas_csv_typed_usecols",
    "pandas_csv_pyarrow",
    "pyarrow_csv",
    "polars_csv_eager",
    "polars_csv_lazy_collect",
)
PARQUET_METHODS = (
    "pandas_parquet_numpy",
    "pandas_parquet_arrow",
    "pyarrow_parquet",
    "polars_parquet_eager",
    "polars_parquet_lazy_collect",
)
LOADER_METHODS = CSV_METHODS + PARQUET_METHODS
PANDAS_COMPATIBLE_CSV = (
    "p1_audited_csv",
    "pandas_csv_default",
    "pandas_csv_typed_usecols",
    # Full test feature construction is exercised separately before this
    # Arrow-backed pandas path is eligible for the recommendation table.
    "pandas_csv_pyarrow",
)
PANDAS_COMPATIBLE_PARQUET = ("pandas_parquet_numpy", "pandas_parquet_arrow")
DATASET_COLUMNS = {
    "train": (
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "label",
        "anomaly_type",
    ),
    "test": ("station", "year", "layer", "time", "temp", "psal", "depth"),
}


def resolve_data_dir(explicit: str | Path | None, *, root: Path | None = None) -> Path:
    """Resolve P1_DATA_DIR first, then require a unique repository fallback."""

    candidate = explicit or os.environ.get("P1_DATA_DIR")
    if candidate:
        directory = Path(candidate).expanduser().resolve(strict=True)
        _validate_source_directory(directory)
        return directory
    search_root = (root or Path.cwd()).resolve(strict=True)
    matches = sorted(
        {
            item.parent.resolve()
            for item in search_root.rglob("train.csv")
            if (item.parent / "test.csv").is_file()
            and (item.parent / "sample_submission.csv").is_file()
        }
    )
    if len(matches) != 1:
        raise RuntimeError(
            "P1_DATA_DIR is unset and repository fallback found "
            f"{len(matches)} candidate directories"
        )
    _validate_source_directory(matches[0])
    return matches[0]


def _validate_source_directory(directory: Path) -> None:
    missing = [name for name in ("train.csv", "test.csv") if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P1 source directory is missing {missing}")


def _column_types(kind: str) -> dict[str, Any]:
    import pyarrow as pa

    values = {
        "station": pa.string(),
        "year": pa.int16(),
        "layer": pa.int8(),
        # Preserve the exact +09:00 key representation rather than normalizing
        # timestamps during cache construction.
        "time": pa.string(),
        "temp": pa.float32(),
        "psal": pa.float32(),
        "depth": pa.float32(),
    }
    if kind == "train":
        values.update({"label": pa.int8(), "anomaly_type": pa.string()})
    return values


def _pandas_dtypes(kind: str) -> dict[str, str]:
    values = {
        "station": "category",
        "year": "int16",
        "layer": "int8",
        "time": "string[pyarrow]",
        "temp": "float32",
        "psal": "float32",
        "depth": "float32",
    }
    if kind == "train":
        values.update({"label": "int8", "anomaly_type": "string[pyarrow]"})
    return values


def _prepare_loader(
    method: str,
    source: Path,
    parquet: Path | None,
    kind: str,
    config_path: Path | None,
) -> Callable[[], tuple[Any, dict[str, Any]]]:
    columns = list(DATASET_COLUMNS[kind])
    if method == "p1_audited_csv":
        from p1_qc.data import load_dataset

        return lambda: (load_dataset(source, kind=kind, audit=True, strict=True), {})
    if method == "pandas_csv_default":
        import pandas as pd

        return lambda: (pd.read_csv(source, low_memory=False), {})
    if method == "pandas_csv_typed_usecols":
        import pandas as pd

        return lambda: (
            pd.read_csv(
                source,
                usecols=columns,
                dtype=_pandas_dtypes(kind),
                low_memory=False,
            ),
            {},
        )
    if method == "pandas_csv_pyarrow":
        import pandas as pd

        return lambda: (
            pd.read_csv(
                source,
                engine="pyarrow",
                dtype_backend="pyarrow",
                usecols=columns,
            ),
            {},
        )
    if method == "pyarrow_csv":
        import pyarrow.csv as csv

        convert = csv.ConvertOptions(
            column_types=_column_types(kind),
            include_columns=columns,
            strings_can_be_null=True,
        )
        read = csv.ReadOptions(use_threads=True, block_size=1 << 20)
        return lambda: (csv.read_csv(source, read_options=read, convert_options=convert), {})
    if method == "polars_csv_eager":
        import polars as pl

        return lambda: (
            pl.read_csv(source, columns=columns, try_parse_dates=False, low_memory=False),
            {},
        )
    if method == "polars_csv_lazy_collect":
        import polars as pl

        def load_polars_csv_lazy() -> tuple[Any, dict[str, Any]]:
            started = time.perf_counter()
            plan = pl.scan_csv(source, try_parse_dates=False).select(columns)
            plan_seconds = time.perf_counter() - started
            return plan.collect(engine="streaming"), {"lazy_plan_seconds": plan_seconds}

        return load_polars_csv_lazy
    if method == "feature_build":
        import pandas as pd

        from p1_qc.config import load_config
        from p1_qc.features import build_features

        if config_path is None:
            raise ValueError("feature_build requires --config")
        frame = pd.read_csv(source, low_memory=False)
        config = load_config(config_path)

        def build_feature_bundle() -> tuple[Any, dict[str, Any]]:
            bundle = build_features(
                frame,
                config=config,
                cadence_minutes=config.data.cadence_minutes,
                group_columns=config.data.group_columns,
            )
            return bundle.frame, {"feature_mode": config.features.mode}

        return build_feature_bundle
    if parquet is None:
        raise ValueError(f"{method} requires a Parquet path")
    if method == "pandas_parquet_numpy":
        import pandas as pd

        return lambda: (pd.read_parquet(parquet, engine="pyarrow", columns=columns), {})
    if method == "pandas_parquet_arrow":
        import pandas as pd

        return lambda: (
            pd.read_parquet(
                parquet,
                engine="pyarrow",
                columns=columns,
                dtype_backend="pyarrow",
            ),
            {},
        )
    if method == "pyarrow_parquet":
        import pyarrow.parquet as pq

        return lambda: (pq.read_table(parquet, columns=columns, use_threads=True), {})
    if method == "polars_parquet_eager":
        import polars as pl

        return lambda: (pl.read_parquet(parquet, columns=columns), {})
    if method == "polars_parquet_lazy_collect":
        import polars as pl

        def load_polars_parquet_lazy() -> tuple[Any, dict[str, Any]]:
            started = time.perf_counter()
            plan = pl.scan_parquet(parquet).select(columns)
            plan_seconds = time.perf_counter() - started
            return plan.collect(engine="streaming"), {"lazy_plan_seconds": plan_seconds}

        return load_polars_parquet_lazy
    raise ValueError(f"unknown benchmark method: {method}")


def _object_contract(value: Any) -> tuple[int, int, list[str], int]:
    module = type(value).__module__.split(".")[0]
    if module == "pandas":
        rows, columns = value.shape
        size = int(value.memory_usage(index=True, deep=True).sum())
        names = [str(item) for item in value.columns]
    elif module == "pyarrow":
        rows, columns = value.num_rows, value.num_columns
        size = int(value.nbytes)
        names = [str(item) for item in value.column_names]
    elif module == "polars":
        rows, columns = value.shape
        size = int(value.estimated_size())
        names = [str(item) for item in value.columns]
    else:
        raise TypeError(f"unsupported loaded object: {type(value)!r}")
    return int(rows), int(columns), names, size


def _quality_contract(value: Any) -> dict[str, Any]:
    """Return aggregate-only parsing invariants shared by all frameworks."""

    module = type(value).__module__.split(".")[0]
    if module == "pandas":
        import pandas as pd

        nullish = {}
        for name in value.columns:
            series = value[name]
            missing = series.isna()
            if pd.api.types.is_string_dtype(series.dtype) or isinstance(
                series.dtype, pd.CategoricalDtype
            ):
                missing = missing | series.astype("string").eq("").fillna(False)
            nullish[str(name)] = int(missing.sum())
        label_positive = int(value["label"].sum()) if "label" in value else None
        station_distinct = (
            int(value["station"].nunique(dropna=True)) if "station" in value else None
        )
        layer_distinct = int(value["layer"].nunique(dropna=True)) if "layer" in value else None
        dtypes = {str(name): str(dtype) for name, dtype in value.dtypes.items()}
    elif module == "pyarrow":
        import pyarrow as pa
        import pyarrow.compute as pc

        nullish = {}
        for name in value.column_names:
            column = value[name]
            count = int(column.null_count)
            if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
                count += int(pc.sum(pc.equal(column, "")).as_py() or 0)
            nullish[str(name)] = count
        label_positive = (
            int(pc.sum(value["label"]).as_py()) if "label" in value.column_names else None
        )
        station_distinct = (
            int(pc.count_distinct(value["station"]).as_py())
            if "station" in value.column_names
            else None
        )
        layer_distinct = (
            int(pc.count_distinct(value["layer"]).as_py())
            if "layer" in value.column_names
            else None
        )
        dtypes = {field.name: str(field.type) for field in value.schema}
    elif module == "polars":
        import polars as pl

        nullish = {}
        for name, dtype in value.schema.items():
            expression = pl.col(name).is_null().sum()
            if dtype == pl.String:
                expression = expression + pl.col(name).eq("").fill_null(False).sum()
            nullish[str(name)] = int(value.select(expression).item())
        label_positive = int(value["label"].sum()) if "label" in value.columns else None
        station_distinct = int(value["station"].n_unique()) if "station" in value.columns else None
        layer_distinct = int(value["layer"].n_unique()) if "layer" in value.columns else None
        dtypes = {str(name): str(dtype) for name, dtype in value.schema.items()}
    else:
        raise TypeError(f"unsupported loaded object: {type(value)!r}")
    return {
        "normalized_nullish_counts": nullish,
        "label_positive": label_positive,
        "station_distinct": station_distinct,
        "layer_distinct": layer_distinct,
        "dtypes": dtypes,
    }


def _measure(
    loader: Callable[[], tuple[Any, dict[str, Any]]],
    *,
    method: str,
    kind: str,
) -> dict[str, Any]:
    import psutil

    gc.collect()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    peak_rss = rss_before
    stopped = threading.Event()

    def sample_rss() -> None:
        nonlocal peak_rss
        while not stopped.wait(0.002):
            peak_rss = max(peak_rss, process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    started = time.perf_counter()
    value, extra = loader()
    elapsed = time.perf_counter() - started
    rss_after = process.memory_info().rss
    peak_rss = max(peak_rss, rss_after)
    stopped.set()
    sampler.join(timeout=1.0)
    rows, columns, names, object_bytes = _object_contract(value)
    quality = _quality_contract(value)
    return {
        "method": method,
        "dataset": kind,
        "seconds": elapsed,
        "peak_rss_delta_bytes": max(0, peak_rss - rss_before),
        "rss_end_delta_bytes": max(0, rss_after - rss_before),
        "object_bytes": object_bytes,
        "rows": rows,
        "columns": columns,
        "column_names": names,
        "quality": quality,
        **extra,
    }


def _prepare_cache_builder(
    source: Path,
    destination: Path,
    kind: str,
) -> Callable[[], tuple[Any, dict[str, Any]]]:
    import pyarrow.csv as csv
    import pyarrow.parquet as pq

    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(DATASET_COLUMNS[kind])
    convert = csv.ConvertOptions(
        column_types=_column_types(kind),
        include_columns=columns,
        strings_can_be_null=True,
    )
    read = csv.ReadOptions(use_threads=True, block_size=1 << 20)

    def build() -> tuple[Any, dict[str, Any]]:
        table = csv.read_csv(source, read_options=read, convert_options=convert)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )
        temporary.replace(destination)
        return table, {
            "cache_bytes": destination.stat().st_size,
            "compression": "zstd",
            "compression_level": 3,
        }

    return build


def _worker(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve(strict=True)
    if args.worker_method == "build_parquet_zstd":
        if args.parquet is None:
            raise ValueError("cache build requires --parquet")
        destination = Path(args.parquet).resolve()
        loader = _prepare_cache_builder(source, destination, args.kind)
    else:
        loader = _prepare_loader(
            args.worker_method,
            source,
            None if args.parquet is None else Path(args.parquet).resolve(strict=True),
            args.kind,
            None if args.config is None else Path(args.config).resolve(strict=True),
        )
    result = _measure(loader, method=args.worker_method, kind=args.kind)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    return 0


def _run_worker(
    *,
    method: str,
    kind: str,
    source: Path,
    parquet: Path | None,
    config_path: Path | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-method",
        method,
        "--kind",
        kind,
        "--source",
        str(source),
    ]
    if parquet is not None:
        command.extend(("--parquet", str(parquet)))
    if config_path is not None:
        command.extend(("--config", str(config_path)))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
    )
    if completed.returncode:
        raise RuntimeError(
            f"worker {method}/{kind} failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _prewarm(paths: Iterable[Path]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for path in paths:
        started = time.perf_counter()
        with path.open("rb") as handle:
            while handle.read(8 << 20):
                pass
        timings[path.name] = time.perf_counter() - started
    return timings


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_runs(
    runs: Sequence[dict[str, Any]],
    *,
    source_sizes: dict[str, int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["dataset"], run["method"]), []).append(run)
    summaries: list[dict[str, Any]] = []
    for (dataset, method), values in sorted(grouped.items()):
        seconds = [float(item["seconds"]) for item in values]
        peak = [float(item["peak_rss_delta_bytes"]) for item in values]
        objects = [float(item["object_bytes"]) for item in values]
        median_seconds = statistics.median(seconds)
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "repeats": len(values),
                "median_seconds": median_seconds,
                "p95_seconds": _percentile(seconds, 95),
                "median_peak_rss_delta_mib": statistics.median(peak) / (1 << 20),
                "p95_peak_rss_delta_mib": _percentile(peak, 95) / (1 << 20),
                "median_object_mib": statistics.median(objects) / (1 << 20),
                "median_input_mib_per_second": (source_sizes[dataset] / (1 << 20) / median_seconds),
                "rows": values[0]["rows"],
                "columns": values[0]["columns"],
            }
        )
    return summaries


def _validate_contracts(runs: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run["method"] == "feature_build":
            continue
        current = {
            "rows": run["rows"],
            "columns": run["columns"],
            "column_names": run["column_names"],
            "normalized_nullish_counts": run["quality"]["normalized_nullish_counts"],
            "label_positive": run["quality"]["label_positive"],
            "station_distinct": run["quality"]["station_distinct"],
            "layer_distinct": run["quality"]["layer_distinct"],
        }
        existing = contracts.setdefault(run["dataset"], current)
        if current != existing:
            raise RuntimeError(f"schema/shape mismatch for {run['dataset']} method {run['method']}")
    return contracts


def select_fastest(
    summaries: Sequence[dict[str, Any]],
    dataset: str,
    methods: Sequence[str],
) -> dict[str, Any]:
    candidates = [
        row for row in summaries if row["dataset"] == dataset and row["method"] in methods
    ]
    if not candidates:
        raise ValueError(f"no benchmark candidates for {dataset}: {methods}")
    return min(candidates, key=lambda row: row["median_seconds"])


def break_even_loads(
    cache_build_seconds: float,
    csv_load_seconds: float,
    parquet_load_seconds: float,
) -> int | None:
    savings = csv_load_seconds - parquet_load_seconds
    if cache_build_seconds < 0 or csv_load_seconds <= 0 or parquet_load_seconds <= 0:
        raise ValueError("timings must be positive (cache build may be zero)")
    if savings <= 0:
        return None
    return max(1, math.ceil(cache_build_seconds / savings))


def build_recommendations(
    loader_summary: Sequence[dict[str, Any]],
    build_summary: Sequence[dict[str, Any]],
    source_sizes: dict[str, int],
    parquet_sizes: dict[str, int],
) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    combined_csv = 0.0
    combined_parquet = 0.0
    combined_build = 0.0
    for kind in ("train", "test"):
        csv_fastest = select_fastest(loader_summary, kind, PANDAS_COMPATIBLE_CSV)
        parquet_fastest = select_fastest(loader_summary, kind, PANDAS_COMPATIBLE_PARQUET)
        build = next(
            row
            for row in build_summary
            if row["dataset"] == kind and row["method"] == "build_parquet_zstd"
        )
        combined_csv += csv_fastest["median_seconds"]
        combined_parquet += parquet_fastest["median_seconds"]
        combined_build += build["median_seconds"]
        recommendations[kind] = {
            "fastest_pipeline_compatible_csv": csv_fastest["method"],
            "fastest_pipeline_compatible_parquet": parquet_fastest["method"],
            "csv_median_seconds": csv_fastest["median_seconds"],
            "parquet_median_seconds": parquet_fastest["median_seconds"],
            "speedup": csv_fastest["median_seconds"] / parquet_fastest["median_seconds"],
            "cache_build_median_seconds": build["median_seconds"],
            "break_even_repeated_loads": break_even_loads(
                build["median_seconds"],
                csv_fastest["median_seconds"],
                parquet_fastest["median_seconds"],
            ),
            "cache_size_ratio_to_csv": parquet_sizes[kind] / source_sizes[kind],
        }
    recommendations["combined_train_test"] = {
        "csv_median_seconds": combined_csv,
        "parquet_median_seconds": combined_parquet,
        "cache_build_median_seconds": combined_build,
        "speedup": combined_csv / combined_parquet,
        "break_even_repeated_loads": break_even_loads(
            combined_build, combined_csv, combined_parquet
        ),
    }
    return recommendations


def resummarize_existing(path: str | Path) -> Path:
    """Refresh policy-derived recommendations without rerunning measurements."""

    destination = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    source_sizes = {kind: int(item["bytes"]) for kind, item in payload["sources"].items()}
    parquet_sizes = {kind: int(item["bytes"]) for kind, item in payload["parquet_cache"].items()}
    payload["recommendations"] = build_recommendations(
        payload["loader_summary"],
        payload["cache_build_summary"],
        source_sizes,
        parquet_sizes,
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _package_versions() -> dict[str, str]:
    import importlib.metadata

    result = {}
    for distribution in ("pandas", "pyarrow", "polars", "psutil"):
        result[distribution] = importlib.metadata.version(distribution)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _safe_output_dir(value: str | Path, root: Path) -> Path:
    output = Path(value)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    artifact_root = (root / "artifacts").resolve()
    if output != artifact_root and artifact_root not in output.parents:
        raise ValueError("benchmark output must remain below the ignored artifacts directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _main(args: argparse.Namespace) -> int:
    if args.repeats < 3:
        raise ValueError("at least three repeats are required")
    root = Path(__file__).resolve().parents[1]
    data_dir = resolve_data_dir(args.data_dir, root=root)
    output_dir = _safe_output_dir(args.output_dir, root)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve(strict=True)
    sources = {kind: data_dir / f"{kind}.csv" for kind in ("train", "test")}
    parquet = {kind: output_dir / "parquet" / f"{kind}.zstd.parquet" for kind in sources}

    build_runs: list[dict[str, Any]] = []
    for repetition in range(args.repeats):
        for kind in ("train", "test"):
            run = _run_worker(
                method="build_parquet_zstd",
                kind=kind,
                source=sources[kind],
                parquet=parquet[kind],
                config_path=None,
            )
            run["repetition"] = repetition + 1
            build_runs.append(run)

    prewarm = {} if args.no_prewarm else _prewarm([*sources.values(), *parquet.values()])
    methods = tuple(args.methods.split(",")) if args.methods else LOADER_METHODS
    unknown = sorted(set(methods).difference(LOADER_METHODS))
    if unknown:
        raise ValueError(f"unknown loader methods: {unknown}")
    jobs = [(kind, method) for kind in ("train", "test") for method in methods]
    runs: list[dict[str, Any]] = []
    for repetition in range(args.repeats):
        order = jobs.copy()
        random.Random(args.seed + repetition).shuffle(order)
        for kind, method in order:
            run = _run_worker(
                method=method,
                kind=kind,
                source=sources[kind],
                parquet=parquet[kind] if method in PARQUET_METHODS else None,
                config_path=config_path,
            )
            run["repetition"] = repetition + 1
            runs.append(run)

    feature_runs: list[dict[str, Any]] = []
    if not args.skip_feature_profile:
        for repetition in range(args.repeats):
            for kind in ("train", "test"):
                run = _run_worker(
                    method="feature_build",
                    kind=kind,
                    source=sources[kind],
                    parquet=None,
                    config_path=config_path,
                )
                run["repetition"] = repetition + 1
                feature_runs.append(run)
    contracts = _validate_contracts(runs)
    source_sizes = {kind: path.stat().st_size for kind, path in sources.items()}
    parquet_sizes = {kind: path.stat().st_size for kind, path in parquet.items()}
    loader_summary = summarize_runs(
        runs,
        source_sizes={
            **source_sizes,
        },
    )
    feature_summary = (
        summarize_runs(feature_runs, source_sizes=source_sizes) if feature_runs else []
    )
    build_summary = summarize_runs(build_runs, source_sizes=source_sizes)

    recommendations = build_recommendations(
        loader_summary, build_summary, source_sizes, parquet_sizes
    )

    payload = {
        "contract_version": 1,
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "warm_cache": not args.no_prewarm,
            "cold_cache_measured": False,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "sources": {
            kind: {
                "name": path.name,
                "bytes": source_sizes[kind],
                "sha256": _sha256(path),
            }
            for kind, path in sources.items()
        },
        "parquet_cache": {
            kind: {
                "name": path.name,
                "bytes": parquet_sizes[kind],
                "sha256": _sha256(path),
            }
            for kind, path in parquet.items()
        },
        "prewarm_seconds": prewarm,
        "contracts": contracts,
        "cache_build_runs": build_runs,
        "cache_build_summary": build_summary,
        "loader_runs": runs,
        "loader_summary": loader_summary,
        "feature_runs": feature_runs,
        "feature_summary": feature_summary,
        "recommendations": recommendations,
        "limitations": [
            "OS file-cache eviction was not attempted; results represent warm-cache repeated work.",
            "Peak RSS is sampled every 2 ms and is a lower bound for shorter native allocations.",
            "Different dataframe memory estimators are framework-specific and not byte-identical.",
        ],
    }
    destination = output_dir / "benchmark_results.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(
        json.dumps(
            {
                "results": str(destination),
                "loader_methods": len(methods),
                "repeats": args.repeats,
                "recommendations": recommendations,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="artifacts/cache/io_benchmark")
    parser.add_argument("--config", default="configs/p1.toml")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--methods", help="comma-separated loader subset")
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--skip-feature-profile", action="store_true")
    parser.add_argument(
        "--resummarize-existing",
        help="refresh recommendations in an existing benchmark JSON without reloading data",
    )
    parser.add_argument("--worker-method", help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=("train", "test"), help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--parquet", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resummarize_existing:
        print(resummarize_existing(args.resummarize_existing))
        return 0
    if args.worker_method:
        if not args.kind or not args.source:
            raise ValueError("worker mode requires --kind and --source")
        return _worker(args)
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
