"""Exactly-once causal multiscale wavelet-scattering P1 falsification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v11_causal_wavelet_scattering_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V10_RUNNER = ROOT / "scripts/run_p1_v10_causal_recurrence_laminar_state_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v11_shared_execution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


core = _module(V10_RUNNER)
base = core.base
_ORIGINAL_WRITE = core._write


def _causal_mean(values: np.ndarray, rows: int) -> np.ndarray:
    cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    index = np.arange(len(values), dtype=np.int64)
    start = np.maximum(0, index + 1 - rows)
    count = index + 1 - start
    return (cumulative[index + 1] - cumulative[start]) / count


def _haar(values: np.ndarray, scale: int) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.float64)
    if len(values) < 2 * scale:
        return output
    cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    stop = np.arange(2 * scale, len(values) + 1, dtype=np.int64)
    recent = (cumulative[stop] - cumulative[stop - scale]) / scale
    prior = (cumulative[stop - scale] - cumulative[stop - 2 * scale]) / scale
    output[stop - 1] = recent - prior
    return output


def wavelet_scattering_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    representation: dict[str, Any],
) -> np.ndarray:
    """Fixed causal Haar-modulus cascades, reset by group and cadence gap."""

    scales = tuple(int(value) for value in representation["first_order_scales_rows"])
    pairs = tuple(
        (int(first), int(second))
        for first, second in representation["second_order_scale_pairs_rows"]
    )
    average_rows = int(representation["modulus_average_rows"])
    persistence_rows = int(representation["energy_persistence_rows"])
    minimum_prefix = int(representation["minimum_prefix_rows"])
    width = len(scales) + len(pairs) + 2
    output = np.zeros((len(frame), width), dtype=np.float32)
    for _key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = core._time_ns(ordered["_time"])
        raw = ordered["temp"].to_numpy(np.float64)
        prefix = raw[(times <= train_boundary_ns) & np.isfinite(raw)]
        if len(prefix) < minimum_prefix:
            continue
        center = float(np.median(prefix))
        scale = float(1.4826 * np.median(np.abs(prefix - center)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(prefix))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        normalized = np.clip(
            np.nan_to_num((raw - center) / scale, nan=0.0, posinf=12.0, neginf=-12.0),
            -12.0,
            12.0,
        )
        gap = np.r_[True, np.diff(times) != CADENCE_NS]
        starts = np.flatnonzero(gap)
        ends = np.r_[starts[1:], len(ordered)]
        feature = np.zeros((len(ordered), width), dtype=np.float64)
        for start, end in zip(starts, ends, strict=True):
            values = normalized[start:end]
            raw_modulus = {wavelet_scale: np.abs(_haar(values, wavelet_scale)) for wavelet_scale in scales}
            first = np.column_stack(
                [_causal_mean(raw_modulus[wavelet_scale], average_rows) for wavelet_scale in scales]
            )
            second_columns = []
            for first_scale, second_scale in pairs:
                second_columns.append(
                    _causal_mean(
                        np.abs(_haar(raw_modulus[first_scale], second_scale)),
                        average_rows,
                    )
                )
            second = np.column_stack(second_columns)
            short_energy = first[:, :3].sum(axis=1)
            long_energy = first[:, -3:].sum(axis=1)
            ratio = long_energy / (short_energy + long_energy + 1e-9)
            total_raw = np.column_stack(list(raw_modulus.values())).mean(axis=1)
            persistence = _causal_mean(total_raw, persistence_rows)
            feature[start:end] = np.column_stack([first, second, ratio, persistence])
        if not np.isfinite(feature).all():
            raise RuntimeError("wavelet scattering features are nonfinite")
        output[positions] = feature.astype(np.float32)
    return output


def _write_v11(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        payload = dict(value)
        payload["schema_version"] = json.loads(CONFIG.read_text(encoding="utf-8"))[
            "result_schema_version"
        ]
    _ORIGINAL_WRITE(path, payload)


def _configure() -> None:
    core.EXPERIMENT_ID = EXPERIMENT_ID
    core.CONFIG = CONFIG
    core.ARTIFACT = ARTIFACT
    core.LOCK = LOCK
    core.__file__ = str(Path(__file__).resolve())
    core.recurrence_laminar_features = wavelet_scattering_features
    core._write = _write_v11


def preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.preflight(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.qa(data_dir)


def execute(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.execute(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        value = preflight(args.data_dir)
    elif args.qa:
        value = qa(args.data_dir)
    else:
        value = execute(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
