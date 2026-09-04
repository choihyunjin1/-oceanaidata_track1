"""Exactly-once causal endpoint-visibility topology P1 falsification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v13_causal_endpoint_visibility_topology_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V10_RUNNER = ROOT / "scripts/run_p1_v10_causal_recurrence_laminar_state_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v13_shared_execution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


core = _module(V10_RUNNER)
base = core.base
_ORIGINAL_WRITE = core._write
_ORIGINAL_SELECT = base._select
_CURRENT_ENVIRONMENTS: dict[str, np.ndarray] | None = None


def endpoint_visibility(values: np.ndarray, window: int) -> np.ndarray:
    """Upper horizontal visibility summaries for each causal endpoint."""

    output = np.zeros((len(values), 4), dtype=np.float64)
    for endpoint in range(1, len(values)):
        start = max(0, endpoint - window + 1)
        maximum_between = -np.inf
        spans = []
        for prior in range(endpoint - 1, start - 1, -1):
            if maximum_between < min(values[prior], values[endpoint]):
                spans.append(endpoint - prior)
            maximum_between = max(maximum_between, values[prior])
            if maximum_between >= values[endpoint]:
                break
        if not spans:
            continue
        span = np.asarray(spans, dtype=np.float64)
        weights = span / span.sum()
        entropy = -float(np.sum(weights * np.log(weights)))
        if len(span) > 1:
            entropy /= math.log(len(span))
        output[endpoint] = [len(span) / window, span.max() / window, span.mean() / window, entropy]
    return output


def _causal_mean(values: np.ndarray, rows: int) -> np.ndarray:
    cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    index = np.arange(len(values), dtype=np.int64)
    start = np.maximum(0, index + 1 - rows)
    return (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)


def visibility_topology_features(frame: pd.DataFrame, train_boundary_ns: int, representation: dict[str, Any]) -> np.ndarray:
    """Upper/lower endpoint visibility, reset by station-layer and cadence gap."""

    global _CURRENT_ENVIRONMENTS

    window = int(representation["visibility_window_rows"])
    persistence_rows = int(representation["persistence_rows"])
    minimum_prefix = int(representation["minimum_prefix_rows"])
    output = np.zeros((len(frame), 10), dtype=np.float32)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    all_ns = core._time_ns(frame["_time"])
    _CURRENT_ENVIRONMENTS = None
    for item in config["parts"].values():
        audit = json.loads((ROOT / item["audit"]).read_text(encoding="utf-8"))
        cutoff = pd.Timestamp(audit["adjusted_cutoff_utc"]).value
        prefix_times = np.sort(np.unique(all_ns[all_ns <= cutoff]))
        boundary = int(prefix_times[max(0, int(len(prefix_times) * config["selection"]["inner_train_fraction"]) - 1)])
        if boundary != train_boundary_ns:
            continue
        inner = (all_ns > boundary) & (all_ns <= cutoff)
        inner_times = all_ns[inner]
        ordered_times = np.sort(np.unique(inner_times))
        half_cutoff = ordered_times[max(0, len(ordered_times) // 2 - 1)]
        _CURRENT_ENVIRONMENTS = {
            "station": frame.loc[inner, "station"].astype(str).to_numpy(),
            "layer": frame.loc[inner, "layer"].astype(str).to_numpy(),
            "half": (inner_times > half_cutoff).astype(np.int8),
        }
        break
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
        values = np.clip(np.nan_to_num((raw - center) / scale, nan=0.0), -12.0, 12.0)
        gaps = np.r_[True, np.diff(times) != CADENCE_NS]
        starts = np.flatnonzero(gaps)
        ends = np.r_[starts[1:], len(ordered)]
        feature = np.zeros((len(ordered), 10), dtype=np.float64)
        for start, end in zip(starts, ends, strict=True):
            upper = endpoint_visibility(values[start:end], window)
            lower = endpoint_visibility(-values[start:end], window)
            feature[start:end] = np.column_stack([
                upper,
                _causal_mean(upper[:, 0], persistence_rows),
                lower,
                _causal_mean(lower[:, 0], persistence_rows),
            ])
        if not np.isfinite(feature).all():
            raise RuntimeError("visibility topology features are nonfinite")
        output[positions] = feature.astype(np.float32)
    return output


def _transport_stability(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    environments: dict[str, np.ndarray],
    contract: dict[str, Any],
) -> dict[str, Any]:
    proposed = scores >= threshold
    diagnostics = []
    for station, layer, half in sorted(
        set(zip(environments["station"], environments["layer"], environments["half"], strict=True))
    ):
        mask = (
            (environments["station"] == station)
            & (environments["layer"] == layer)
            & (environments["half"] == half)
            & proposed
        )
        count = int(mask.sum())
        if count < contract["minimum_proposals_per_supported_environment"]:
            continue
        true = int(labels[mask].sum())
        diagnostics.append({"station": station, "layer": layer, "half": int(half), "count": count, "true": true, "precision": true / count})
    halves = {item["half"] for item in diagnostics}
    passed = (
        len(diagnostics) >= contract["minimum_supported_environments"]
        and (not contract["require_both_chronological_halves"] or halves == {0, 1})
        and all(item["true"] > contract["reject_if_any_supported_environment_true_positives_eq"] for item in diagnostics)
        and all(item["precision"] > contract["minimum_environment_precision_strictly_gt"] for item in diagnostics)
    )
    return {"passed": passed, "supported_environments": diagnostics}


def _select_transport(scores: np.ndarray, labels: np.ndarray, selection: dict[str, Any]) -> dict[str, Any]:
    result = _ORIGINAL_SELECT(scores, labels, selection)
    if _CURRENT_ENVIRONMENTS is None or len(_CURRENT_ENVIRONMENTS["half"]) != len(scores):
        raise RuntimeError("transport environment context unavailable")
    evaluated = []
    eligible = []
    contract = selection["transport_stability"]
    for candidate in result["candidates"]:
        stability = _transport_stability(scores, labels, float(candidate["threshold"]), _CURRENT_ENVIRONMENTS, contract)
        item = {**candidate, "transport_stability": stability}
        evaluated.append(item)
        if (
            candidate["count"] >= selection["minimum_additions"]
            and candidate["precision_lcb"] >= selection["wilson90_lcb_minimum"]
            and stability["passed"]
        ):
            eligible.append(item)
    eligible.sort(key=lambda item: (item["quantile"], item["precision_lcb"]), reverse=True)
    return {"candidates": evaluated, "chosen": eligible[0] if eligible else None, "transport_contract": contract}


def _write_v13(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        payload = dict(value)
        payload["schema_version"] = json.loads(CONFIG.read_text(encoding="utf-8"))["result_schema_version"]
    _ORIGINAL_WRITE(path, payload)


def _configure() -> None:
    core.EXPERIMENT_ID, core.CONFIG, core.ARTIFACT, core.LOCK = EXPERIMENT_ID, CONFIG, ARTIFACT, LOCK
    core.__file__ = str(Path(__file__).resolve())
    core.recurrence_laminar_features = visibility_topology_features
    core._write = _write_v13
    base._select = _select_transport


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
    value = preflight(args.data_dir) if args.preflight else qa(args.data_dir) if args.qa else execute(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
