"""Build the fail-closed post-run evidence bundle for the P1 MS-TCN/ASRF v2 run.

This helper is intentionally downstream-only.  It never reads training labels,
official test/sample/submission paths, checkpoints, or blind prediction arrays.
It consumes only the terminal result, aggregate metric receipts, and optimizer
histories already written by the sealed one-shot experiment.

The normal invocation is::

    python scripts/build_p1_mstcn_asrf_v2_postrun_bundle.py

Nothing is written unless the run has a complete Q3+Q4 terminal status and all
cross-file integrity checks pass.  ``--validate-only`` runs the same contract
checks without importing the plotting stack or writing output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_incumbent_preserving_mstcn_asrf_v2"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
OUTPUT_DIR = REPORT_DIR / "postrun_bundle"

TERMINAL_SCHEMA = "p1.mstcn_asrf.terminal.v2"
CONFIRMATORY_SCHEMA = "p1.mstcn_asrf.confirmatory_metrics.v1"
Q2_SCHEMA = "p1.mstcn_asrf.q2_selection.v1"
RECIPE_SCHEMA = "p1.mstcn_asrf.selected_recipe.v1"

COMPLETE_STATUSES = {
    "GO_HIGH_IMPACT_OFFICIAL_PROBE_ELIGIBLE_NOT_AUTHORIZED",
    "GO_RESEARCH_ONLY_NOT_OFFICIAL_PROBE_ELIGIBLE",
    "NO_GO_CONFIRMATORY",
}
EXPECTED_WIDTHS = (256, 512)
EXPECTED_SEEDS = (20260827, 20260839, 20260863)
EXPECTED_ROUTER_COUNTS = {"tp": 8961, "fp": 203, "fn": 1724}
EXPECTED_ROUTER_F1 = 17922.0 / 19849.0

BLUE = "#2E74B5"
DARK_BLUE = "#1F4D78"
GOLD = "#C48A1B"
INK = "#172B3A"
MUTED = "#596574"
LIGHT = "#E8EEF5"
GRID = "#D6DCE5"
WHITE = "#FFFFFF"


class PostrunContractError(RuntimeError):
    """Raised before any report output is committed."""


@dataclass(frozen=True)
class ValidatedInputs:
    terminal: dict[str, Any]
    confirmatory: dict[str, Any]
    q2_selection: dict[str, Any]
    recipe: dict[str, Any]
    histories: dict[str, list[dict[str, Any]]]
    history_stats: dict[str, dict[str, Any]]
    source_paths: tuple[Path, ...]
    source_sha256: dict[str, str]


def _fail(message: str) -> None:
    raise PostrunContractError(message)


def _reject_json_constant(value: str) -> None:
    _fail(f"non-standard/non-finite JSON constant: {value}")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        _fail(f"required input is absent: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except PostrunContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot read valid UTF-8 JSON from {path}: {error}")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be a JSON array")
    return value


def _require_key(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        _fail(f"{label} is missing required key {key!r}")
    return mapping[key]


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    return value


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        _fail(f"{label} must be finite")
    return output


def _as_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} must be boolean")
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        _fail(f"{label} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        _fail(f"{label} is not valid ISO-8601: {error}")
    if parsed.tzinfo is None:
        _fail(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _assert_sources_unchanged(inputs: ValidatedInputs) -> None:
    try:
        current = {_source_key(path): _sha256(path) for path in inputs.source_paths}
    except OSError as error:
        _fail(f"cannot re-hash source receipts: {error}")
    if current != inputs.source_sha256:
        _fail("one or more source receipts changed during post-run report construction")


def _require_equal(left: Any, right: Any, label: str) -> None:
    if _canonical_json_bytes(left) != _canonical_json_bytes(right):
        _fail(f"cross-file identity mismatch: {label}")


def _validate_binary_metrics(metrics: Any, label: str) -> dict[str, Any]:
    obj = _require_object(metrics, label)
    tp = _as_int(_require_key(obj, "tp", label), f"{label}.tp")
    fp = _as_int(_require_key(obj, "fp", label), f"{label}.fp")
    fn = _as_int(_require_key(obj, "fn", label), f"{label}.fn")
    if min(tp, fp, fn) < 0:
        _fail(f"{label} confusion counts must be non-negative")
    precision = _as_float(_require_key(obj, "precision", label), f"{label}.precision")
    recall = _as_float(_require_key(obj, "recall", label), f"{label}.recall")
    f1 = _as_float(_require_key(obj, "f1", label), f"{label}.f1")
    expected_precision = tp / (tp + fp) if tp + fp else 0.0
    expected_recall = tp / (tp + fn) if tp + fn else 0.0
    expected_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    for name, observed, expected in (
        ("precision", precision, expected_precision),
        ("recall", recall, expected_recall),
        ("f1", f1, expected_f1),
    ):
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12):
            _fail(f"{label}.{name} is inconsistent with TP/FP/FN")
    return obj


def _validate_checks(value: Any, label: str, result: str) -> dict[str, bool]:
    checks = _require_object(value, label)
    if not checks:
        _fail(f"{label} cannot be empty")
    for key, item in checks.items():
        _as_bool(item, f"{label}.{key}")
    expected = "PASS" if all(checks.values()) else "FAIL"
    if result != expected:
        _fail(f"{label} is inconsistent with declared result {result!r}")
    return checks


def _validate_monotone_union(
    anchor: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    added_rows: int,
    added_precision: float,
    label: str,
) -> None:
    """Reconcile aggregate counts for candidate = anchor OR proposal."""

    added_true_positive = int(candidate["tp"]) - int(anchor["tp"])
    added_false_positive = int(candidate["fp"]) - int(anchor["fp"])
    recovered_false_negative = int(anchor["fn"]) - int(candidate["fn"])
    if min(added_true_positive, added_false_positive, recovered_false_negative) < 0:
        _fail(f"{label} violates the incumbent-preserving monotone union")
    if recovered_false_negative != added_true_positive:
        _fail(f"{label} TP/FN changes are inconsistent")
    if added_true_positive + added_false_positive != added_rows:
        _fail(f"{label} added_rows is inconsistent with confusion-count changes")
    expected_precision = added_true_positive / added_rows if added_rows else 0.0
    if not math.isclose(added_precision, expected_precision, rel_tol=0.0, abs_tol=1.0e-12):
        _fail(f"{label} added_row_precision is inconsistent")


def _linear_slope(rows: Sequence[dict[str, Any]], tail: int) -> float:
    subset = rows[-min(tail, len(rows)) :]
    x = [float(row["epoch"]) for row in subset]
    y = [float(row["total_loss"]) for row in subset]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((item - x_mean) ** 2 for item in x)
    if denominator == 0.0:
        return 0.0
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True)) / denominator


def _history_statistics(
    rows: list[dict[str, Any]], *, expected_epochs: int, label: str
) -> dict[str, Any]:
    if len(rows) != expected_epochs:
        _fail(f"{label} has {len(rows)} epochs; expected {expected_epochs}")
    epochs: list[int] = []
    losses: list[float] = []
    wall_seconds: list[float] = []
    nonfinite_total = 0
    clipping_total = 0
    for index, row_value in enumerate(rows, start=1):
        row = _require_object(row_value, f"{label}[{index - 1}]")
        epoch = _as_int(_require_key(row, "epoch", label), f"{label}.epoch")
        if epoch != index:
            _fail(f"{label} epochs must be the exact sequence 1..{expected_epochs}")
        loss = _as_float(_require_key(row, "total_loss", label), f"{label}.total_loss")
        wall = _as_float(
            _require_key(row, "epoch_wall_seconds", label),
            f"{label}.epoch_wall_seconds",
        )
        nonfinite = _as_int(_require_key(row, "nonfinite_count", label), f"{label}.nonfinite_count")
        clipping = _as_int(
            _require_key(row, "gradient_clip_count", label),
            f"{label}.gradient_clip_count",
        )
        if loss < 0.0 or wall <= 0.0 or nonfinite < 0 or clipping < 0:
            _fail(f"{label} contains an invalid negative/non-positive diagnostic")
        epochs.append(epoch)
        losses.append(loss)
        wall_seconds.append(wall)
        nonfinite_total += nonfinite
        clipping_total += clipping
    minimum = min(losses)
    minimum_index = losses.index(minimum)
    return {
        "epochs": len(rows),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": minimum,
        "minimum_loss_epoch": epochs[minimum_index],
        "final_over_minimum_ratio": losses[-1] / minimum if minimum > 0.0 else None,
        "tail_25_epoch_loss_slope": _linear_slope(rows, 25),
        "tail_50_epoch_loss_slope": _linear_slope(rows, 50),
        "median_epoch_wall_seconds": statistics.median(wall_seconds),
        "total_epoch_wall_seconds": sum(wall_seconds),
        "nonfinite_count_total": nonfinite_total,
        "gradient_clip_count_total": clipping_total,
        "interpretation": "training-loss diagnostic only; no holdout convergence claim",
    }


def _validate_history_file(
    path: Path, *, expected_epochs: int, label: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_any = _require_list(_read_json(path), label)
    rows = [_require_object(item, f"{label} row") for item in rows_any]
    return rows, _history_statistics(rows, expected_epochs=expected_epochs, label=label)


def validate_inputs() -> ValidatedInputs:
    """Validate all aggregate receipts without reading predictions or labels."""

    terminal_path = ARTIFACT_DIR / "terminal_result.json"
    terminal = _require_object(_read_json(terminal_path), "terminal_result")
    if terminal.get("schema_version") != TERMINAL_SCHEMA:
        _fail("terminal_result schema is not the expected v2 schema")
    if terminal.get("experiment_id") != EXPERIMENT_ID:
        _fail("terminal_result experiment identity mismatch")
    status = terminal.get("status")
    if status not in COMPLETE_STATUSES:
        _fail(f"terminal status is absent or incomplete for a Q3+Q4 report: {status!r}")
    started = _parse_utc(terminal.get("started_at_utc"), "started_at_utc")
    completed = _parse_utc(terminal.get("completed_at_utc"), "completed_at_utc")
    if completed < started:
        _fail("terminal completion time precedes start time")
    for key in (
        "submission_created",
        "upload_performed",
        "official_three_point_gain_claimed",
    ):
        if _as_bool(_require_key(terminal, key, "terminal_result"), key):
            _fail(f"terminal_result unexpectedly declares {key}=true")

    confirm_path = ARTIFACT_DIR / "confirmatory_metrics.json"
    confirmatory = _require_object(_read_json(confirm_path), "confirmatory_metrics")
    if confirmatory.get("schema_version") != CONFIRMATORY_SCHEMA:
        _fail("confirmatory_metrics schema mismatch")
    _require_equal(
        terminal.get("confirmatory_metrics"),
        confirmatory,
        "terminal.confirmatory_metrics vs confirmatory_metrics.json",
    )
    if _as_bool(
        _require_key(confirmatory, "three_official_points_claimed", "confirmatory_metrics"),
        "confirmatory_metrics.three_official_points_claimed",
    ):
        _fail("confirmatory metrics cannot claim an official three-point gain")

    folds = _require_object(confirmatory.get("folds"), "confirmatory_metrics.folds")
    if set(folds) != {"q3", "q4"}:
        _fail("confirmatory fold inventory must be exactly q3 and q4")
    validated_folds: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for phase in ("q3", "q4"):
        fold = _require_object(folds[phase], f"folds.{phase}")
        anchor = _validate_binary_metrics(fold.get("anchor"), f"folds.{phase}.anchor")
        candidate = _validate_binary_metrics(fold.get("candidate"), f"folds.{phase}.candidate")
        delta = _as_float(fold.get("delta_f1"), f"folds.{phase}.delta_f1")
        if not math.isclose(
            delta,
            float(candidate["f1"]) - float(anchor["f1"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            _fail(f"folds.{phase}.delta_f1 is internally inconsistent")
        if (
            int(candidate["tp"]) < int(anchor["tp"])
            or int(candidate["fp"]) < int(anchor["fp"])
            or int(candidate["fn"]) > int(anchor["fn"])
        ):
            _fail(f"folds.{phase} violates the monotone Router-union invariant")
        validated_folds[phase] = (anchor, candidate)

    pooled = _require_object(confirmatory.get("pooled"), "confirmatory_metrics.pooled")
    pooled_anchor = _validate_binary_metrics(pooled.get("anchor"), "pooled.anchor")
    pooled_candidate = _validate_binary_metrics(pooled.get("candidate"), "pooled.candidate")
    for key, expected in EXPECTED_ROUTER_COUNTS.items():
        if int(pooled_anchor[key]) != expected:
            _fail(f"pooled current-Router {key} identity changed")
    if not math.isclose(float(pooled_anchor["f1"]), EXPECTED_ROUTER_F1, rel_tol=0.0, abs_tol=1e-12):
        _fail("pooled current-Router exact F1 identity changed")
    pooled_delta = _as_float(pooled.get("delta_f1"), "pooled.delta_f1")
    if not math.isclose(
        pooled_delta,
        float(pooled_candidate["f1"]) - float(pooled_anchor["f1"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        _fail("pooled.delta_f1 is internally inconsistent")
    if (
        _as_int(
            pooled.get("anchor_positive_removed_rows"),
            "pooled.anchor_positive_removed_rows",
        )
        != 0
    ):
        _fail("incumbent-preserving invariant was violated")
    added_rows = _as_int(pooled.get("added_rows"), "pooled.added_rows")
    added_precision = _as_float(pooled.get("added_row_precision"), "pooled.added_row_precision")
    if added_rows < 0 or not 0.0 <= added_precision <= 1.0:
        _fail("pooled addition diagnostics are outside their valid domain")
    pooled_rows = _as_int(pooled.get("rows"), "pooled.rows")
    if pooled_rows <= 0:
        _fail("pooled.rows must be positive")
    _validate_monotone_union(
        pooled_anchor,
        pooled_candidate,
        added_rows=added_rows,
        added_precision=added_precision,
        label="pooled",
    )
    for metric_name, pooled_metric in (
        ("anchor", pooled_anchor),
        ("candidate", pooled_candidate),
    ):
        index = 0 if metric_name == "anchor" else 1
        for count_name in ("tp", "fp", "fn"):
            fold_total = sum(
                int(validated_folds[phase][index][count_name]) for phase in ("q3", "q4")
            )
            if fold_total != int(pooled_metric[count_name]):
                _fail(f"Q3+Q4 {metric_name}.{count_name} does not equal pooled count")

    bootstrap = _require_object(confirmatory.get("bootstrap"), "confirmatory_metrics.bootstrap")
    lower = _as_float(bootstrap.get("ci90_lower"), "bootstrap.ci90_lower")
    mean = _as_float(bootstrap.get("delta_f1_mean"), "bootstrap.delta_f1_mean")
    upper = _as_float(bootstrap.get("ci90_upper"), "bootstrap.ci90_upper")
    if not lower <= mean <= upper:
        _fail("bootstrap CI ordering is invalid")
    if _as_int(bootstrap.get("replicates"), "bootstrap.replicates") != 10_000:
        _fail("bootstrap replicate count changed from the sealed contract")

    by_station = _require_object(confirmatory.get("by_station"), "confirmatory_metrics.by_station")
    if len(by_station) != 3:
        _fail("station inventory must contain exactly three stations")
    station_improved = 0
    validated_stations: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for station, station_value in by_station.items():
        station_obj = _require_object(station_value, f"by_station.{station}")
        anchor = _validate_binary_metrics(station_obj.get("anchor"), f"by_station.{station}.anchor")
        candidate = _validate_binary_metrics(
            station_obj.get("candidate"), f"by_station.{station}.candidate"
        )
        delta = _as_float(station_obj.get("delta_f1"), f"by_station.{station}.delta_f1")
        if not math.isclose(
            delta,
            float(candidate["f1"]) - float(anchor["f1"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail(f"by_station.{station}.delta_f1 is inconsistent")
        station_improved += int(delta > 0.0)
        validated_stations.append((anchor, candidate))
    if station_improved != _as_int(
        confirmatory.get("stations_improved"), "confirmatory_metrics.stations_improved"
    ):
        _fail("stations_improved is inconsistent with station-level deltas")
    for metric_name, pooled_metric in (
        ("anchor", pooled_anchor),
        ("candidate", pooled_candidate),
    ):
        index = 0 if metric_name == "anchor" else 1
        for count_name in ("tp", "fp", "fn"):
            station_total = sum(int(metrics[index][count_name]) for metrics in validated_stations)
            if station_total != int(pooled_metric[count_name]):
                _fail(f"station {metric_name}.{count_name} does not equal pooled count")

    research_result = confirmatory.get("research_result")
    high_result = confirmatory.get("high_impact_official_probe_result")
    if research_result not in {"PASS", "FAIL"} or high_result not in {"PASS", "FAIL"}:
        _fail("confirmatory gate results must be PASS or FAIL")
    if high_result == "PASS" and research_result != "PASS":
        _fail("high-impact PASS cannot coexist with research FAIL")
    _validate_checks(
        confirmatory.get("research_success_checks"),
        "research_success_checks",
        research_result,
    )
    _validate_checks(
        confirmatory.get("high_impact_official_probe_checks"),
        "high_impact_official_probe_checks",
        high_result,
    )
    expected_status = (
        "GO_HIGH_IMPACT_OFFICIAL_PROBE_ELIGIBLE_NOT_AUTHORIZED"
        if high_result == "PASS"
        else "GO_RESEARCH_ONLY_NOT_OFFICIAL_PROBE_ELIGIBLE"
        if research_result == "PASS"
        else "NO_GO_CONFIRMATORY"
    )
    if status != expected_status:
        _fail("terminal status is inconsistent with confirmatory gate results")

    q2_path = ARTIFACT_DIR / "q2_selection.json"
    q2_selection = _require_object(_read_json(q2_path), "q2_selection")
    if q2_selection.get("schema_version") != Q2_SCHEMA:
        _fail("q2_selection schema mismatch")
    if q2_selection.get("result") != "PASS":
        _fail("a complete confirmatory run requires Q2 selection PASS")

    recipe_path = ARTIFACT_DIR / "selected_recipe.json"
    recipe = _require_object(_read_json(recipe_path), "selected_recipe")
    if recipe.get("schema_version") != RECIPE_SCHEMA:
        _fail("selected_recipe schema mismatch")
    _require_equal(terminal.get("selected_recipe"), recipe, "terminal selected recipe")
    selected = _require_object(q2_selection.get("selected"), "q2_selection.selected")
    for key in ("width", "epoch", "threshold", "representation"):
        if selected.get(key) != recipe.get(key):
            _fail(f"Q2 selection and selected recipe disagree on {key}")
    seeds = tuple(_require_list(recipe.get("seeds"), "selected_recipe.seeds"))
    if seeds != EXPECTED_SEEDS:
        _fail("selected recipe seed inventory changed")
    convergence = _require_object(
        q2_selection.get("convergence_evidence"), "q2_selection.convergence_evidence"
    )
    widths = tuple(convergence.get("widths", []))
    if widths != EXPECTED_WIDTHS:
        _fail("Q2 capacity inventory changed")
    maximum_epoch = _as_int(convergence.get("maximum_epoch"), "maximum_epoch")
    selected_epoch = _as_int(recipe.get("epoch"), "selected_recipe.epoch")
    if not 1 <= selected_epoch <= maximum_epoch:
        _fail("selected epoch lies outside the Q2 training range")
    q2_metrics = _require_object(q2_selection.get("selected_metrics"), "selected_metrics")
    if q2_metrics.get("result") != "PASS":
        _fail("selected Q2 metrics must pass the continuation gate")
    _validate_checks(q2_metrics.get("gate_checks"), "q2 gate_checks", "PASS")
    q2_anchor = _validate_binary_metrics(q2_metrics.get("anchor"), "q2.anchor")
    q2_candidate = _validate_binary_metrics(q2_metrics.get("candidate"), "q2.candidate")
    q2_added_rows = _as_int(q2_metrics.get("added_rows"), "q2.added_rows")
    q2_added_precision = _as_float(q2_metrics.get("added_row_precision"), "q2.added_row_precision")
    _validate_monotone_union(
        q2_anchor,
        q2_candidate,
        added_rows=q2_added_rows,
        added_precision=q2_added_precision,
        label="q2 selected candidate",
    )
    q2_delta = _as_float(q2_metrics.get("delta_f1"), "q2.delta_f1")
    if not math.isclose(
        q2_delta,
        float(q2_candidate["f1"]) - float(q2_anchor["f1"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        _fail("q2.delta_f1 is inconsistent")
    _require_equal(selected.get("candidate"), q2_candidate, "Q2 selected candidate metrics")
    for key, expected in (
        ("delta_f1", q2_delta),
        ("added_rows", q2_added_rows),
        ("added_row_precision", q2_added_precision),
    ):
        observed = _as_float(selected.get(key), f"q2_selection.selected.{key}")
        if not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1.0e-12):
            _fail(f"Q2 selected record and selected_metrics disagree on {key}")

    grid = _require_list(q2_selection.get("grid_records"), "q2_selection.grid_records")
    epochs = tuple(convergence.get("epochs", []))
    thresholds = tuple(convergence.get("thresholds", []))
    expected_grid_rows = len(EXPECTED_WIDTHS) * len(epochs) * len(thresholds)
    if len(grid) != expected_grid_rows or q2_selection.get("grid_candidates") != len(grid):
        _fail("Q2 finite-grid cardinality is inconsistent")
    grid_keys: set[tuple[int, int, float]] = set()
    for index, record_value in enumerate(grid):
        record = _require_object(record_value, f"grid_records[{index}]")
        key = (
            _as_int(record.get("width"), f"grid_records[{index}].width"),
            _as_int(record.get("epoch"), f"grid_records[{index}].epoch"),
            _as_float(record.get("threshold"), f"grid_records[{index}].threshold"),
        )
        if key in grid_keys:
            _fail(f"duplicate Q2 grid coordinate: {key}")
        grid_keys.add(key)
        _validate_binary_metrics(record.get("candidate"), f"grid_records[{index}].candidate")
    expected_keys = {
        (int(width), int(epoch), float(threshold))
        for width in EXPECTED_WIDTHS
        for epoch in epochs
        for threshold in thresholds
    }
    if grid_keys != expected_keys:
        _fail("Q2 finite-grid coordinates are incomplete")

    histories: dict[str, list[dict[str, Any]]] = {}
    history_stats: dict[str, dict[str, Any]] = {}
    source_paths: list[Path] = [terminal_path, confirm_path, q2_path, recipe_path]
    for width in EXPECTED_WIDTHS:
        for seed in EXPECTED_SEEDS:
            key = f"q2_width_{width}_seed_{seed}"
            path = ARTIFACT_DIR / f"{key}_training_history.json"
            rows, stats = _validate_history_file(path, expected_epochs=maximum_epoch, label=key)
            histories[key] = rows
            history_stats[key] = stats
            source_paths.append(path)
    selected_width = _as_int(recipe.get("width"), "selected_recipe.width")
    for phase in ("q3", "q4"):
        for seed in EXPECTED_SEEDS:
            key = f"{phase}_width_{selected_width}_seed_{seed}"
            path = ARTIFACT_DIR / f"{key}_refit_history.json"
            rows, stats = _validate_history_file(path, expected_epochs=selected_epoch, label=key)
            histories[key] = rows
            history_stats[key] = stats
            source_paths.append(path)

    source_paths_tuple = tuple(source_paths)
    try:
        source_sha256 = {_source_key(path): _sha256(path) for path in source_paths_tuple}
    except OSError as error:
        _fail(f"cannot hash validated source receipts: {error}")
    return ValidatedInputs(
        terminal=terminal,
        confirmatory=confirmatory,
        q2_selection=q2_selection,
        recipe=recipe,
        histories=histories,
        history_stats=history_stats,
        source_paths=source_paths_tuple,
        source_sha256=source_sha256,
    )


def _decision_text(status: str) -> tuple[str, str]:
    if status == "GO_HIGH_IMPACT_OFFICIAL_PROBE_ELIGIBLE_NOT_AUTHORIZED":
        return (
            "강한 로컬 승격 기준을 통과했다.",
            "공식 probe 검토 대상이지만 제출은 승인되지 않았고 공식 +3점은 확인되지 않았다.",
        )
    if status == "GO_RESEARCH_ONLY_NOT_OFFICIAL_PROBE_ELIGIBLE":
        return (
            "확인 구간의 연구 성공 기준은 통과했지만 강한 공식 probe 기준에는 미달했다.",
            "구조 가설의 방향성 증거로만 보존하며 공식 제출 후보로 승격하지 않는다.",
        )
    return (
        "Q3·Q4 확인 기준을 통과하지 못했다.",
        "Q2 선택 성능과 무관하게 공식 제출 후보로 승격하지 않는다.",
    )


def _metric_projection(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {key: metric[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")}


def _q2_delta_envelope(q2_selection: Mapping[str, Any]) -> dict[int, dict[int, float]]:
    grouped: dict[int, dict[int, float]] = {width: {} for width in EXPECTED_WIDTHS}
    for record in q2_selection["grid_records"]:
        width = int(record["width"])
        epoch = int(record["epoch"])
        delta = float(record["delta_f1"])
        grouped[width][epoch] = max(grouped[width].get(epoch, -math.inf), delta)
    return grouped


def build_summary(inputs: ValidatedInputs) -> dict[str, Any]:
    terminal = inputs.terminal
    confirmatory = inputs.confirmatory
    q2 = inputs.q2_selection
    recipe = inputs.recipe
    pooled = confirmatory["pooled"]
    headline, interpretation = _decision_text(terminal["status"])
    started = _parse_utc(terminal["started_at_utc"], "started_at_utc")
    completed = _parse_utc(terminal["completed_at_utc"], "completed_at_utc")
    folds: dict[str, Any] = {}
    for phase in ("q3", "q4"):
        fold = confirmatory["folds"][phase]
        folds[phase] = {
            "anchor": _metric_projection(fold["anchor"]),
            "candidate": _metric_projection(fold["candidate"]),
            "delta_f1": fold["delta_f1"],
        }
    stations = {
        station: {
            "anchor_f1": value["anchor"]["f1"],
            "candidate_f1": value["candidate"]["f1"],
            "delta_f1": value["delta_f1"],
        }
        for station, value in sorted(confirmatory["by_station"].items())
    }
    source_hashes = dict(inputs.source_sha256)
    q2_envelope = _q2_delta_envelope(q2)
    selected_width = int(recipe["width"])
    selected_epoch = int(recipe["epoch"])
    selected_delta = float(q2["selected_metrics"]["delta_f1"])
    selected_width_epochs = sorted(q2_envelope[selected_width])
    selected_position = selected_width_epochs.index(selected_epoch)
    neighbor_epochs = [
        epoch
        for epoch in (
            selected_width_epochs[selected_position - 1] if selected_position > 0 else None,
            selected_width_epochs[selected_position + 1]
            if selected_position + 1 < len(selected_width_epochs)
            else None,
        )
        if epoch is not None
    ]
    neighbor_values = {str(epoch): q2_envelope[selected_width][epoch] for epoch in neighbor_epochs}
    best_neighbor_delta = max(neighbor_values.values()) if neighbor_values else None
    strict_local_peak = best_neighbor_delta is not None and selected_delta > best_neighbor_delta
    return {
        "schema_version": "p1.mstcn_asrf.postrun_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "terminal_status": terminal["status"],
        "headline": headline,
        "interpretation": interpretation,
        "claims": {
            "local_research_success": confirmatory["research_result"] == "PASS",
            "local_high_impact_official_probe_eligible": (
                confirmatory["high_impact_official_probe_result"] == "PASS"
            ),
            "official_submission_authorized": False,
            "official_three_point_gain_confirmed": False,
            "official_plus3_confirmation_threshold_f1": 0.930749,
            "wording_guard": (
                "로컬 확인 결과와 공식 +3점 확인을 동일시하지 않는다. "
                "공식 F1 0.930749 이상을 별도 승인·평가에서 관측하기 전에는 "
                "공식 +3점을 주장할 수 없다."
            ),
        },
        "run": {
            "started_at_utc": terminal["started_at_utc"],
            "completed_at_utc": terminal["completed_at_utc"],
            "elapsed_seconds": (completed - started).total_seconds(),
            "device": terminal.get("device"),
        },
        "selected_recipe": recipe,
        "convergence": {
            "contract_interpretation": recipe["convergence_assessment"],
            "q2_selected_epoch": recipe["epoch"],
            "q2_maximum_epoch": q2["convergence_evidence"]["maximum_epoch"],
            "right_censored_at_max_epoch": q2["convergence_evidence"][
                "right_censored_at_max_epoch"
            ],
            "training_history_diagnostics": inputs.history_stats,
            "claim_limit": (
                "optimizer training loss is diagnostic only; Q3/Q4 holdout metrics and "
                "the preregistered gates determine promotion"
            ),
        },
        "q2_selection_only": {
            "role": q2["role"],
            "anchor": _metric_projection(q2["selected_metrics"]["anchor"]),
            "candidate": _metric_projection(q2["selected_metrics"]["candidate"]),
            "delta_f1": q2["selected_metrics"]["delta_f1"],
            "added_rows": q2["selected_metrics"]["added_rows"],
            "added_row_precision": q2["selected_metrics"]["added_row_precision"],
            "long_event_recall_gain": q2["selected_metrics"]["long_event_recall_gain"],
            "normal_false_positive_row_ratio": q2["selected_metrics"][
                "normal_false_positive_row_ratio"
            ],
            "anchor_positive_removed_rows": q2["selected_metrics"]["anchor_positive_removed_rows"],
            "gate_checks": q2["selected_metrics"]["gate_checks"],
            "optimistic_peak_context": {
                "selected_width": selected_width,
                "selected_epoch": selected_epoch,
                "selected_delta_f1": selected_delta,
                "adjacent_checkpoint_best_delta_f1": neighbor_values,
                "selected_minus_best_adjacent_delta_f1": (
                    selected_delta - best_neighbor_delta
                    if best_neighbor_delta is not None
                    else None
                ),
                "strict_local_peak": strict_local_peak,
                "interpretation": (
                    "Q2 maximum-over-grid에서 선택된 고립된 낙관적 peak로 취급하며, "
                    "Q3/Q4 확인 없이 일반화 증거로 사용하지 않는다."
                ),
            },
        },
        "confirmatory": {
            "role": confirmatory["role"],
            "folds": folds,
            "pooled": {
                "rows": pooled["rows"],
                "anchor": _metric_projection(pooled["anchor"]),
                "candidate": _metric_projection(pooled["candidate"]),
                "delta_f1": pooled["delta_f1"],
                "added_rows": pooled["added_rows"],
                "added_row_precision": pooled["added_row_precision"],
                "anchor_positive_removed_rows": pooled["anchor_positive_removed_rows"],
            },
            "bootstrap": confirmatory["bootstrap"],
            "stations": stations,
            "stations_improved": confirmatory["stations_improved"],
            "research_success_checks": confirmatory["research_success_checks"],
            "research_result": confirmatory["research_result"],
            "high_impact_official_probe_checks": confirmatory["high_impact_official_probe_checks"],
            "high_impact_official_probe_result": confirmatory["high_impact_official_probe_result"],
        },
        "source_sha256": source_hashes,
        "planned_figures": [
            "figures/figure_01_training_loss_convergence.png",
            "figures/figure_02_q2_qualification_envelope.png",
            "figures/figure_03_confirmatory_effects_and_gates.png",
        ],
    }


def _pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f}%"


def _pass_text(value: bool) -> str:
    return "PASS" if value else "FAIL"


def build_markdown(summary: Mapping[str, Any]) -> str:
    recipe = summary["selected_recipe"]
    q2 = summary["q2_selection_only"]
    q2_peak = q2["optimistic_peak_context"]
    confirm = summary["confirmatory"]
    pooled = confirm["pooled"]
    bootstrap = confirm["bootstrap"]
    lines = [
        "# P1 incumbent-preserving MS-TCN++/ASRF v2 실행 결과",
        "",
        f"상태: **{summary['terminal_status']}**  ",
        f"결론: **{summary['headline']}** {summary['interpretation']}",
        "",
        "> 이 문서는 로컬 Q2/Q3/Q4 증거만 요약한다. 공식 제출은 생성·승인·업로드되지 않았다. 별도 승인된 공식 평가에서 F1 0.930749 이상을 관측하기 전에는 공식 +3점이 미확정이다.",
        "",
        "## 선택 사양과 수렴 해석",
        "",
        f"- width `{recipe['width']}`, batch `{recipe['batch_size']}`, epoch `{recipe['epoch']}`, high threshold `{recipe['threshold']}`",
        f"- seed: `{', '.join(str(item) for item in recipe['seeds'])}`; 표현: `{recipe['representation']}`",
        f"- 수렴 계약: `{summary['convergence']['contract_interpretation']}`",
        f"- Q2 선택 epoch `{summary['convergence']['q2_selected_epoch']}` / 최대 `{summary['convergence']['q2_maximum_epoch']}`. training loss는 진단용이며 holdout 수렴을 뜻하지 않는다.",
        "",
        "## Q2 선택 전용 결과",
        "",
        "| 기준 | Router | 후보 | 차이/보조 지표 |",
        "|---|---:|---:|---:|",
        f"| F1 | {q2['anchor']['f1']:.9f} | {q2['candidate']['f1']:.9f} | {q2['delta_f1']:+.9f} |",
        f"| 추가 행 precision | — | — | {q2['added_row_precision']:.6f} |",
        f"| long-event recall gain | — | — | {q2['long_event_recall_gain']:+.6f} |",
        f"| 정상 FP 비율 | — | — | {q2['normal_false_positive_row_ratio']:.6f}× |",
        f"| Router 양성 제거 | — | — | {q2['anchor_positive_removed_rows']}행 |",
        "",
        f"Q2 epoch `{q2_peak['selected_epoch']}`의 ΔF1 `{q2_peak['selected_delta_f1']:+.9f}`는 maximum-over-grid에서 선택된 **고립된 낙관적 peak**로 취급한다. 인접 checkpoint 최고 ΔF1 대비 차이는 `{q2_peak['selected_minus_best_adjacent_delta_f1']:+.9f}`이며, Q2는 사양 선택에만 사용하고 승격 증거에는 포함하지 않는다.",
        "",
        "## Q3·Q4 확인 결과",
        "",
        "| 구간 | Router F1 | 후보 F1 | ΔF1 |",
        "|---|---:|---:|---:|",
    ]
    for phase, label in (("q3", "Q3"), ("q4", "Q4")):
        fold = confirm["folds"][phase]
        lines.append(
            f"| {label} | {fold['anchor']['f1']:.9f} | {fold['candidate']['f1']:.9f} | {fold['delta_f1']:+.9f} |"
        )
    lines.extend(
        [
            f"| Pooled | {pooled['anchor']['f1']:.9f} | {pooled['candidate']['f1']:.9f} | {pooled['delta_f1']:+.9f} |",
            "",
            f"추가 `{pooled['added_rows']:,}`행의 precision은 `{pooled['added_row_precision']:.6f}`이고 Router 양성 제거는 `{pooled['anchor_positive_removed_rows']}`행이다. 21일 paired circular block bootstrap `{bootstrap['replicates']:,}`회의 ΔF1 평균은 `{bootstrap['delta_f1_mean']:+.9f}`, CI90은 `[{bootstrap['ci90_lower']:+.9f}, {bootstrap['ci90_upper']:+.9f}]`다.",
            "",
            "## Gate 판정",
            "",
            f"- 연구 성공: **{confirm['research_result']}**",
        ]
    )
    for key, value in confirm["research_success_checks"].items():
        lines.append(f"  - `{key}`: {_pass_text(value)}")
    lines.append(f"- 강한 공식 probe 검토 기준: **{confirm['high_impact_official_probe_result']}**")
    for key, value in confirm["high_impact_official_probe_checks"].items():
        lines.append(f"  - `{key}`: {_pass_text(value)}")
    lines.extend(
        [
            "",
            "## Station별 재현성",
            "",
            "| Station | Router F1 | 후보 F1 | ΔF1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for station, value in confirm["stations"].items():
        lines.append(
            f"| {station} | {value['anchor_f1']:.9f} | {value['candidate_f1']:.9f} | {value['delta_f1']:+.9f} |"
        )
    lines.extend(
        [
            "",
            f"개선 station은 `{confirm['stations_improved']}/3`이다.",
            "",
            "## Optimizer 이력 진단",
            "",
            "| Fit | Epoch | 최소 loss (epoch) | 마지막 loss | tail-25 slope | tail-50 slope | nonfinite |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fit, value in summary["convergence"]["training_history_diagnostics"].items():
        lines.append(
            f"| `{fit}` | {value['epochs']} | {value['minimum_loss']:.6g} ({value['minimum_loss_epoch']}) | {value['final_loss']:.6g} | {value['tail_25_epoch_loss_slope']:+.3e} | {value['tail_50_epoch_loss_slope']:+.3e} | {value['nonfinite_count_total']} |"
        )
    lines.extend(
        [
            "",
            "이 표는 optimizer training loss의 안정성·tail 형태만 진단한다. 일반화 성능과 holdout 수렴 판단은 Q3·Q4 metric과 사전 고정 gate를 따른다.",
            "",
            "## 해석 한계",
            "",
            "- Q2 epoch 125는 인접 checkpoint와 분리된 낙관적 최대점이므로 Q3·Q4 확인 결과와 분리해서 해석한다.",
            "- optimizer training loss는 일반화 성능이나 수렴을 직접 증명하지 않는다.",
            "- 로컬 F1과 공식 F1의 크기 운송은 보장되지 않는다. 공식 F1 0.930749 이상을 별도 승인된 공식 평가에서 관측하기 전에는 공식 +3점을 주장할 수 없다.",
            "- bootstrap은 날짜 표본 변동을 다루지만 Q2 다중선택과 전체 HPO 불확실성을 포함하지 않는다.",
            "",
            "## 시각 증거",
            "",
            "1. `figures/figure_01_training_loss_convergence.png` — Q2 두 용량 6개 곡선과 Q3·Q4 선택 용량 6개 곡선",
            "2. `figures/figure_02_q2_qualification_envelope.png` — Q2에서 각 epoch·width의 고정 threshold best ΔF1 envelope",
            "3. `figures/figure_03_confirmatory_effects_and_gates.png` — Q3·Q4·pooled 효과, CI90, station별 부호",
            "",
        ]
    )
    return "\n".join(lines)


def _configure_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        _fail(f"matplotlib is required only after terminal validation: {error}")
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )
    return plt


def _save_figure(fig: Any, path: Path, plt: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    if not path.is_file() or path.stat().st_size < 1_000:
        _fail(f"rendered figure is missing or implausibly small: {path}")
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        _fail(f"rendered figure does not have a PNG signature: {path}")


def _plot_training_convergence(inputs: ValidatedInputs, output: Path, plt: Any) -> None:
    # Decimal major ticks avoid a missing Unicode-minus glyph in the Korean
    # font fallback used by Matplotlib's default logarithmic mathtext labels.
    from matplotlib.ticker import FuncFormatter

    recipe = inputs.recipe
    selected_width = int(recipe["width"])
    selected_epoch = int(recipe["epoch"])
    panels = (
        ("q2", 256, "Q2 qualification · width 256"),
        ("q2", 512, "Q2 qualification · width 512"),
        ("q3", selected_width, f"Q3 fresh refit · width {selected_width}"),
        ("q4", selected_width, f"Q4 fresh refit · width {selected_width}"),
    )
    fig, axes_grid = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    axes = list(axes_grid.flat)
    styles = ((BLUE, "-"), (GOLD, "--"), (DARK_BLUE, ":"))
    for axis, (phase, width, title) in zip(axes, panels, strict=True):
        for seed, (color, style) in zip(EXPECTED_SEEDS, styles, strict=True):
            key = f"{phase}_width_{width}_seed_{seed}"
            rows = inputs.histories[key]
            axis.plot(
                [row["epoch"] for row in rows],
                [row["total_loss"] for row in rows],
                color=color,
                linestyle=style,
                linewidth=1.45,
                label=f"seed {seed}",
            )
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Training total loss (log)")
        if phase == "q2":
            axis.axvline(
                selected_epoch,
                color=INK,
                linewidth=1.0,
                linestyle="-.",
                label=f"selected epoch {selected_epoch}",
            )
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Q2 전체 용량과 Q3·Q4 refit의 seed별 학습 손실",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.965,
        "Q2 width 256/512 각 3 seed는 300 epoch, Q3·Q4 각 3 seed는 선택 epoch까지 새로 적합. 손실은 진단용이며 holdout 수렴 주장이 아님.",
        color=MUTED,
        fontsize=9.5,
    )
    _save_figure(fig, output, plt)


def _plot_q2_envelope(inputs: ValidatedInputs, output: Path, plt: Any) -> None:
    q2 = inputs.q2_selection
    selected = q2["selected"]
    grouped = _q2_delta_envelope(q2)
    fig, axis = plt.subplots(figsize=(11.5, 5.6), constrained_layout=False)
    for width, color, style in ((256, LIGHT, "--"), (512, BLUE, "-")):
        epochs = sorted(grouped[width])
        values = [grouped[width][epoch] for epoch in epochs]
        line_color = DARK_BLUE if width == 256 else color
        axis.plot(
            epochs,
            values,
            color=line_color,
            linestyle=style,
            linewidth=2.0,
            label=f"width {width}: epoch별 7 threshold 최고 ΔF1",
        )
    axis.axhline(0.0, color=MUTED, linestyle=":", linewidth=1.2, label="Router ΔF1 = 0")
    axis.scatter(
        [selected["epoch"]],
        [selected["delta_f1"]],
        s=75,
        color=GOLD,
        edgecolor=INK,
        linewidth=0.8,
        zorder=5,
        label=(
            f"selected: width {selected['width']}, epoch {selected['epoch']}, "
            f"threshold {selected['threshold']}"
        ),
    )
    axis.annotate(
        f"ΔF1 {selected['delta_f1']:+.6f}",
        (selected["epoch"], selected["delta_f1"]),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    fig.suptitle(
        "Q2 유한 탐색의 epoch별 best ΔF1",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.915,
        "각 점은 사전 고정된 7개 threshold 중 최고 ΔF1. epoch 125는 고립된 낙관적 peak로 취급하며 Q2는 selection-only임.",
        color=MUTED,
        fontsize=9.5,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Q2 best ΔF1 vs Router")
    axis.set_xlim(0, max(grouped[512]) + 5)
    axis.margins(y=0.10)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", fontsize=8.7)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.13, top=0.84)
    _save_figure(fig, output, plt)


def _plot_confirmatory_effects(inputs: ValidatedInputs, output: Path, plt: Any) -> None:
    confirm = inputs.confirmatory
    categories = ("Q3", "Q4", "Pooled")
    folds = confirm["folds"]
    anchors = [
        float(folds["q3"]["anchor"]["f1"]),
        float(folds["q4"]["anchor"]["f1"]),
        float(confirm["pooled"]["anchor"]["f1"]),
    ]
    candidates = [
        float(folds["q3"]["candidate"]["f1"]),
        float(folds["q4"]["candidate"]["f1"]),
        float(confirm["pooled"]["candidate"]["f1"]),
    ]
    deltas = [candidate - anchor for anchor, candidate in zip(anchors, candidates, strict=True)]
    thresholds = [0.010, 0.010, 0.027832]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 5.5),
        gridspec_kw={"width_ratios": [1.05, 1.15, 1.0]},
        constrained_layout=True,
    )
    x = list(range(len(categories)))
    bar_width = 0.34
    axes[0].bar(
        [item - bar_width / 2 for item in x],
        anchors,
        bar_width,
        color=LIGHT,
        edgecolor=DARK_BLUE,
        linewidth=1.0,
        label="Router",
    )
    axes[0].bar(
        [item + bar_width / 2 for item in x],
        candidates,
        bar_width,
        color=BLUE,
        edgecolor=DARK_BLUE,
        linewidth=1.0,
        label="Candidate",
    )
    for index, value in enumerate(anchors):
        axes[0].text(index - bar_width / 2, value + 0.012, f"{value:.4f}", ha="center", fontsize=8)
    for index, value in enumerate(candidates):
        axes[0].text(index + bar_width / 2, value + 0.012, f"{value:.4f}", ha="center", fontsize=8)
    axes[0].set_xticks(x, categories)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("F1")
    axes[0].set_title("확인 구간 F1", loc="left", fontsize=11, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8.5)

    y = list(range(len(categories)))
    axes[1].axvline(0.0, color=INK, linewidth=0.9)
    axes[1].scatter(deltas, y, color=BLUE, s=62, zorder=4, label="observed ΔF1")
    axes[1].scatter(
        thresholds,
        y,
        marker="D",
        facecolor=WHITE,
        edgecolor=GOLD,
        linewidth=1.5,
        s=55,
        zorder=4,
        label="high-impact threshold",
    )
    boot = confirm["bootstrap"]
    boot_mean = float(boot["delta_f1_mean"])
    boot_lower = float(boot["ci90_lower"])
    boot_upper = float(boot["ci90_upper"])
    axes[1].errorbar(
        [boot_mean],
        [2],
        xerr=[[boot_mean - boot_lower], [boot_upper - boot_mean]],
        fmt="none",
        ecolor=INK,
        elinewidth=2.0,
        capsize=4,
        label="pooled bootstrap CI90",
        zorder=3,
    )
    for index, value in enumerate(deltas):
        axes[1].annotate(
            f"{value:+.5f}",
            (value, index),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
        )
    axes[1].set_yticks(y, categories)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ΔF1 vs Router")
    axes[1].set_title("효과와 사전 고정 문턱", loc="left", fontsize=11, fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=7.9)

    station_rows = sorted(
        ((station, float(value["delta_f1"])) for station, value in confirm["by_station"].items()),
        key=lambda item: item[1],
    )
    station_names = [item[0] for item in station_rows]
    station_deltas = [item[1] for item in station_rows]
    colors = [BLUE if value >= 0.0 else WHITE for value in station_deltas]
    bars = axes[2].barh(
        station_names,
        station_deltas,
        color=colors,
        edgecolor=DARK_BLUE,
        linewidth=1.2,
    )
    axes[2].axvline(0.0, color=INK, linewidth=0.9)
    station_min = min([0.0, *station_deltas])
    station_max = max([0.0, *station_deltas])
    station_span = max(station_max - station_min, 0.01)
    axes[2].set_xlim(
        station_min - station_span * 0.18,
        station_max + station_span * 0.28,
    )
    for bar, value in zip(bars, station_deltas, strict=True):
        axes[2].annotate(
            f"{value:+.5f}",
            (value, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
        )
    axes[2].set_xlabel("ΔF1 vs Router")
    axes[2].set_title("Station별 효과", loc="left", fontsize=11, fontweight="bold")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Q3·Q4 확인 효과와 승격 기준",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.94,
        "Q2 제외 확인 결과. Pooled CI90은 21일 paired circular block bootstrap 10,000회; 공식 +3점 주장이 아님.",
        color=MUTED,
        fontsize=9.5,
    )
    _save_figure(fig, output, plt)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def build_bundle(inputs: ValidatedInputs) -> dict[str, Any]:
    _assert_sources_unchanged(inputs)
    summary = build_summary(inputs)
    markdown = build_markdown(summary)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    allowed_existing = {
        "result_summary.json",
        "result_summary.md",
        "bundle_manifest.json",
        "figures/figure_01_training_loss_convergence.png",
        "figures/figure_02_q2_qualification_envelope.png",
        "figures/figure_03_confirmatory_effects_and_gates.png",
    }
    unexpected_existing = sorted(
        str(path.relative_to(OUTPUT_DIR)).replace("\\", "/")
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT_DIR)).replace("\\", "/") not in allowed_existing
    )
    if unexpected_existing:
        _fail(
            "postrun output directory contains unexpected files; refusing to mix bundles: "
            + ", ".join(unexpected_existing)
        )
    plt = _configure_matplotlib()
    with tempfile.TemporaryDirectory(prefix=".postrun_stage_", dir=REPORT_DIR) as temp:
        stage = Path(temp)
        figure_dir = stage / "figures"
        _plot_training_convergence(
            inputs, figure_dir / "figure_01_training_loss_convergence.png", plt
        )
        _plot_q2_envelope(inputs, figure_dir / "figure_02_q2_qualification_envelope.png", plt)
        _plot_confirmatory_effects(
            inputs, figure_dir / "figure_03_confirmatory_effects_and_gates.png", plt
        )
        _write_json(stage / "result_summary.json", summary)
        _write_text(stage / "result_summary.md", markdown)

        _assert_sources_unchanged(inputs)

        staged_files = sorted(path for path in stage.rglob("*") if path.is_file())
        for source in staged_files:
            relative = source.relative_to(stage)
            destination = OUTPUT_DIR / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)

    output_files = sorted(
        path
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    )
    _assert_sources_unchanged(inputs)
    manifest = {
        "schema_version": "p1.mstcn_asrf.postrun_bundle_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "terminal_sha256": _sha256(ARTIFACT_DIR / "terminal_result.json"),
        "complete_terminal_status": inputs.terminal["status"],
        "source_sha256": summary["source_sha256"],
        "outputs": {
            str(path.relative_to(OUTPUT_DIR)).replace("\\", "/"): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        },
        "output_count_excluding_manifest": len(output_files),
        "official_test_sample_submission_accessed": False,
        "submission_created": False,
        "upload_performed": False,
    }
    expected_outputs = {
        "result_summary.json",
        "result_summary.md",
        "figures/figure_01_training_loss_convergence.png",
        "figures/figure_02_q2_qualification_envelope.png",
        "figures/figure_03_confirmatory_effects_and_gates.png",
    }
    if set(manifest["outputs"]) != expected_outputs:
        _fail("staged output inventory is not the exact five-file contract")
    manifest_path = OUTPUT_DIR / "bundle_manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=OUTPUT_DIR,
        prefix=".manifest_",
        suffix=".json.tmp",
    ) as handle:
        json.dump(
            manifest,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        temp_manifest = Path(handle.name)
    os.replace(temp_manifest, manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed aggregate-only report bundle after P1 terminal completion."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate terminal, receipts, and histories without writing report outputs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = validate_inputs()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "mode": "validate_only_no_outputs_written",
                    "terminal_status": inputs.terminal["status"],
                    "terminal_sha256": _sha256(ARTIFACT_DIR / "terminal_result.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    manifest = build_bundle(inputs)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PostrunContractError as error:
        print(f"POSTRUN_FAIL_CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
