"""Pure-stdlib CSV boundary for the P2 Stage-B parser correction r1.

This module is deliberately safe to import before the one-shot execution lock:
it imports no numerical or model package, performs no fit or prediction, and
retains no observation values.  Its full-source audit exercises the exact field
decoder used by the late-imported execution engine while preserving the
fold-local target firewall.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

EXPECTED_COLUMNS = (
    "station",
    "year",
    "layer",
    "time",
    "temp",
    "psal",
    "depth",
    "nominal_depth",
)
PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)
ALL_LAYERS = frozenset((*PUBLIC_LAYERS, *TARGET_LAYERS))
KST = ZoneInfo("Asia/Seoul")


class StageBCSVError(ValueError):
    """Raised when the pinned Stage-B CSV boundary changes."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_field_spans(
    raw_line: bytes,
    *,
    expected_fields: int,
) -> tuple[bytes, list[tuple[int, int]]]:
    """Locate one RFC-4180 record's fields without decoding their contents."""

    line = raw_line[:-2] if raw_line.endswith(b"\r\n") else raw_line.rstrip(b"\n")
    if b"\r" in line or b"\n" in line:
        raise StageBCSVError("P2 CSV contains an unsupported embedded newline")
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    in_quotes = False
    while index < len(line):
        value = line[index]
        if value == 34:
            if in_quotes and index + 1 < len(line) and line[index + 1] == 34:
                index += 2
                continue
            in_quotes = not in_quotes
        elif value == 44 and not in_quotes:
            spans.append((start, index))
            start = index + 1
        index += 1
    if in_quotes:
        raise StageBCSVError("P2 CSV contains an unterminated quoted field")
    spans.append((start, len(line)))
    if len(spans) != int(expected_fields):
        raise StageBCSVError("P2 CSV row width changed")
    return line, spans


def decode_csv_field(raw_line: bytes, span: tuple[int, int]) -> str:
    """Decode exactly one selected field, including a valid unquoted empty field."""

    start, stop = span
    token = raw_line[start:stop].decode("utf-8")
    if token == "":
        # ``csv.reader([""])`` treats the isolated token as an empty record and
        # returns ``[]``.  Here the span itself proves that one field exists, so
        # its correct scalar value is the empty string.
        return ""
    parsed = next(csv.reader([token], strict=True), None)
    if parsed is None or len(parsed) != 1:
        raise StageBCSVError("selected P2 CSV field did not decode to one scalar")
    return parsed[0]


def _numeric(value: str) -> float:
    return float(value) if value else float("nan")


def _parse_one_fold(
    observations_path: Path,
    *,
    outer_start_kst: str,
    embargo_days: int,
    expected_rows: int,
) -> dict[str, Any]:
    outer_start = datetime.fromisoformat(outer_start_kst)
    if outer_start.utcoffset() != KST.utcoffset(outer_start):
        raise StageBCSVError("outer start is not KST")
    cutoff = outer_start - timedelta(days=int(embargo_days))
    rows = 0
    public_rows = 0
    allowed_target_rows = 0
    withheld_target_rows = 0
    successful_scalar_returns = 0
    numeric_calls = 0
    unquoted_empty_scalar_count = 0
    per_layer_rows = {str(layer): 0 for layer in sorted(ALL_LAYERS)}

    with observations_path.open("rb") as stream:
        raw_header = stream.readline()
        if not raw_header:
            raise StageBCSVError("observations.csv is empty")
        header_line, header_spans = csv_field_spans(
            raw_header,
            expected_fields=len(EXPECTED_COLUMNS),
        )
        header = [decode_csv_field(header_line, span) for span in header_spans]
        if tuple(header) != EXPECTED_COLUMNS:
            raise StageBCSVError("observations.csv schema changed")

        for row_number, raw_row in enumerate(stream, 2):
            try:
                row_line, spans = csv_field_spans(
                    raw_row,
                    expected_fields=len(EXPECTED_COLUMNS),
                )
                station = decode_csv_field(row_line, spans[0])
                year_text = decode_csv_field(row_line, spans[1])
                layer_text = decode_csv_field(row_line, spans[2])
                time_text = decode_csv_field(row_line, spans[3])
                successful_scalar_returns += 4
                if not station:
                    raise StageBCSVError("observation station is empty")
                int(year_text)
                layer = int(layer_text)
                if layer not in ALL_LAYERS:
                    raise StageBCSVError("blind observation layer is outside the pinned 1..8 set")
                if not time_text.endswith("+09:00"):
                    raise StageBCSVError("blind observation timestamp lost its KST offset")
                keyed_time = datetime.fromisoformat(time_text)
                if keyed_time.utcoffset() != KST.utcoffset(keyed_time):
                    raise StageBCSVError("blind observation timestamp lost its KST offset")

                public_layer = layer in PUBLIC_LAYERS
                time_safe_target = layer in TARGET_LAYERS and keyed_time < cutoff
                if public_layer or time_safe_target:
                    for index in (4, 5):
                        value = decode_csv_field(row_line, spans[index])
                        successful_scalar_returns += 1
                        unquoted_empty_scalar_count += int(spans[index][0] == spans[index][1])
                        _numeric(value)
                        numeric_calls += 1
                    if public_layer:
                        public_rows += 1
                    else:
                        allowed_target_rows += 1
                else:
                    # Fields 4 and 5 are intentionally not sliced, decoded,
                    # converted, compared, hashed, or retained for this fold.
                    withheld_target_rows += 1

                for index in (6, 7):
                    value = decode_csv_field(row_line, spans[index])
                    successful_scalar_returns += 1
                    unquoted_empty_scalar_count += int(spans[index][0] == spans[index][1])
                    _numeric(value)
                    numeric_calls += 1
                per_layer_rows[str(layer)] += 1
                rows += 1
            except (UnicodeDecodeError, ValueError, StageBCSVError) as exc:
                raise StageBCSVError(
                    f"pinned observations parser failed at aggregate line number {row_number}"
                ) from exc

    if rows != int(expected_rows):
        raise StageBCSVError("pinned observations row count changed")
    if set(layer for layer, count in per_layer_rows.items() if count) != {
        str(layer) for layer in ALL_LAYERS
    }:
        raise StageBCSVError("pinned observations no longer contain every registered layer")
    if public_rows + allowed_target_rows + withheld_target_rows != rows:
        raise StageBCSVError("fold-local row accounting changed")
    return {
        "outer_start_kst": outer_start.isoformat(),
        "cutoff_kst_exclusive": cutoff.isoformat(),
        "rows": rows,
        "public_rows": public_rows,
        "allowed_training_target_rows": allowed_target_rows,
        "withheld_target_rows": withheld_target_rows,
        "successful_data_scalar_returns": successful_scalar_returns,
        "numeric_calls_completed": numeric_calls,
        "unquoted_empty_selected_scalar_count": unquoted_empty_scalar_count,
        "registered_layers_present": sorted(int(layer) for layer in per_layer_rows),
        "public_layers_loaded_at_all_times": list(PUBLIC_LAYERS),
        "target_layers_loaded_only_before_cutoff": list(TARGET_LAYERS),
        "withheld_target_scalar_fields_decoded": 0,
        "withheld_target_scalar_fields_converted": 0,
        "withheld_target_scalar_fields_used": 0,
        "raw_values_retained": 0,
    }


def full_pinned_source_parser_preflight(
    data_dir: Path,
    *,
    source_sha256: str,
    source_bytes: int,
    expected_rows: int,
    outer_folds: Sequence[Mapping[str, Any]],
    embargo_days: int,
) -> dict[str, Any]:
    """Exercise the exact selective parser across every fold before lock use."""

    directory = data_dir.resolve(strict=True)
    observations = (directory / "observations.csv").resolve(strict=True)
    if observations.parent != directory:
        raise PermissionError("observations.csv escaped the pinned P2 data directory")
    if observations.stat().st_size != int(source_bytes):
        raise StageBCSVError("pinned observations byte count changed")
    observed_sha = sha256_file(observations)
    if observed_sha != source_sha256:
        raise StageBCSVError("pinned observations SHA-256 changed")
    audits = {
        str(fold["name"]): _parse_one_fold(
            observations,
            outer_start_kst=str(fold["start_kst"]),
            embargo_days=int(embargo_days),
            expected_rows=int(expected_rows),
        )
        for fold in outer_folds
    }
    if not audits or any(
        audit["withheld_target_scalar_fields_decoded"] != 0
        or audit["withheld_target_scalar_fields_converted"] != 0
        or audit["withheld_target_scalar_fields_used"] != 0
        for audit in audits.values()
    ):
        raise StageBCSVError("parser preflight violated the withheld-target firewall")
    return {
        "status": "PASS_FULL_PINNED_SOURCE_SELECTIVE_PARSER",
        "source": {
            "filename": "observations.csv",
            "bytes": observations.stat().st_size,
            "sha256": observed_sha,
        },
        "parser_imports_numerical_or_model_modules": False,
        "folds": audits,
        "withheld_target_scalar_fields_decoded": 0,
        "withheld_target_scalar_fields_converted": 0,
        "withheld_target_scalar_fields_used": 0,
        "fits": 0,
        "predictions": 0,
        "scores": 0,
        "files_written": 0,
        "uploads": 0,
    }


__all__ = [
    "ALL_LAYERS",
    "EXPECTED_COLUMNS",
    "PUBLIC_LAYERS",
    "StageBCSVError",
    "TARGET_LAYERS",
    "csv_field_spans",
    "decode_csv_field",
    "full_pinned_source_parser_preflight",
    "sha256_file",
]
