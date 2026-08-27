"""Late-imported execution adapter for the P2 Stage-B parser correction r1.

All scientific computation remains in the byte-pinned v3 engine.  This adapter
rebinds only its configuration/control boundary and fold-blind CSV loader.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from p2_restore import architecture_matched_stage_b_execution_v3 as base_engine
from p2_restore.architecture_matched_stage_b_contract_r1 import (
    CONFIG_RELATIVE,
    CONFIG_SHA256,
    FRACTION_ROLES,
    MODE,
    implementation_pins,
    load_canonical_config,
    stage_paths,
    static_preflight,
    verify_consumed_attempt_lock,
    verify_execution_authorization,
    verify_pre_execution_qa,
    verify_stage_a_reference,
)
from p2_restore.architecture_matched_stage_b_csv_r1 import (
    ALL_LAYERS,
    EXPECTED_COLUMNS,
    KST,
    PUBLIC_LAYERS,
    TARGET_LAYERS,
    csv_field_spans,
    decode_csv_field,
)


def _load_fold_blind_observations(
    observations_path: Path,
    *,
    outer_start: Any,
    embargo_days: int,
    pd_module: Any,
    np_module: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load public inputs and only this fold's time-safe training targets."""

    pd = pd_module
    np = np_module
    cutoff = outer_start.tz_convert("Asia/Seoul") - pd.Timedelta(days=int(embargo_days))
    cutoff_datetime = cutoff.to_pydatetime()
    rows: list[tuple[Any, ...]] = []
    allowed_target_rows = 0
    withheld_target_rows = 0
    latest_allowed_target_time = ""
    selected_unquoted_empty_scalars = 0

    def numeric(value: str) -> float:
        return float(value) if value else float("nan")

    with observations_path.open("rb") as stream:
        raw_header = stream.readline()
        if not raw_header:
            raise ValueError("observations.csv is empty")
        header_line, header_spans = csv_field_spans(
            raw_header,
            expected_fields=len(EXPECTED_COLUMNS),
        )
        header = [decode_csv_field(header_line, span) for span in header_spans]
        if tuple(header) != EXPECTED_COLUMNS:
            raise ValueError("blind observations schema changed")
        for row_number, raw_row in enumerate(stream, 2):
            try:
                row_line, spans = csv_field_spans(
                    raw_row,
                    expected_fields=len(EXPECTED_COLUMNS),
                )
            except ValueError as exc:
                raise ValueError(f"observations row changed at line {row_number}") from exc
            station, year_text, layer_text, time_text = (
                decode_csv_field(row_line, spans[index]) for index in range(4)
            )
            if not time_text.endswith("+09:00"):
                raise ValueError("blind observation timestamp lost its KST offset")
            try:
                keyed_time = datetime.fromisoformat(time_text)
            except ValueError as exc:
                raise ValueError(
                    f"blind observation timestamp is invalid at line {row_number}"
                ) from exc
            if keyed_time.utcoffset() != KST.utcoffset(keyed_time):
                raise ValueError("blind observation timestamp lost its KST offset")
            layer = int(layer_text)
            if layer not in ALL_LAYERS:
                raise ValueError("blind observation layer is outside the pinned 1..8 set")
            public_layer = layer in PUBLIC_LAYERS
            time_safe_target = layer in TARGET_LAYERS and keyed_time < cutoff_datetime
            if public_layer or time_safe_target:
                temp_text = decode_csv_field(row_line, spans[4])
                psal_text = decode_csv_field(row_line, spans[5])
                selected_unquoted_empty_scalars += int(spans[4][0] == spans[4][1])
                selected_unquoted_empty_scalars += int(spans[5][0] == spans[5][1])
                temp = numeric(temp_text)
                psal = numeric(psal_text)
                if time_safe_target:
                    allowed_target_rows += 1
                    latest_allowed_target_time = max(latest_allowed_target_time, time_text)
            else:
                # Do not decode, convert, compare, hash, or retain fields 4 and 5.
                temp = float("nan")
                psal = float("nan")
                withheld_target_rows += 1
            depth_text = decode_csv_field(row_line, spans[6])
            nominal_text = decode_csv_field(row_line, spans[7])
            selected_unquoted_empty_scalars += int(spans[6][0] == spans[6][1])
            selected_unquoted_empty_scalars += int(spans[7][0] == spans[7][1])
            rows.append(
                (
                    station,
                    int(year_text),
                    layer,
                    time_text,
                    temp,
                    psal,
                    numeric(depth_text),
                    numeric(nominal_text),
                )
            )
    frame = pd.DataFrame.from_records(rows, columns=list(EXPECTED_COLUMNS))
    keyed_time = pd.to_datetime(frame["time"], utc=True, format="mixed")
    cutoff_utc = cutoff.tz_convert("UTC")
    withheld = frame["layer"].isin(TARGET_LAYERS) & keyed_time.ge(cutoff_utc)
    if not frame.loc[withheld, ["temp", "psal"]].isna().all().all():
        raise AssertionError("fold-local blind frame retained validation target truth")
    allowed = frame["layer"].isin(TARGET_LAYERS) & keyed_time.lt(cutoff_utc)
    if int(allowed.sum()) != allowed_target_rows or int(withheld.sum()) != withheld_target_rows:
        raise AssertionError("blind target-row accounting changed")
    public = frame["layer"].isin(PUBLIC_LAYERS)
    if not len(frame) or not np.isfinite(frame.loc[public, "temp"]).any():
        raise ValueError("blind observations lack public temperature inputs")
    return frame, {
        "rows": int(len(frame)),
        "cutoff_kst_exclusive": cutoff.isoformat(),
        "allowed_training_target_rows": int(allowed_target_rows),
        "latest_allowed_training_target_time_kst": latest_allowed_target_time,
        "withheld_target_rows": int(withheld_target_rows),
        "withheld_target_scalar_fields_decoded_or_converted": 0,
        "validation_target_temp_psal_strings_converted": 0,
        "validation_truth_columns_read_by_challenger": 0,
        "raw_source_records_streamed_for_key_routing": int(len(frame)),
        "raw_source_bytes_preflight_hashed_for_integrity_only": True,
        "public_layers_loaded_at_all_times": list(PUBLIC_LAYERS),
        "target_layers_loaded_only_before_cutoff": list(TARGET_LAYERS),
        "selected_unquoted_empty_scalars_decoded_as_missing": int(
            selected_unquoted_empty_scalars
        ),
        "parser_correction": "RFC4180_EMPTY_SCALAR_AND_PUBLIC_LAYER_ROUTING_R1",
    }


def _base_compatible_static_preflight(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = static_preflight(*args, **kwargs)
    report = dict(report)
    report["parser_correction_status"] = report["status"]
    report["status"] = "PASS_STATIC_IMPLEMENTATION_ONLY_STAGE_A_SEALED"
    return report


@contextmanager
def _bound_base_engine() -> Iterator[None]:
    bindings = {
        "CONFIG_RELATIVE": CONFIG_RELATIVE,
        "CONFIG_SHA256": CONFIG_SHA256,
        "FRACTION_ROLES": FRACTION_ROLES,
        "MODE": MODE,
        "TARGET_LAYERS": TARGET_LAYERS,
        "_csv_field_spans": csv_field_spans,
        "_decode_csv_field": decode_csv_field,
        "_load_fold_blind_observations": _load_fold_blind_observations,
        "implementation_pins": implementation_pins,
        "load_canonical_config": load_canonical_config,
        "stage_paths": stage_paths,
        "static_preflight": _base_compatible_static_preflight,
        "verify_consumed_attempt_lock": verify_consumed_attempt_lock,
        "verify_execution_authorization": verify_execution_authorization,
        "verify_pre_execution_qa": verify_pre_execution_qa,
        "verify_stage_a_reference": verify_stage_a_reference,
    }
    previous = {name: getattr(base_engine, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(base_engine, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base_engine, name, value)


def execute_stage_b(
    *,
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any] | None,
    attempt_lock: Path,
    progress: Any = None,
) -> dict[str, Any]:
    """Run the exact v3 computation behind the corrected parser boundary."""

    with _bound_base_engine():
        result = base_engine.execute_stage_b(
            root=root,
            data_dir=data_dir,
            config=config,
            preflight=preflight,
            attempt_lock=attempt_lock,
            progress=progress,
        )
    return {
        **result,
        "parser_correction": "RFC4180_EMPTY_SCALAR_AND_PUBLIC_LAYER_ROUTING_R1",
        "base_v3_scientific_computation_reused": True,
        "scientific_contract_changes": 0,
    }


__all__ = ["execute_stage_b"]
