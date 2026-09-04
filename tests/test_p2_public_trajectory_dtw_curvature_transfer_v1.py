from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from p2_restore import public_trajectory_dtw as dtw


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/run_p2_public_trajectory_dtw_curvature_transfer_v1.py"
DESIGN_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v1_design.json"
DESIGN_SHA256 = "341b41b79f867208de0d1494d3ea6c45108b648e87f6c347178de955897779fb"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p2_dtw_runner_under_test", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _small_score_frame(*, delta: float = 0.0, days: int = 2) -> pd.DataFrame:
    times = pd.date_range("2024-01-01T00:00:00+09:00", periods=days, freq="1D")
    rows = []
    for time_value in times:
        for layer in dtw.TARGET_LAYERS:
            truth = float(layer)
            anchor = truth + 1.0
            rows.append(
                {
                    "time": time_value,
                    "layer": layer,
                    "truth": truth,
                    "anchor": anchor,
                    "candidate": anchor + delta,
                    "support": True,
                    "dtw_distance": 0.1,
                }
            )
    return pd.DataFrame(rows)


def _exact_no_go_frame() -> pd.DataFrame:
    times = pd.date_range(
        "2024-09-01T00:00:00+09:00",
        "2024-10-31T23:50:00+09:00",
        freq="10min",
    )
    rows = []
    for layer in dtw.TARGET_LAYERS:
        truth = np.full(len(times), float(layer))
        anchor = truth + 0.5
        rows.append(
            pd.DataFrame(
                {
                    "time": times,
                    "layer": layer,
                    "truth": truth,
                    "anchor": anchor,
                    "candidate": anchor,
                    "support": True,
                    "dtw_distance": 0.1,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _patch_attempt_paths(monkeypatch: pytest.MonkeyPatch, runner, tmp_path: Path) -> None:
    seal = tmp_path / "preexecution_seal.json"
    seal.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "SEAL_PATH", seal)
    monkeypatch.setattr(runner, "CLAIM_PATH", tmp_path / "claims" / "attempt.claim.json")
    monkeypatch.setattr(runner, "JOURNAL_PATH", tmp_path / "journals" / "attempt.ndjson")
    monkeypatch.setattr(runner, "OOB_PATH", tmp_path / "receipts" / "terminal.json")
    monkeypatch.setattr(runner, "FINAL_DIR", tmp_path / "final")
    monkeypatch.setattr(runner, "STAGING_ROOT", tmp_path / "staging")


def test_design_hash_is_immutable_and_not_authorized() -> None:
    assert _sha256(DESIGN_PATH) == DESIGN_SHA256
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    assert design["schema"] == "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
    assert design["status"] == "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
    assert design["fixed_search_and_budget"]["model_fit_calls"] == 0


def test_exactly_six_cells_and_fixed_materialization_graph() -> None:
    assert [(cell.context_days, cell.neighbors) for cell in dtw.CELLS] == [
        (1, 3),
        (1, 7),
        (3, 3),
        (3, 7),
        (7, 3),
        (7, 7),
    ]
    slots = dtw.materialization_slots()
    assert len(slots) == 22
    assert [row["slot"] for row in slots] == list(range(1, 23))
    assert sum(row["stage"] == "INNER" for row in slots) == 18
    assert sum(row["stage"] == "EXACT" for row in slots) == 1
    assert sum(row["stage"] == "P100" for row in slots) == 3
    assert all(row["conditional"] for row in slots[-3:])


def test_utc_ns_adapter_avoids_microsecond_integer_collapse() -> None:
    values = pd.Series(
        pd.date_range("2024-09-01T00:00:00+09:00", periods=20, freq="10min")
    ).astype("datetime64[us, UTC]")
    ns = dtw.normalize_utc_ns(values)
    restored = dtw.utc_ns_to_datetime(ns)
    assert str(restored.dtype) == "datetime64[ns, UTC]"
    assert np.all(np.diff(ns) == 600_000_000_000)
    assert restored.equals(pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns"))
    assert restored.year.min() == 2024
    assert not restored.isna().any()


def test_exact_sep_oct_time_contract_is_61_kst_days_and_roundtrips() -> None:
    unique = pd.date_range(
        "2024-09-01T00:00:00+09:00",
        "2024-10-31T23:50:00+09:00",
        freq="10min",
    )
    repeated = np.repeat(unique.to_numpy(), 3)
    contract = dtw.exact_time_contract(repeated)
    assert contract == {
        "minimum_kst": "2024-09-01T00:00:00+09:00",
        "maximum_kst": "2024-10-31T23:50:00+09:00",
        "kst_days": 61,
        "unique_timestamps": 8784,
        "unit": "ns",
        "roundtrip_identity": True,
        "nat_count": 0,
    }


def test_align_on_utc_ns_preserves_full_range_and_identity() -> None:
    times = pd.date_range("2024-09-01T00:00:00+09:00", periods=12, freq="10min")
    left = pd.DataFrame({"time": times.astype("datetime64[us, UTC]"), "layer": [2] * 12, "a": range(12)})
    right = pd.DataFrame({"time": times.astype(str), "layer": [2] * 12, "b": range(12)})
    merged = dtw.align_on_utc_ns(left, right)
    assert len(merged) == 12
    assert merged["time_utc"].iloc[0].year == 2024
    assert merged["_time_ns"].is_unique
    assert not merged["time_utc"].isna().any()


def test_bootstrap_requires_at_least_two_real_kst_days() -> None:
    one_day = _small_score_frame(days=1)
    with pytest.raises(ValueError, match="at least two KST days"):
        dtw.paired_day_bootstrap(one_day)
    result = dtw.paired_day_bootstrap(_small_score_frame(days=2))
    assert result["day_count"] == 2
    assert result["replicates"] == 5000
    assert math.isfinite(result["ci90_upper_c"])


@pytest.mark.parametrize("column", ["temp_2", "psal_4", "truth", "target", "sample_submission_key"])
def test_target_leakage_firewall_rejects_forbidden_query_columns(column: str) -> None:
    with pytest.raises(ValueError, match="firewall"):
        dtw.assert_query_feature_firewall(["temp_l1", column])


def test_target_leakage_firewall_accepts_only_public_channels() -> None:
    dtw.assert_query_feature_firewall(
        ["temp_l1", "psal_l5", "temp_delta6h_l8", "m2_sin", "temp_missing_l7"]
    )


def test_continuous_block_and_seven_day_source_embargo() -> None:
    source = pd.date_range("2024-01-01T00:00:00Z", periods=24, freq="1h")
    query = pd.date_range("2024-01-09T00:00:00Z", periods=24, freq="1h")
    dtw.validate_continuous_block(source, expected_step_minutes=60, expected_rows=24)
    dtw.validate_source_before_query(source, query)
    broken = source.delete(5)
    with pytest.raises(ValueError, match="continuous"):
        dtw.validate_continuous_block(broken, expected_step_minutes=60)
    too_close = pd.date_range("2024-01-08T00:00:00Z", periods=24, freq="1h")
    with pytest.raises(ValueError, match="embargo"):
        dtw.validate_source_before_query(source, too_close)


def test_constrained_dtw_is_deterministic_banded_and_endpoint_anchored() -> None:
    x = np.linspace(0.0, 1.0, 24)
    query = np.column_stack([x, np.sin(x), np.ones_like(x)])
    source = np.column_stack([x + 0.01, np.sin(x + 0.01), np.ones_like(x)])
    first = dtw.constrained_dtw(query, source, [1.0, 0.5, 0.25], corridor_steps=3)
    second = dtw.constrained_dtw(query, source, [1.0, 0.5, 0.25], corridor_steps=3)
    assert first == second
    assert first.path[0] == (0, 0)
    assert first.path[-1] == (23, 23)
    assert all(abs(i - j) <= 3 for i, j in first.path)
    assert math.isfinite(first.normalized_distance)


def test_constrained_dtw_fails_closed_on_memory_ceiling() -> None:
    query = np.zeros((700, 2))
    source = np.zeros((700, 2))
    with pytest.raises(MemoryError, match="ceiling"):
        dtw.constrained_dtw(query, source, [1.0, 1.0], corridor_steps=12)


def test_historical_residual_alignment_and_weighted_median_need_three_neighbors() -> None:
    path = ((0, 0), (1, 1), (2, 2))
    residual = np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    aligned = dtw.align_historical_residual(path, residual, 3)
    combined = dtw.aggregate_neighbor_residuals(
        [aligned, aligned + 1.0, aligned + 2.0], [0.1, 0.2, 0.3]
    )
    assert combined.shape == (3, 2)
    assert np.isfinite(combined).all()
    with pytest.raises(ValueError, match="insufficient"):
        dtw.aggregate_neighbor_residuals([aligned, aligned], [0.1, 0.2])


def test_inner_selection_uses_only_eighteen_preregistered_records() -> None:
    records = []
    for window in dtw.INNER_WINDOWS:
        for cell in dtw.CELLS:
            score = 1.0 + 0.01 * cell.context_days + 0.001 * cell.neighbors
            records.append(
                {
                    "window_id": window.window_id,
                    "cell_id": cell.cell_id,
                    "layer_equal_rmse_c": score,
                    "worst_layer_rmse_c": score + 0.1,
                }
            )
    assert dtw.select_inner_cell(records).cell_id == "d1_k3"
    with pytest.raises(ValueError, match="exactly"):
        dtw.select_inner_cell(records[:-1])
    contaminated = [dict(row) for row in records]
    contaminated[0]["official_score"] = 1.0
    with pytest.raises(ValueError, match="non-preregistered"):
        dtw.select_inner_cell(contaminated)


def test_zero_fit_protocol_stops_p100_after_exact_no_go(monkeypatch: pytest.MonkeyPatch) -> None:
    slot_events = []

    def materialize(window, cell, deadline):
        del cell, deadline
        return _exact_no_go_frame() if isinstance(window, dtw.WindowSpec) and window.window_id == dtw.EXACT_WINDOW.window_id else _small_score_frame()

    result = dtw.execute_zero_fit_protocol(
        materialize,
        deadline_monotonic=time.monotonic() + 60,
        on_slot=lambda slot, state, details: slot_events.append((slot, state, dict(details))),
    )
    assert result["physical_fit_calls"] == 0
    assert result["overall_gate"] == "RESEARCH_NO_GO"
    assert result["p100"] is None
    assert [state for _, state, _ in slot_events].count("SKIPPED_GATE") == 3
    assert max(slot for slot, _, _ in slot_events) == 22


def test_p100_materializes_only_when_exact_gate_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    slot_events = []
    monkeypatch.setattr(dtw, "exact_research_gate", lambda metrics: True)
    monkeypatch.setattr(dtw, "p100_research_gate", lambda metrics: True)

    def materialize(window, cell, deadline):
        del cell, deadline
        if isinstance(window, str):
            return _small_score_frame(delta=-0.2).assign(fold=window)
        if window.window_id == dtw.EXACT_WINDOW.window_id:
            frame = _exact_no_go_frame()
            frame["candidate"] = frame["truth"]
            return frame
        return _small_score_frame(delta=-0.2)

    result = dtw.execute_zero_fit_protocol(
        materialize,
        deadline_monotonic=time.monotonic() + 60,
        on_slot=lambda slot, state, details: slot_events.append((slot, state, dict(details))),
    )
    assert result["overall_gate"] == "RESEARCH_GO"
    assert result["p100"] is not None
    p100_completed = [details for _, state, details in slot_events if state == "COMPLETED" and details["stage"] == "P100"]
    assert [row["window_id"] for row in p100_completed] == list(dtw.P100_FOLDS)
    assert all(state != "SKIPPED_GATE" for _, state, _ in slot_events)


def test_held_byte_reader_binds_hash_and_open_handle(tmp_path: Path) -> None:
    runner = _load_runner()
    source = tmp_path / "observations.csv"
    source.write_bytes(b"historical-only\n")
    pin = {"bytes": source.stat().st_size, "sha256": _sha256(source)}
    with runner._held_verified_bytes(source, pin) as (handle, payload, digest):
        assert not handle.closed
        assert payload == b"historical-only\n"
        assert digest == pin["sha256"]
    assert handle.closed


def test_claim_creation_failure_performs_no_fit_or_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    _patch_attempt_paths(monkeypatch, runner, tmp_path)

    def fail_claim(path, value):
        del path, value
        raise OSError("claim-build-failure")

    with pytest.raises(OSError, match="claim-build-failure"):
        runner._acquire_attempt("a" * 64, create_json=fail_claim)
    assert not runner.JOURNAL_PATH.exists()
    assert not runner.OOB_PATH.exists()


def test_journal_initialization_failure_is_terminal_fit0_oob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    _patch_attempt_paths(monkeypatch, runner, tmp_path)

    def fail_journal(path, event):
        del path, event
        raise OSError("journal-init-failure")

    with pytest.raises(OSError, match="journal-init-failure"):
        runner._acquire_attempt("b" * 64, append_event=fail_journal)
    receipt = json.loads(runner.OOB_PATH.read_text(encoding="utf-8"))
    assert receipt["status"] == "TERMINAL_FAILURE_NO_RERUN"
    assert receipt["physical_fit_calls"] == 0
    assert receipt["materializations_reserved"] == 0
    assert receipt["materializations_completed"] == 0


def test_materialization_build_failure_consumes_slot_without_retry() -> None:
    events = []

    def fail_build(window, cell, deadline):
        del window, cell, deadline
        raise ValueError("synthetic-build-failure")

    with pytest.raises(ValueError, match="synthetic-build-failure"):
        dtw.execute_zero_fit_protocol(
            fail_build,
            deadline_monotonic=time.monotonic() + 60,
            on_slot=lambda slot, state, details: events.append((slot, state, dict(details))),
        )
    assert events[0][0:2] == (1, "RESERVED")
    assert events[1][0:2] == (1, "FAILED")
    assert len(events) == 2


def test_result_build_failure_records_sanitized_terminal_and_fit0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    _patch_attempt_paths(monkeypatch, runner, tmp_path)
    attempt_id = "attempt-result-failure"
    runner._exclusive_create_json(
        runner.CLAIM_PATH,
        {"attempt_id": attempt_id, "authorization_sha256": "c" * 64},
    )
    runner._append_journal(runner.JOURNAL_PATH, runner._event("ATTEMPT_CLAIMED", attempt_id))
    runner._append_journal(
        runner.JOURNAL_PATH,
        runner._event("PARENT_LAUNCHING_SINGLE_WORKER", attempt_id),
    )
    with pytest.raises(ValueError, match="schema drift") as caught:
        runner._build_parent_result({"physical_fit_calls": 0}, attempt_id=attempt_id)
    runner._record_terminal_failure(caught.value, attempt_id=attempt_id, phase="RESULT_BUILD")
    terminal = runner._read_journal(runner.JOURNAL_PATH)[-1]
    assert terminal["event"] == "ATTEMPT_TERMINAL_FAILED"
    assert terminal["physical_fit_calls"] == 0
    assert terminal["error"]["phase"] == "RESULT_BUILD"
    assert terminal["error"]["locals_captured"] is False
    assert terminal["error"]["raw_traceback_persisted"] is False


def test_result_publication_hardlink_failure_never_commits_terminal_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    _patch_attempt_paths(monkeypatch, runner, tmp_path)

    def fail_link(source, destination):
        del source, destination
        raise OSError("hardlink-failure")

    with pytest.raises(OSError, match="hardlink-failure"):
        runner._publish_aggregate(
            {"status": "aggregate-only", "physical_fit_calls": 0},
            attempt_id="publication-failure",
            link_fn=fail_link,
        )
    assert not (runner.FINAL_DIR / "terminal_success.json").exists()
    assert not (runner.FINAL_DIR / "result.json").exists()


def test_sanitized_error_provenance_redacts_paths_and_keeps_chain() -> None:
    runner = _load_runner()
    try:
        try:
            raise ValueError(r"failure at C:\private\values.csv")
        except ValueError as error:
            raise RuntimeError("outer failure") from error
    except RuntimeError as error:
        provenance = runner.sanitized_error_provenance(error, phase="BUILD")
    assert [row["type"] for row in provenance["chain"]] == ["RuntimeError", "ValueError"]
    assert "private" not in json.dumps(provenance)
    assert provenance["traceback_frame_count"] >= 1
    assert len(provenance["traceback_sha256"]) == 64


def test_journal_materialization_accounting_is_exact_and_fit0() -> None:
    runner = _load_runner()
    attempt = "accounting"
    events = []
    for slot in range(1, 20):
        events.append(runner._event("MATERIALIZATION_RESERVED", attempt, slot=slot))
        events.append(runner._event("MATERIALIZATION_COMPLETED", attempt, slot=slot))
    for slot in range(20, 23):
        events.append(runner._event("MATERIALIZATION_SKIPPED_GATE", attempt, slot=slot))
    counts = runner._materialization_counts(events)
    assert counts == {
        "states": {**{str(slot): "COMPLETED" for slot in range(1, 20)}, **{str(slot): "SKIPPED_GATE" for slot in range(20, 23)}},
        "reserved": 19,
        "completed": 19,
        "failed": 0,
        "skipped_gate": 3,
        "physical_fit_calls": 0,
    }


def test_runner_and_numerical_module_have_no_fit_call() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8") + (REPO_ROOT / "src/p2_restore/public_trajectory_dtw.py").read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "PHYSICAL_FIT_CALLS = 0" in source


def test_external_authorization_is_false_and_preflight_is_read_only() -> None:
    runner = _load_runner()
    preflight = runner.read_only_preflight()
    assert preflight["status"] == "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
    assert preflight["read_only"] is True
    assert preflight["authorized"] is False
    assert all(value == 0 for value in preflight["operation_counters"].values())
    with pytest.raises(runner.AuthorizationError):
        runner._authorization_state(require_authorized=True)
