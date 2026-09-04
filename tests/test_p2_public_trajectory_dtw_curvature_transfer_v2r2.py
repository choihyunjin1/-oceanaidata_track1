from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from p2_restore import public_trajectory_dtw_v2r2 as dtw

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/run_p2_public_trajectory_dtw_curvature_transfer_v2r2.py"
DESIGN_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r2_design.json"
DESIGN_SHA256 = "e90a866e69d01731704db878332c121a4f42a0258e61d8489afa9344fa8264a1"


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
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "SEAL_PATH", seal)
    monkeypatch.setattr(runner, "CLAIM_PATH", tmp_path / "claims" / "attempt.claim.json")
    monkeypatch.setattr(runner, "JOURNAL_PATH", tmp_path / "journals" / "attempt.ndjson")
    monkeypatch.setattr(runner, "OOB_PATH", tmp_path / "receipts" / "terminal.json")
    monkeypatch.setattr(runner, "FINAL_DIR", tmp_path / "final")
    monkeypatch.setattr(runner, "STAGING_ROOT", tmp_path / "staging")


def _operation_ceiling() -> dict[str, int]:
    return {
        "attempts": 1,
        "physical_fit_calls": 0,
        "inner_materializations": 18,
        "exact_materializations": 1,
        "conditional_p100_materializations": 3,
        "total_materialization_slots": 22,
        "result_driven_reruns": 0,
        "candidate_files": 0,
        "uploads": 0,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
    }


def test_design_hash_is_immutable_and_not_authorized() -> None:
    assert _sha256(DESIGN_PATH) == DESIGN_SHA256
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    assert design["schema"] == "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
    assert design["status"] == "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
    assert design["unchanged_scientific_contract"]["model_fit_calls"] == 0
    assert design["preimplementation_freeze"]["materializations_at_freeze"] == 0
    assert design["preimplementation_freeze"]["p100_filesystem_accesses_at_freeze"] == 0


def test_trigger_resolution_freezes_infrastructure_failure_without_scientific_result() -> None:
    trigger_path = (
        REPO_ROOT
        / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2_trigger_resolution.json"
    )
    trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
    assert _sha256(trigger_path) == "36c9a2943b14d56867784c5e1e91d2bee086eebb9247f90e14a11db6f4eaf9e9"
    evidence = trigger["predecessor_evidence"]
    assert evidence["scientific_outcome"] == "UNEVALUATED_INFRASTRUCTURE_FAILURE"
    assert set(evidence["must_not_be_relabelled_as"]) == {
        "RESEARCH_GO",
        "RESEARCH_NO_GO",
        "SUBMISSION_GO",
    }
    resolution = trigger["prospective_branch_resolution"]
    assert resolution["branch"] == "FROZEN_EXACT_INCUMBENT_ONLY"
    assert resolution["ncr_predictions_or_metrics_consumed"] is False


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


def test_independent_observation_truth_binds_anchor_and_candidate() -> None:
    times = pd.date_range("2024-09-01T00:00:00+09:00", periods=4, freq="10min")
    observations = pd.DataFrame(
        [
            {"time": time_value, "layer": layer, "temp": layer + offset / 10.0}
            for offset, time_value in enumerate(times)
            for layer in dtw.TARGET_LAYERS
        ]
    )
    anchor = observations.rename(columns={"temp": "truth"}).copy()
    anchor["prediction"] = anchor["truth"] + 0.5
    trusted, certificate = dtw.verify_anchor_against_observations(
        observations,
        anchor,
        anchor_prediction_column="prediction",
        expected_rows=len(anchor),
    )
    assert certificate["anchor_truth_equal_observations"] is True
    candidate = trusted[["time", "layer"]].copy()
    candidate["truth"] = -999.0
    candidate["candidate"] = trusted["anchor"] - 0.1
    candidate["support"] = True
    verified, candidate_certificate = dtw.verify_candidate_against_truth(candidate, trusted)
    assert np.array_equal(verified["truth"], trusted["truth"])
    assert candidate_certificate["candidate_truth_column_used"] is False
    corrupted = anchor.copy()
    corrupted.loc[0, "truth"] += 1.0
    with pytest.raises(ValueError, match="independently reconstructed"):
        dtw.verify_anchor_against_observations(
            observations,
            corrupted,
            anchor_prediction_column="prediction",
            expected_rows=len(anchor),
        )


def test_p100_78156_row_canonical_binding_is_order_invariant_and_adversarial() -> None:
    frames = []
    observations = []
    starts = {
        "outer_2024_sep_oct": "2024-09-01T00:00:00+09:00",
        "outer_2025_may_jun": "2025-05-01T00:00:00+09:00",
        "outer_2025_jul_aug": "2025-07-01T00:00:00+09:00",
    }
    for fold_index, (fold, start) in enumerate(starts.items()):
        station = f"S{fold_index + 1}"
        times = pd.date_range(start, periods=8684, freq="10min")
        repeated_times = np.repeat(times.to_numpy(), 3)
        layers = np.tile(np.asarray(dtw.TARGET_LAYERS), len(times))
        truth = fold_index * 100.0 + np.arange(len(repeated_times), dtype=float) / 10000.0
        frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "station": station,
                    "time": repeated_times,
                    "layer": layers,
                    "truth": truth,
                    "INCUMBENT_NOOP": truth + 0.5,
                }
            )
        )
        observations.append(
            pd.DataFrame(
                {
                    "station": station,
                    "time": repeated_times,
                    "layer": layers,
                    "temp": truth,
                }
            )
        )
    anchor = pd.concat(frames, ignore_index=True)
    historical = pd.concat(observations, ignore_index=True)
    assert len(anchor) == 78156

    legacy_metadata = anchor.sort_values(["layer", "time"], kind="mergesort")[["fold", "station"]]
    canonical_rows = anchor.sort_values(["time", "layer"], kind="mergesort")
    legacy_mismatches = int(
        np.count_nonzero(
            legacy_metadata["fold"].to_numpy() != canonical_rows["fold"].to_numpy()
        )
    )
    assert legacy_mismatches == 52104

    first, first_certificate = dtw.verify_p100_anchor_against_observations(
        historical.sample(frac=1.0, random_state=11),
        anchor.sample(frac=1.0, random_state=12),
    )
    second, second_certificate = dtw.verify_p100_anchor_against_observations(
        historical.sample(frac=1.0, random_state=21),
        anchor.sample(frac=1.0, random_state=22),
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_certificate == second_certificate
    assert first_certificate["positional_metadata_assignment_used"] is False

    candidate = first[["fold", "station", "time", "layer"]].copy()
    candidate["candidate"] = first["anchor"] - 0.1
    candidate["support"] = True
    verified, certificate = dtw.verify_candidate_against_truth(
        candidate.sample(frac=1.0, random_state=23),
        first,
    )
    assert len(verified) == 78156
    assert certificate["candidate_key_equal_verified_truth"] is True

    corrupted = anchor.copy()
    corrupted["fold"] = corrupted["fold"].map(
        {
            "outer_2024_sep_oct": "outer_2025_may_jun",
            "outer_2025_may_jun": "outer_2025_jul_aug",
            "outer_2025_jul_aug": "outer_2024_sep_oct",
        }
    )
    with pytest.raises(ValueError, match="fold metadata"):
        dtw.verify_p100_anchor_against_observations(historical, corrupted)


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
    exact = dtw.validate_source_end_before_trajectory_start(
        pd.Timestamp("2024-01-02T00:00:00Z"),
        pd.Timestamp("2024-01-09T00:00:00Z"),
    )
    assert exact["embargo_ns"] == exact["minimum_embargo_ns"]


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
    lazy_events = []
    monkeypatch.setattr(dtw, "exact_research_gate", lambda metrics: True)
    monkeypatch.setattr(dtw, "p100_research_gate", lambda metrics: True)
    monkeypatch.setattr(
        dtw,
        "derive_p100_metrics",
        lambda frame: {"rows": len(frame), "by_fold": {}, "by_layer": {}},
    )

    def materialize(window, cell, deadline):
        del cell, deadline
        if isinstance(window, str):
            assert lazy_events == ["LOADED_AFTER_EXACT"]
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
        on_exact_research_go=lambda: lazy_events.append("LOADED_AFTER_EXACT"),
    )
    assert result["overall_gate"] == "RESEARCH_GO"
    assert result["p100"] is not None
    p100_completed = [details for _, state, details in slot_events if state == "COMPLETED" and details["stage"] == "P100"]
    assert [row["window_id"] for row in p100_completed] == list(dtw.P100_FOLDS)
    assert all(state != "SKIPPED_GATE" for _, state, _ in slot_events)
    exact_completed_index = next(
        index
        for index, (_, state, details) in enumerate(slot_events)
        if state == "COMPLETED" and details["stage"] == "EXACT"
    )
    first_p100_index = next(
        index
        for index, (_, state, details) in enumerate(slot_events)
        if state == "RESERVED" and details["stage"] == "P100"
    )
    assert exact_completed_index < first_p100_index


def test_p100_required_slices_are_computed_and_population_balanced() -> None:
    rows = 78156
    index = np.arange(rows)
    support = index % 11 != 0
    frame = pd.DataFrame(
        {
            "fold": np.asarray(dtw.P100_FOLDS, dtype=object)[index % 3],
            "layer": np.asarray(dtw.TARGET_LAYERS, dtype=int)[(index // 3) % 3],
            "time": pd.Timestamp("2024-01-01T00:00:00Z")
            + pd.to_timedelta((index % (366 * 24 * 6)) * 10, unit="min"),
            "truth": np.sin(index / 1000.0),
            "anchor": np.sin(index / 1000.0) + 0.5,
            "candidate": np.sin(index / 1000.0) + 0.4,
            "support": support,
            "dtw_distance": np.where(support, (index % 100) / 40.0, np.nan),
            "public_missingness": np.where(support, (index % 20) / 100.0, np.nan),
            "trajectory_coverage": np.where(support, 0.8 + (index % 20) / 100.0, np.nan),
        }
    )
    slices = dtw.derive_p100_slice_metrics(frame)
    assert slices["required_slices_logged"] is True
    assert slices["derived_not_hardcoded"] is True
    assert set(slices["families"]) == {
        "fold",
        "layer",
        "season",
        "distance_decile",
        "missingness_decile",
        "coverage_decile",
        "distance_by_layer",
    }
    assert all(slices["family_row_count_identity"].values())
    assert len(slices["inventory_sha256"]) == 64


def test_weak_support_no_op_integrity_is_derived() -> None:
    frame = _small_score_frame()
    frame.loc[0, "support"] = False
    integrity = dtw.exact_no_op_integrity(frame)
    assert integrity["weak_support_candidate_equals_anchor"] is True
    frame.loc[0, "candidate"] += 0.01
    assert dtw.exact_no_op_integrity(frame)["weak_support_candidate_equals_anchor"] is False


def test_missing_endpoint_and_weak_support_force_exact_anchor_noop() -> None:
    anchor = np.asarray([2.0, 3.0, 4.0])
    candidate = np.asarray([2.1, 3.1, 4.1])
    projected, support, reason = dtw.endpoint_safe_candidate(
        math.nan,
        anchor,
        candidate,
        5.0,
        [True, True, True],
    )
    assert np.array_equal(projected, anchor)
    assert not support.any()
    assert reason == "MISSING_L1_OR_L5_ENDPOINT"
    partial, partial_support, reason = dtw.endpoint_safe_candidate(
        1.0,
        anchor,
        candidate,
        5.0,
        [True, False, True],
    )
    assert partial[1] == anchor[1]
    assert partial_support.tolist() == [True, False, True]
    assert reason == "PARTIAL_WEAK_SUPPORT_NO_PROJECTION"


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


def test_pre_exact_source_readiness_never_verifies_p100(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    observed = []

    class LiteralOnlyP100:
        def __str__(self) -> str:
            return "C:/sealed/history/evaluated_oof_100.parquet"

        def resolve(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("p100 resolve before exact gate")

        def stat(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("p100 stat before exact gate")

        def open(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("p100 open before exact gate")

    p100_path = LiteralOnlyP100()
    monkeypatch.setattr(runner, "P100_ANCHOR_PATH", p100_path)
    paths = {
        "observations": (tmp_path / "observations.csv", {"bytes": 1, "sha256": "a" * 64}),
        "exact_anchor": (tmp_path / "exact.parquet", {"bytes": 1, "sha256": "b" * 64}),
    }
    p100_literal = {
        "path_literal": str(p100_path),
        "bytes": 7,
        "sha256": "c" * 64,
    }
    monkeypatch.setattr(runner, "_preexact_source_pins", lambda seal: paths)
    monkeypatch.setattr(runner, "_p100_literal_pin", lambda seal: p100_literal)

    def verify(path, pin):
        observed.append(path.name)
        return {"path": str(path), **pin}

    monkeypatch.setattr(runner, "_verify_pin", verify)
    readiness = runner._source_readiness({})
    assert observed == ["observations.csv", "exact.parquet"]
    assert readiness["p100_anchor"]["filesystem_accesses"] == 0
    assert not (tmp_path / "must_not_be_touched.parquet").exists()


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
    assert receipt["materialization_accounting"]["reserved"] == 0
    assert receipt["materialization_accounting"]["completed"] == 0
    assert receipt["design_sha256"] == runner.DESIGN_SHA256
    assert receipt["authorization_sha256"] == "b" * 64
    assert receipt["observed_inventory"]


def test_oob_identity_or_inventory_corruption_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    _patch_attempt_paths(monkeypatch, runner, tmp_path)
    attempt = "oob-corruption"
    runner._exclusive_create_json(
        runner.CLAIM_PATH,
        {"attempt_id": attempt, "authorization_sha256": "d" * 64},
    )
    receipt = runner._write_oob_receipt(
        attempt_id=attempt,
        authorization_hash="d" * 64,
        reason="FAULT_INJECTION",
        error=ValueError("synthetic"),
        phase="OOB_TEST",
        accounting={"physical_fit_calls": 0},
    )
    corrupted = dict(receipt)
    corrupted["authorization_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="identity"):
        runner._verify_oob_receipt(
            corrupted,
            attempt_id=attempt,
            authorization_hash="d" * 64,
        )


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


def test_real_windows_publication_reaches_terminal_and_terminal_link_is_last_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    _patch_attempt_paths(monkeypatch, runner, tmp_path)
    attempt = "real-windows-publication"
    runner._append_journal(runner.JOURNAL_PATH, runner._event("ATTEMPT_CLAIMED", attempt))
    io_events = []
    real_hash = runner._sha256
    real_sync = runner._fsync_directory

    def logged_hash(path):
        io_events.append(("hash", Path(path).name))
        return real_hash(path)

    def logged_sync(path):
        io_events.append(("sync", Path(path).name))
        return real_sync(path)

    def terminal_link(source, destination):
        runner._terminal_hardlink_last(source, destination)
        io_events.append(("TERMINAL_LINK", Path(destination).name))

    monkeypatch.setattr(runner, "_sha256", logged_hash)
    publication = runner._publish_aggregate(
        {"status": "aggregate-only", "physical_fit_calls": 0},
        attempt_id=attempt,
        sync_directory=logged_sync,
        terminal_link_fn=terminal_link,
    )
    assert io_events[-1] == ("TERMINAL_LINK", "terminal_success.json")
    assert publication["post_terminal_repository_io"] == 0
    assert runner._terminal_state()["logical_state"] == "SUCCESS"
    assert os.path.samefile(
        runner.FINAL_DIR / "result.json",
        next(runner.STAGING_ROOT.glob("*/result.json")),
    )
    assert os.path.samefile(
        runner.FINAL_DIR / "manifest.json",
        next(runner.STAGING_ROOT.glob("*/manifest.json")),
    )


@pytest.mark.parametrize("fault", ["PRETERMINAL_FSYNC", "COMMIT_READY", "TERMINAL_LINK"])
def test_publication_faults_fail_closed_before_terminal_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    runner = _load_runner()
    root = tmp_path / fault.lower()
    root.mkdir()
    _patch_attempt_paths(monkeypatch, runner, root)
    attempt = f"fault-{fault.lower()}"
    runner._append_journal(runner.JOURNAL_PATH, runner._event("ATTEMPT_CLAIMED", attempt))
    sync_calls = 0

    def sync(path):
        nonlocal sync_calls
        sync_calls += 1
        if fault == "PRETERMINAL_FSYNC" and sync_calls == 5:
            raise OSError("preterminal-fsync-fault")
        return runner._fsync_directory(path)

    def append(path, event):
        if fault == "COMMIT_READY" and event["event"] == "ATTEMPT_COMMIT_READY":
            raise OSError("commit-ready-fault")
        runner._append_journal(path, event)

    def terminal_link(source, destination):
        if fault == "TERMINAL_LINK":
            raise OSError("terminal-link-fault")
        runner._terminal_hardlink_last(source, destination)

    with pytest.raises(OSError, match="fault"):
        runner._publish_aggregate(
            {"status": "aggregate-only", "physical_fit_calls": 0},
            attempt_id=attempt,
            sync_directory=sync,
            append_event=append,
            terminal_link_fn=terminal_link,
        )
    assert not (runner.FINAL_DIR / "terminal_success.json").exists()
    state = runner._terminal_state()
    if fault == "TERMINAL_LINK":
        assert state["logical_state"] == "COMMIT_INCOMPLETE"
        runner._record_terminal_failure(
            OSError("terminal-link-fault"),
            attempt_id=attempt,
            phase="RESULT_PUBLICATION",
            authorization_hash="f" * 64,
        )
        assert runner._terminal_state()["logical_state"] == "FAILURE"
    else:
        assert state["logical_state"] == "NONE"


def test_atomic_result_replace_failure_never_creates_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    _patch_attempt_paths(monkeypatch, runner, tmp_path)
    destination = tmp_path / "staging-one" / "result.json"

    def fail_replace(source, target):
        del source, target
        raise OSError("replace-fault")

    monkeypatch.setattr(runner.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace-fault"):
        runner._atomic_replace_json(destination, {"aggregate": True})
    assert not destination.exists()


def test_directory_fsync_failure_consumes_create_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    target = tmp_path / "claim.json"
    monkeypatch.setattr(
        runner,
        "_fsync_directory",
        lambda path: (_ for _ in ()).throw(OSError("dir-fsync-fault")),
    )
    with pytest.raises(OSError, match="dir-fsync-fault"):
        runner._exclusive_create_json(target, {"single_attempt": True})
    assert target.exists()
    with pytest.raises(FileExistsError):
        runner._exclusive_create_json(target, {"single_attempt": True})


def test_terminal_success_failure_conflict_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    _patch_attempt_paths(monkeypatch, runner, tmp_path)
    attempt = "terminal-conflict"
    ready = runner._event("ATTEMPT_COMMIT_READY", attempt)
    runner._append_journal(runner.JOURNAL_PATH, ready)
    runner._append_journal(runner.JOURNAL_PATH, runner._event("ATTEMPT_TERMINAL_FAILED", attempt))
    runner.FINAL_DIR.mkdir(parents=True)
    runner._exclusive_create_json(
        runner.FINAL_DIR / "terminal_success.json",
        {
            "experiment_id": runner.EXPERIMENT_ID,
            "attempt_id": attempt,
            "commit_ready_event_sha256": runner._sha256_bytes(runner._canonical_bytes(ready)),
        },
    )
    with pytest.raises(RuntimeError, match="conflicting"):
        runner._terminal_state()


def test_windows_process_tree_proof_enumerates_and_verifies_descendants(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    process = SimpleNamespace(pid=100, wait=lambda timeout: None, poll=lambda: 1)
    captured = [
        {"pid": 100, "create_time": 1.0},
        {"pid": 101, "create_time": 2.0},
        {"pid": 102, "create_time": 3.0},
    ]
    monkeypatch.setattr(runner, "_enumerate_process_tree", lambda pid: captured)
    monkeypatch.setattr(runner, "_captured_processes_still_present", lambda rows: [])
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    evidence = runner._terminate_process_tree(process)
    assert evidence["pretermination_descendant_pids"] == [101, 102]
    assert evidence["all_captured_descendants_absent"] is True
    assert evidence["root_absence_verified"] is True


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
    source = RUNNER_PATH.read_text(encoding="utf-8") + (REPO_ROOT / "src/p2_restore/public_trajectory_dtw_v2r2.py").read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "PHYSICAL_FIT_CALLS = 0" in source


def test_authorized_worker_loads_sealed_module_without_package_init() -> None:
    runner = _load_runner()
    source = RUNNER_PATH.read_text(encoding="utf-8")
    worker = source[source.index("def _load_sealed_numerical_module") : source.index("def _run_internal_worker")]
    assert "spec_from_file_location" in worker
    assert "from p2_restore" not in worker
    loaded = runner._load_sealed_numerical_module()
    assert loaded.CELLS[0].cell_id == "d1_k3"
    assert Path(loaded.__file__).resolve() == runner.MODULE_PATH.resolve()


@pytest.mark.parametrize(
    "corruption",
    ["missing_report", "wrong_bytes", "wrong_sha", "wrong_verdict", "wrong_design", "wrong_seal"],
)
def test_authorization_verifies_real_qa_receipt_and_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    corruption: str,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    seal_path = tmp_path / runner.SEAL_RELATIVE
    seal_path.parent.mkdir(parents=True)
    seal_path.write_text('{"sealed":true}\n', encoding="utf-8")
    monkeypatch.setattr(runner, "SEAL_PATH", seal_path)
    qa_path = tmp_path / "reports/p2_v2r2_independent_qa.json"
    qa_path.parent.mkdir(parents=True)
    qa_report = {
        "experiment_id": runner.EXPERIMENT_ID,
        "verdict": "PASS",
        "design_sha256": runner.DESIGN_SHA256,
        "seal_sha256": _sha256(seal_path),
    }
    qa_path.write_text(json.dumps(qa_report, sort_keys=True) + "\n", encoding="utf-8")
    qa_pin = {
        "path": str(qa_path.relative_to(tmp_path)).replace("\\", "/"),
        "bytes": qa_path.stat().st_size,
        "sha256": _sha256(qa_path),
        "verdict": "PASS",
        "design_sha256": runner.DESIGN_SHA256,
        "seal_sha256": _sha256(seal_path),
    }
    static_relatives = {
        runner.DESIGN_RELATIVE,
        runner.TRIGGER_RELATIVE,
        runner.EXECUTION_CONFIG_RELATIVE,
        runner.RUNNER_RELATIVE,
        runner.MODULE_RELATIVE,
        runner.TEST_RELATIVE,
        runner.SEAL_RELATIVE,
        runner.CLOSURE_MATRIX_RELATIVE,
    }
    static_files = {}
    for relative in static_relatives:
        path = tmp_path / relative
        if path != seal_path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"sealed:{relative}\n", encoding="utf-8")
        static_files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if corruption == "missing_report":
        qa_pin["path"] = "reports/absent_qa.json"
    elif corruption == "wrong_bytes":
        qa_pin["bytes"] += 1
    elif corruption == "wrong_sha":
        qa_pin["sha256"] = "0" * 64
    elif corruption == "wrong_verdict":
        qa_pin["verdict"] = "NO_GO"
    elif corruption == "wrong_design":
        qa_pin["design_sha256"] = "1" * 64
    elif corruption == "wrong_seal":
        qa_pin["seal_sha256"] = "2" * 64
    authorization = {
        "schema_version": "p2_public_trajectory_dtw.execution_authorization.v2r2",
        "status": "AUTHORIZED_AFTER_INDEPENDENT_QA",
        "experiment_id": runner.EXPERIMENT_ID,
        "authorized": True,
        "design_sha256": runner.DESIGN_SHA256,
        "trigger_resolution_sha256": runner.TRIGGER_SHA256,
        "seal_sha256": _sha256(seal_path),
        "bundle": {
            "static_files": static_files,
            "independent_qa_report": dict(qa_pin),
        },
        "operation_ceiling": _operation_ceiling(),
        "independent_qa": dict(qa_pin),
        "blockers": [],
    }
    auth_path = tmp_path / "authorization.json"
    auth_path.write_text(json.dumps(authorization, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "AUTHORIZATION_PATH", auth_path)
    monkeypatch.setenv(runner.AUTHORIZATION_ENV, _sha256(auth_path))
    with pytest.raises((runner.AuthorizationError, FileNotFoundError, RuntimeError)):
        runner._authorization_state(require_authorized=True)


def test_authorization_accepts_only_a_pinned_real_pass_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    seal_path = tmp_path / runner.SEAL_RELATIVE
    seal_path.parent.mkdir(parents=True)
    seal_path.write_text('{"sealed":true}\n', encoding="utf-8")
    monkeypatch.setattr(runner, "SEAL_PATH", seal_path)
    qa_path = tmp_path / "reports/p2_v2r2_independent_qa.json"
    qa_path.parent.mkdir(parents=True)
    qa_report = {
        "experiment_id": runner.EXPERIMENT_ID,
        "verdict": "PASS",
        "design_sha256": runner.DESIGN_SHA256,
        "seal_sha256": _sha256(seal_path),
    }
    qa_path.write_text(json.dumps(qa_report, sort_keys=True) + "\n", encoding="utf-8")
    qa_pin = {
        "path": str(qa_path.relative_to(tmp_path)).replace("\\", "/"),
        "bytes": qa_path.stat().st_size,
        "sha256": _sha256(qa_path),
        "verdict": "PASS",
        "design_sha256": runner.DESIGN_SHA256,
        "seal_sha256": _sha256(seal_path),
    }
    static_relatives = {
        runner.DESIGN_RELATIVE,
        runner.TRIGGER_RELATIVE,
        runner.EXECUTION_CONFIG_RELATIVE,
        runner.RUNNER_RELATIVE,
        runner.MODULE_RELATIVE,
        runner.TEST_RELATIVE,
        runner.SEAL_RELATIVE,
        runner.CLOSURE_MATRIX_RELATIVE,
    }
    static_files = {}
    for relative in static_relatives:
        path = tmp_path / relative
        if path != seal_path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"sealed:{relative}\n", encoding="utf-8")
        static_files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    authorization = {
        "schema_version": "p2_public_trajectory_dtw.execution_authorization.v2r2",
        "status": "AUTHORIZED_AFTER_INDEPENDENT_QA",
        "experiment_id": runner.EXPERIMENT_ID,
        "authorized": True,
        "design_sha256": runner.DESIGN_SHA256,
        "trigger_resolution_sha256": runner.TRIGGER_SHA256,
        "seal_sha256": _sha256(seal_path),
        "bundle": {"static_files": static_files, "independent_qa_report": qa_pin},
        "operation_ceiling": _operation_ceiling(),
        "independent_qa": qa_pin,
        "blockers": [],
    }
    auth_path = tmp_path / "authorization.json"
    auth_path.write_text(json.dumps(authorization, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "AUTHORIZATION_PATH", auth_path)
    monkeypatch.setenv(runner.AUTHORIZATION_ENV, _sha256(auth_path))
    observed, observed_hash = runner._authorization_state(require_authorized=True)
    assert observed["authorized"] is True
    assert observed_hash == _sha256(auth_path)


def test_seal_transitive_inventory_is_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    seal, verification = runner._verify_seal()
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    assert seal["transitive_static_inputs"] == design["transitive_static_inputs"]
    assert len(verification["transitive_static_inputs"]) == 14
    assert verification["predecessor_semantics"]["terminal_event"] == "ATTEMPT_TERMINAL_FAILED"
    corrupted = dict(seal)
    corrupted["transitive_static_inputs"] = seal["transitive_static_inputs"][:-1]
    real_read_json = runner._read_json

    def read_json(path):
        if Path(path).resolve() == runner.SEAL_PATH.resolve():
            return corrupted
        return real_read_json(path)

    monkeypatch.setattr(runner, "_read_json", read_json)
    with pytest.raises(ValueError, match="transitive inventory"):
        runner._verify_seal()


def test_external_authorization_is_false_and_preflight_is_read_only() -> None:
    runner = _load_runner()
    preflight = runner.read_only_preflight()
    assert preflight["status"] == "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
    assert preflight["read_only"] is True
    assert preflight["authorized"] is False
    assert all(value == 0 for value in preflight["operation_counters"].values())
    with pytest.raises(runner.AuthorizationError):
        runner._authorization_state(require_authorized=True)
