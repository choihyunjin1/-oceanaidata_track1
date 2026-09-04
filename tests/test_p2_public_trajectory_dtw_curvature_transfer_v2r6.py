from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = REPO_ROOT / (
    "configs/experiments/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_design.json"
)
TRIGGER_PATH = REPO_ROOT / (
    "configs/experiments/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_trigger_resolution.json"
)
EXECUTION_PATH = REPO_ROOT / (
    "configs/experiments/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_execution.json"
)
MODULE_PATH = REPO_ROOT / "src/p2_restore/public_trajectory_dtw_v2r6.py"
RUNNER_PATH = REPO_ROOT / (
    "scripts/run_p2_public_trajectory_dtw_curvature_transfer_v2r6.py"
)
SEAL_PATH = REPO_ROOT / (
    "artifacts/p2_public_trajectory_dtw_curvature_transfer_v2r6_preexecution/"
    "preexecution_seal.json"
)
AUTH_PATH = REPO_ROOT / (
    "configs/experiments/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_execution_authorization.json"
)
SLOT_DIAGNOSTIC_PATH = REPO_ROOT / (
    "reports/p2_public_trajectory_dtw_curvature_transfer_v2r6_"
    "slot1_error_only_diagnostic_20260826.json"
)
IDENTIFIABILITY_PATH = REPO_ROOT / (
    "reports/p2_public_trajectory_dtw_curvature_transfer_v2r6_"
    "inner_identifiability_certificate_20260826.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = _load("_test_p2_dtw_v2r6_guard", MODULE_PATH)


def _load_runner():
    return _load("_test_p2_dtw_v2r6_runner", RUNNER_PATH)


def _synthetic_panel(*, march_truth: bool) -> SimpleNamespace:
    index = pd.date_range(
        "2024-03-01T00:00:00+09:00",
        "2024-08-01T00:00:00+09:00",
        inclusive="left",
        freq="10min",
    ).tz_convert("UTC")
    temp = pd.DataFrame(1.0, index=index, columns=range(1, 9))
    baseline = pd.DataFrame(1.0, index=index, columns=guard.TARGET_LAYERS)
    if not march_truth:
        march = (
            index < pd.Timestamp("2024-04-01T00:00:00+09:00").tz_convert("UTC")
        )
        temp.loc[march, list(guard.TARGET_LAYERS)] = np.nan
    return SimpleNamespace(index=index, temp=temp, baseline=baseline)


def _runtime_certificate_from_frozen() -> dict:
    frozen = json.loads(IDENTIFIABILITY_PATH.read_text(encoding="utf-8"))
    windows = [
        {
            key: row[key]
            for key in (
                "window_id",
                "time_keys",
                "layer_finite_masks",
                "rows_after_finite_truth_anchor_mask",
                "kst_days_after_mask",
                "identifiable_for_scoring",
            )
        }
        for row in frozen["windows"]
    ]
    return {
        "schema_version": "p2_dtw_v2r6.inner_identifiability_runtime.v1",
        "windows": windows,
        "all_three_registered_windows_identifiable": False,
        "frozen_eighteen_cell_selection_complete": False,
        "prediction_values_read_or_written": 0,
        "metrics_computed": 0,
        "scores_computed": 0,
        "physical_fit_calls": 0,
        "p100_accesses": 0,
    }


def test_prospective_design_and_trigger_resolution_are_frozen() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    trigger = json.loads(TRIGGER_PATH.read_text(encoding="utf-8"))
    assert DESIGN_PATH.stat().st_size == 14_145
    assert _sha256(DESIGN_PATH) == (
        "dc8684950bf7466038270a33ab2f2c6425820433aa205790bd40c1d5e842004e"
    )
    assert TRIGGER_PATH.stat().st_size == 3_557
    assert _sha256(TRIGGER_PATH) == (
        "3348f8bf10335a1da414fcbb8f04f7fb7e00ec798ff774bba63edbc13af73f87"
    )
    assert design["status"] == "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
    assert trigger["resolution"]["terminal_status"] == (
        "NO_GO_INNER_WINDOW_UNIDENTIFIABLE"
    )
    assert trigger["resolution"]["claim_may_be_created"] is False
    assert trigger["resolution"]["authorization_may_become_true"] is False


def test_execution_contract_is_permanently_preclaim_blocked() -> None:
    execution = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))
    assert execution["authorized"] is False
    assert execution["execution_permitted"] is False
    assert execution["status"] == (
        "PRECLAIM_BLOCKED_NO_GO_INNER_WINDOW_UNIDENTIFIABLE"
    )
    assert all(value == 0 for value in execution["operation_ceiling"].values())


def test_v2r5_failure_lineage_is_immutable_and_zero_completed() -> None:
    runner = _load_runner()
    claim = runner.V2R5_CLAIM_PATH.read_bytes()
    journal = runner.V2R5_JOURNAL_PATH.read_bytes()
    assert hashlib.sha256(claim).hexdigest() == (
        "00a8f84de695475eaec52ec691ab219448408c6dc90f5e9879ace71cc09c232f"
    )
    assert hashlib.sha256(journal).hexdigest() == (
        "710a4eb202467aca26b13c24f74aff71378a0c23cea0e1e890c5b07f0e8f2e4f"
    )
    result = runner._verify_v2r5_lineage(claim, journal)
    assert result["reserved"] == 1
    assert result["failed"] == 1
    assert result["completed"] == 0
    assert result["scores"] == 0
    assert result["physical_fit_calls"] == 0
    assert result["p100_accesses"] == 0


def test_one_shot_diagnostic_consumed_no_scientific_result() -> None:
    diagnostic = json.loads(SLOT_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    attempt = diagnostic["diagnostic_attempt"]
    assert attempt["ordinal"] == attempt["attempt_ceiling"] == 1
    assert attempt["materializer_status"] == (
        "RETURNED_WITHOUT_EXCEPTION_PAYLOAD_IMMEDIATELY_DISCARDED"
    )
    assert attempt["additional_real_slot_diagnostic_authorized"] is False
    assert all(value == 0 for value in diagnostic["operation_counters"].values())


def test_frozen_identifiability_certificate_has_exact_march_blocker() -> None:
    frozen = json.loads(IDENTIFIABILITY_PATH.read_text(encoding="utf-8"))
    march, may, july = frozen["windows"]
    assert march["rows_after_finite_truth_anchor_mask"] == 0
    assert [
        march["layer_finite_masks"][str(layer)]["finite_truth_and_anchor"]
        for layer in guard.TARGET_LAYERS
    ] == [0, 0, 0]
    assert may["rows_after_finite_truth_anchor_mask"] == 9_985
    assert july["rows_after_finite_truth_anchor_mask"] == 13_358
    assert frozen["all_three_registered_windows_identifiable"] is False
    assert frozen["frozen_eighteen_cell_selection_complete"] is False


def test_synthetic_all_nan_march_truth_is_preclaim_no_go() -> None:
    certificate = guard.build_inner_identifiability_certificate(
        _synthetic_panel(march_truth=False)
    )
    assert certificate["windows"][0]["time_keys"] == 4_464
    assert certificate["windows"][0]["rows_after_finite_truth_anchor_mask"] == 0
    assert certificate["windows"][0]["kst_days_after_mask"] == 0
    assert certificate["all_three_registered_windows_identifiable"] is False


def test_synthetic_identifiable_windows_are_detected_without_scoring() -> None:
    certificate = guard.build_inner_identifiability_certificate(
        _synthetic_panel(march_truth=True)
    )
    assert certificate["all_three_registered_windows_identifiable"] is True
    assert certificate["frozen_eighteen_cell_selection_complete"] is True
    assert all(row["kst_days_after_mask"] >= 2 for row in certificate["windows"])
    assert certificate["metrics_computed"] == 0
    assert certificate["scores_computed"] == 0
    assert certificate["physical_fit_calls"] == 0
    assert certificate["p100_accesses"] == 0


def test_certificate_rejects_duplicate_or_misaligned_panel_keys() -> None:
    panel = _synthetic_panel(march_truth=False)
    duplicate_index = panel.index.insert(1, panel.index[0])
    duplicate = SimpleNamespace(
        index=duplicate_index,
        temp=panel.temp.reindex(duplicate_index),
        baseline=panel.baseline.reindex(duplicate_index),
    )
    with pytest.raises(ValueError, match="duplicate"):
        guard.build_inner_identifiability_certificate(duplicate)

    misaligned = SimpleNamespace(
        index=panel.index,
        temp=panel.temp.iloc[1:],
        baseline=panel.baseline,
    )
    with pytest.raises(ValueError, match="index-aligned"):
        guard.build_inner_identifiability_certificate(misaligned)


def test_certificate_rejects_layer_surface_drift() -> None:
    panel = _synthetic_panel(march_truth=False)
    wrong_temp = panel.temp.drop(columns=[8])
    with pytest.raises(ValueError, match="temperature layer surface"):
        guard.build_inner_identifiability_certificate(
            SimpleNamespace(
                index=panel.index,
                temp=wrong_temp,
                baseline=panel.baseline,
            )
        )


def test_frozen_runtime_certificate_verifies_and_forces_stop() -> None:
    certificate = _runtime_certificate_from_frozen()
    resolution = guard.verify_frozen_certificate(certificate)
    assert resolution["status"] == "NO_GO_INNER_WINDOW_UNIDENTIFIABLE"
    assert resolution["claim_permitted"] is False
    assert resolution["score_permitted"] is False
    with pytest.raises(guard.InnerWindowUnidentifiable, match="preclaim stop"):
        guard.require_identifiable_or_stop(certificate)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("windows", 0, "rows_after_finite_truth_anchor_mask"), 1),
        (("windows", 1, "layer_finite_masks", "2", "finite_truth"), 3_348),
        (("scores_computed",), 1),
    ],
)
def test_frozen_certificate_fails_closed_on_count_or_operation_drift(
    path: tuple,
    value: object,
) -> None:
    certificate = _runtime_certificate_from_frozen()
    target = certificate
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        guard.verify_frozen_certificate(certificate)


def test_scientific_contract_is_unchanged_six_cells_twenty_two_slots() -> None:
    contract = guard.scientific_contract()
    assert contract["cells"] == [
        "d1_k3",
        "d1_k7",
        "d3_k3",
        "d3_k7",
        "d7_k3",
        "d7_k7",
    ]
    assert contract["inner_windows"] == [
        "inner_2024_mar",
        "inner_2024_may",
        "inner_2024_jul",
    ]
    assert contract["materialization_slots"] == 22
    assert contract["physical_fit_calls"] == 0
    assert contract["scientific_logic_changed"] is False


def test_error_envelope_is_sanitized_bounded_and_value_free() -> None:
    try:
        raise ValueError(
            "failed at C:\\private\\secret.csv\nsecond line"
        )
    except ValueError as error:
        envelope = guard.sanitized_error_envelope(error, phase="SLOT1")
    verification = guard.validate_error_envelope(envelope)
    assert verification["valid"] is True
    assert verification["bytes"] <= guard.MAX_ERROR_ENVELOPE_BYTES
    assert envelope["chain"][0]["type"] == "ValueError"
    assert envelope["chain"][0]["module"] == "builtins"
    assert "private" not in envelope["chain"][0]["message_sanitized"]
    assert "\n" not in envelope["chain"][0]["message_sanitized"]
    assert envelope["raw_traceback_persisted"] is False
    assert envelope["locals_captured"] is False
    assert envelope["prediction_values_captured"] == 0
    assert envelope["truth_values_captured"] == 0
    assert envelope["metric_values_captured"] == 0


def test_error_envelope_chain_and_frames_are_capped() -> None:
    error: BaseException = ValueError("root")
    for index in range(12):
        try:
            raise RuntimeError(f"level {index}") from error
        except RuntimeError as current:
            error = current
    envelope = guard.sanitized_error_envelope(error, phase="WORKER")
    assert len(envelope["chain"]) == guard.MAX_ERROR_CHAIN
    assert len(envelope["frames"]) <= guard.MAX_ERROR_FRAMES


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"raw_traceback_persisted": True}),
        lambda value: value.update({"metric_values_captured": 1}),
        lambda value: value["chain"][0].update({"message_sha256": "bad"}),
        lambda value: value.update({"unexpected": True}),
    ],
)
def test_error_envelope_mutations_fail_closed(mutation) -> None:
    try:
        raise RuntimeError("bounded")
    except RuntimeError as error:
        envelope = guard.sanitized_error_envelope(error, phase="WORKER")
    mutation(envelope)
    with pytest.raises(ValueError):
        guard.validate_error_envelope(envelope)


def test_windows_safe_error_receipt_publication_and_binding(tmp_path: Path) -> None:
    runner = _load_runner()
    try:
        raise RuntimeError("slot1 failed")
    except RuntimeError as error:
        envelope = guard.sanitized_error_envelope(error, phase="SLOT1")
    destination = tmp_path / "worker.error.json"
    published = runner._publish_error_envelope_create_only(
        destination,
        envelope,
        guard_module=guard,
    )
    assert published["durable_create_only"] is True
    assert destination.exists()
    bound = runner._bind_worker_error_receipt(
        destination,
        expected_sha256=published["sha256"],
        guard_module=guard,
    )
    assert bound["held_snapshot_bound"] is True
    assert bound["error"] == envelope
    assert (tmp_path / ".p2_dtw_v2r6_directory_fsync").exists()


def test_error_receipt_publication_is_create_only(tmp_path: Path) -> None:
    runner = _load_runner()
    try:
        raise RuntimeError("first")
    except RuntimeError as error:
        envelope = guard.sanitized_error_envelope(error, phase="WORKER")
    destination = tmp_path / "worker.error.json"
    runner._publish_error_envelope_create_only(
        destination,
        envelope,
        guard_module=guard,
    )
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        runner._publish_error_envelope_create_only(
            destination,
            envelope,
            guard_module=guard,
        )
    assert destination.read_bytes() == before


def test_error_receipt_link_failure_leaves_no_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    try:
        raise RuntimeError("fail link")
    except RuntimeError as error:
        envelope = guard.sanitized_error_envelope(error, phase="WORKER")
    destination = tmp_path / "worker.error.json"

    def fail_link(source, target):
        del source, target
        raise OSError("injected link failure")

    monkeypatch.setattr(runner.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        runner._publish_error_envelope_create_only(
            destination,
            envelope,
            guard_module=guard,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_error_receipt_hash_swap_is_rejected(tmp_path: Path) -> None:
    runner = _load_runner()
    try:
        raise RuntimeError("bind")
    except RuntimeError as error:
        envelope = guard.sanitized_error_envelope(error, phase="WORKER")
    destination = tmp_path / "worker.error.json"
    runner._publish_error_envelope_create_only(
        destination,
        envelope,
        guard_module=guard,
    )
    with pytest.raises(RuntimeError, match="digest"):
        runner._bind_worker_error_receipt(
            destination,
            expected_sha256="0" * 64,
            guard_module=guard,
        )


def test_runner_has_no_claim_or_worker_launch_implementation() -> None:
    runner = _load_runner()
    assert not hasattr(runner, "_acquire_attempt")
    assert not hasattr(runner, "_launch_worker")
    assert not hasattr(runner, "_run_internal_worker")
    assert not hasattr(runner, "_publish_aggregate")


def test_execute_flag_fails_closed_without_control_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    before = runner._control_state()
    monkeypatch.setattr(
        runner,
        "read_only_preflight",
        lambda: {"status": "NO_GO_INNER_WINDOW_UNIDENTIFIABLE"},
    )
    with pytest.raises(runner.BlockedExecutionError, match="cannot create a claim"):
        runner.execute_blocked()
    assert runner._control_state() == before


def test_actual_v2r6_control_namespace_is_clean() -> None:
    runner = _load_runner()
    state = runner._control_state()
    assert state["clean"] is True
    assert all(exists is False for exists in state["exists"].values())


def test_seal_bundle_and_permanent_false_authorization() -> None:
    runner = _load_runner()
    snapshot = runner._verify_seal()
    authorization = runner._verify_authorization(snapshot)
    assert snapshot.seal["status"] == (
        "SEALED_PRECLAIM_BLOCKED_NO_GO_INNER_WINDOW_UNIDENTIFIABLE"
    )
    assert authorization["status"] == "PERMANENTLY_NOT_AUTHORIZED_PRECLAIM_BLOCKED"
    assert authorization["authorized"] is False
    assert authorization["execution_permitted"] is False


def test_real_pinned_runtime_certificate_stops_before_claim() -> None:
    assert os.environ.get("P2_DATA_DIR"), "test requires command-scoped P2_DATA_DIR"
    runner = _load_runner()
    snapshot = runner._verify_seal()
    before = runner._control_state()
    result = runner._runtime_identifiability(snapshot)
    after = runner._control_state()
    assert result["status"] == "NO_GO_INNER_WINDOW_UNIDENTIFIABLE"
    assert result["blocking_window"] == "inner_2024_mar"
    assert result["blocking_rows_after_mask"] == 0
    assert result["claim_permitted"] is False
    assert result["materialization_permitted"] is False
    assert result["score_permitted"] is False
    assert result["historical_sources"]["p100"]["filesystem_accesses"] == 0
    assert all(value == 0 for value in result["operation_counters"].values())
    assert before == after


def test_read_only_preflight_is_idempotent_and_operation_zero() -> None:
    assert os.environ.get("P2_DATA_DIR"), "test requires command-scoped P2_DATA_DIR"
    runner = _load_runner()
    tracked = [
        runner.DESIGN_PATH,
        runner.TRIGGER_PATH,
        runner.EXECUTION_PATH,
        runner.MODULE_PATH,
        runner.TEST_PATH,
        runner.SLOT_DIAGNOSTIC_PATH,
        runner.IDENTIFIABILITY_PATH,
        runner.SEAL_PATH,
        runner.AUTHORIZATION_PATH,
    ]
    before = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in tracked
    }
    first = runner.read_only_preflight()
    second = runner.read_only_preflight()
    after = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in tracked
    }
    assert first == second
    assert before == after
    assert first["status"] == "NO_GO_INNER_WINDOW_UNIDENTIFIABLE"
    assert first["read_only"] is True
    assert first["preclaim_blocked"] is True
    assert first["authorized"] is False
    assert first["execution_permitted"] is False
    assert all(value == 0 for value in first["operation_counters"].values())
    assert first["control_state"]["clean"] is True
    assert first["identifiability"]["historical_sources"]["p100"][
        "filesystem_accesses"
    ] == 0


def test_authorization_swap_to_true_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    snapshot = runner._verify_seal()
    original = runner._read_snapshot
    authorization = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    authorization["authorized"] = True
    authorization["execution_permitted"] = True
    payload = runner._canonical_json_bytes(authorization)

    def swapped(path: Path):
        if path.resolve() == AUTH_PATH.resolve():
            return payload, hashlib.sha256(payload).hexdigest()
        return original(path)

    monkeypatch.setattr(runner, "_read_snapshot", swapped)
    with pytest.raises(runner.BlockedExecutionError):
        runner._verify_authorization(snapshot)


def test_static_sources_never_embed_personal_data_path_or_p100_literal() -> None:
    combined = MODULE_PATH.read_text(encoding="utf-8") + RUNNER_PATH.read_text(
        encoding="utf-8"
    )
    assert "C:\\Users\\" not in combined
    assert "evaluated_oof_100.parquet" not in combined
    assert "--internal-worker" not in combined


def test_no_result_candidate_or_submission_artifact_is_created() -> None:
    runner = _load_runner()
    assert not runner.FINAL_DIR.exists()
    assert not runner.STAGING_PATH.exists()
    assert not runner.ERROR_RECEIPT_PATH.exists()
    assert not runner.CLAIM_PATH.exists()
    assert not runner.JOURNAL_PATH.exists()
    assert not runner.TERMINAL_PATH.exists()
