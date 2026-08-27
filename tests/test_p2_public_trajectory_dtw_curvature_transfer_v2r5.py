from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from p2_restore import public_trajectory_dtw_v2r5 as dtw

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/run_p2_public_trajectory_dtw_curvature_transfer_v2r5.py"
DESIGN_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r5_design.json"
DESIGN_SHA256 = "c044ae23d14f85c634d8145cbd8f85b004536e378277dc91dc00097dc78f7fe4"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p2_dtw_runner_under_test", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
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


def _synthetic_verified_snapshot(runner, *, seal_bytes: bytes = b'{"seal":"A"}\n'):
    design_bytes = DESIGN_PATH.read_bytes()
    execution_bytes = (
        REPO_ROOT
        / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r5_execution.json"
    ).read_bytes()
    module_bytes = (REPO_ROOT / "src/p2_restore/public_trajectory_dtw_v2r5.py").read_bytes()
    bundle_bytes = {
        runner.DESIGN_RELATIVE: design_bytes,
        runner.EXECUTION_CONFIG_RELATIVE: execution_bytes,
        runner.MODULE_RELATIVE: module_bytes,
        runner.TRIGGER_RELATIVE: b"synthetic-trigger\n",
        runner.RUNNER_RELATIVE: b"synthetic-runner\n",
        runner.TEST_RELATIVE: b"synthetic-test\n",
        runner.CLOSURE_MATRIX_RELATIVE: b"synthetic-closure\n",
        runner.V2R4_QA_RELATIVE: b"synthetic-v2r4-qa\n",
        runner.DIAGNOSIS_RELATIVE: b"synthetic-diagnosis\n",
    }
    return runner.VerifiedSnapshot(
        design_bytes=design_bytes,
        design_digest=hashlib.sha256(design_bytes).hexdigest(),
        execution_bytes=execution_bytes,
        execution_digest=hashlib.sha256(execution_bytes).hexdigest(),
        seal_bytes=seal_bytes,
        seal_digest=hashlib.sha256(seal_bytes).hexdigest(),
        module_bytes=module_bytes,
        module_digest=hashlib.sha256(module_bytes).hexdigest(),
        bundle_bytes=bundle_bytes,
        verification={},
    )


def _authorization_static_pins(runner, snapshot) -> dict[str, dict[str, int | str]]:
    relatives = {
        runner.DESIGN_RELATIVE,
        runner.TRIGGER_RELATIVE,
        runner.EXECUTION_CONFIG_RELATIVE,
        runner.RUNNER_RELATIVE,
        runner.MODULE_RELATIVE,
        runner.TEST_RELATIVE,
        runner.SEAL_RELATIVE,
        runner.CLOSURE_MATRIX_RELATIVE,
        runner.V2R4_QA_RELATIVE,
        runner.DIAGNOSIS_RELATIVE,
    }
    pins = {}
    for relative in relatives:
        payload = (
            snapshot.seal_bytes
            if relative == runner.SEAL_RELATIVE
            else snapshot.bundle_bytes[relative]
        )
        pins[relative] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return pins


def test_design_hash_is_immutable_and_not_authorized() -> None:
    assert _sha256(DESIGN_PATH) == DESIGN_SHA256
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    assert design["schema"] == "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
    assert design["status"] == "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
    assert design["unchanged_scientific_contract"]["model_fit_calls"] == 0
    freeze = design["preimplementation_freeze"]
    assert freeze["v2r5_materializations_at_freeze"] == 0
    assert freeze["v2r5_p100_filesystem_accesses_at_freeze"] == 0
    assert freeze["v2r4_scientific_materializations"] == 0
    amendment = design["integrity_adapter_amendment"]
    assert amendment["full_continuous_presence_required"] is False
    assert amendment["canonical_step_ns"] == 600_000_000_000
    assert amendment["all_other_v2r4_numerical_logic_changed"] is False


def test_v2r5_numerical_module_diff_is_integrity_adapter_only() -> None:
    v2r4 = (REPO_ROOT / "src/p2_restore/public_trajectory_dtw_v2r4.py").read_text(
        encoding="utf-8"
    )
    v2r5 = (REPO_ROOT / "src/p2_restore/public_trajectory_dtw_v2r5.py").read_text(
        encoding="utf-8"
    )
    assert v2r5.count("v2r5") == 1
    allowed = {"exact_time_contract", "exact_sparse_key_contract", "verify_anchor_against_observations"}

    def frozen_ast(source: str) -> str:
        tree = ast.parse(source)
        body = [
            node
            for index, node in enumerate(tree.body)
            if index != 0
            and not (isinstance(node, ast.FunctionDef) and node.name in allowed)
        ]
        return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)

    assert frozen_ast(v2r5) == frozen_ast(v2r4)
    v2r4_functions = {
        node.name for node in ast.parse(v2r4).body if isinstance(node, ast.FunctionDef)
    }
    v2r5_functions = {
        node.name for node in ast.parse(v2r5).body if isinstance(node, ast.FunctionDef)
    }
    assert v2r5_functions - v2r4_functions == {"exact_sparse_key_contract"}


def test_v2r5_active_namespace_has_no_stale_v2r4_binding() -> None:
    runner = _load_runner()
    active = {
        "experiment": runner.EXPERIMENT_ID,
        "design": runner.DESIGN_RELATIVE,
        "execution": runner.EXECUTION_CONFIG_RELATIVE,
        "runner": runner.RUNNER_RELATIVE,
        "module": runner.MODULE_RELATIVE,
        "test": runner.TEST_RELATIVE,
        "seal": runner.SEAL_RELATIVE,
        "closure": runner.CLOSURE_MATRIX_RELATIVE,
        "authorization_env": runner.AUTHORIZATION_ENV,
    }
    assert all("v2r5" in value.lower() for value in active.values())
    assert "v2r4" not in dtw.__doc__.lower()
    assert runner.V2R4_QA_RELATIVE.endswith(
        "p2_public_trajectory_dtw_curvature_transfer_v2r4_independent_preexecution_qa_20260826.json"
    )


def test_design_and_execution_parse_their_one_captured_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    design_path = tmp_path / "design.json"
    execution_path = tmp_path / "execution.json"
    design_path.write_bytes(runner.DESIGN_PATH.read_bytes())
    execution_path.write_bytes(runner.EXECUTION_CONFIG_PATH.read_bytes())
    monkeypatch.setattr(runner, "DESIGN_PATH", design_path)
    monkeypatch.setattr(runner, "EXECUTION_CONFIG_PATH", execution_path)
    real_snapshot = runner._read_byte_snapshot
    opened = []

    def capture_then_toggle(path: Path):
        payload, digest = real_snapshot(path)
        opened.append(path.name)
        path.write_text('{"toggled_after_snapshot":true}\n', encoding="utf-8")
        return payload, digest

    monkeypatch.setattr(runner, "_read_byte_snapshot", capture_then_toggle)
    design, design_bytes, design_digest, execution, execution_bytes, execution_digest = (
        runner._verify_design()
    )
    assert opened == ["design.json", "execution.json"]
    assert design["experiment_id"] == runner.EXPERIMENT_ID
    assert execution["experiment_id"] == runner.EXPERIMENT_ID
    assert hashlib.sha256(design_bytes).hexdigest() == design_digest
    assert hashlib.sha256(execution_bytes).hexdigest() == execution_digest
    assert json.loads(design_path.read_text(encoding="utf-8"))["toggled_after_snapshot"]
    assert json.loads(execution_path.read_text(encoding="utf-8"))["toggled_after_snapshot"]


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


def test_v2r4_qa_and_zero_materialization_failure_lineage_are_pinned() -> None:
    report_path = (
        REPO_ROOT
        / "reports/p2_public_trajectory_dtw_curvature_transfer_v2r4_independent_preexecution_qa_20260826.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert _sha256(report_path) == "0a3861151e0c3367a66760e552aa4ce2f04291f0704a10c80ae21cd517dadcc9"
    assert report["verdict"] == "PASS"
    assert report["severity_counts"] == {"P0": 0, "P1": 0, "P2": 1}
    claim_path = (
        REPO_ROOT
        / "artifacts/_p2_trajectory_claims/p2_public_trajectory_dtw_curvature_transfer_20260826_v2r4.claim.json"
    )
    journal_path = (
        REPO_ROOT
        / "artifacts/_p2_trajectory_attempt_journals/p2_public_trajectory_dtw_curvature_transfer_20260826_v2r4.ndjson"
    )
    assert _sha256(claim_path) == "003916cd1b062f97be39e85d422f954c25895519fecacbc9f4a7d8dbd53e2062"
    assert _sha256(journal_path) == "10f2efc207c3a5d768da630a8c89b9dfc92fb3b8bfee89ffede8f213bd817f69"
    events = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == [
        "ATTEMPT_CLAIMED",
        "PARENT_LAUNCHING_SINGLE_WORKER",
        "ATTEMPT_TERMINAL_FAILED",
    ]
    accounting = events[-1]["materialization_accounting"]
    assert {key: accounting[key] for key in ("reserved", "completed", "failed", "skipped_gate")} == {
        "reserved": 0,
        "completed": 0,
        "failed": 0,
        "skipped_gate": 0,
    }


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
        "full_grid_timestamps": 8784,
        "missing_union_timestamps": 0,
        "gap_intervals": 0,
        "off_grid_timestamps": 0,
        "canonical_step_ns": 600_000_000_000,
        "continuous_presence_required": False,
        "unit": "ns",
        "roundtrip_identity": True,
        "nat_count": 0,
    }


def test_sparse_multilayer_exact_keys_allow_missing_presence_but_require_grid() -> None:
    full = pd.date_range(
        "2024-09-01T00:00:00+09:00",
        "2024-10-31T23:50:00+09:00",
        freq="10min",
    )
    rows = []
    removals = {2: {100, 400}, 3: {200, 201, 400}, 4: {300, 301, 302, 400}}
    for layer in dtw.TARGET_LAYERS:
        keep = np.ones(len(full), dtype=bool)
        keep[list(removals[layer])] = False
        rows.append(pd.DataFrame({"time": full[keep], "layer": layer}))
    sparse = pd.concat(rows, ignore_index=True)
    contract = dtw.exact_sparse_key_contract(sparse, expected_rows=len(sparse))
    assert contract["duplicate_time_layer_keys"] == 0
    assert contract["unique_timestamps"] == 8783
    assert contract["missing_union_timestamps"] == 1
    assert contract["gap_intervals"] == 1
    assert contract["off_grid_timestamps"] == 0
    assert contract["layer_rows"] == {"2": 8782, "3": 8781, "4": 8780}
    assert contract["layer_missing_from_full_grid"] == {"2": 2, "3": 3, "4": 4}

    duplicated = pd.concat([sparse, sparse.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        dtw.exact_sparse_key_contract(duplicated, expected_rows=len(duplicated))

    off_grid = sparse.copy()
    off_grid.loc[1, "time"] = pd.Timestamp(off_grid.loc[1, "time"]) + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="off the frozen"):
        dtw.exact_sparse_key_contract(off_grid, expected_rows=len(off_grid))

    wrong_layer = sparse.copy()
    wrong_layer.loc[wrong_layer["layer"] == 4, "layer"] = 5
    with pytest.raises(ValueError, match="layer set"):
        dtw.exact_sparse_key_contract(wrong_layer, expected_rows=len(wrong_layer))

    missing_bound = sparse.loc[pd.to_datetime(sparse["time"]) != full[0]].copy()
    with pytest.raises(ValueError, match="bounds"):
        dtw.exact_sparse_key_contract(missing_bound, expected_rows=len(missing_bound))


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


def test_real_pinned_exact_anchor_initializes_without_materialization_or_p100() -> None:
    raw_root = os.environ.get("P2_DATA_DIR")
    assert raw_root, "test requires command-scoped P2_DATA_DIR"
    observation_path = Path(raw_root).resolve(strict=True) / "observations.csv"
    exact_path = REPO_ROOT / "artifacts/p2_extrapolated_soft_gate_v2/oof.parquet"
    assert observation_path.stat().st_size == 49_058_719
    assert _sha256(observation_path) == "cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a"
    assert exact_path.stat().st_size == 2_477_660
    assert _sha256(exact_path) == "dab52579e99a20cc0444bf13bc3a1400191024a10303cb996ba59a89509c9cb4"
    observations = pd.read_csv(
        observation_path,
        dtype={"station": "string", "time": "string"},
    )
    exact = pd.read_parquet(exact_path)
    assert len(observations) == 789_408
    assert len(exact) == 69_850
    materializer = dtw.HistoricalTrajectoryMaterializer(observations, exact)
    certificate = materializer.exact_truth_certificate
    contract = certificate["time_contract"]
    assert certificate["rows"] == 26_273
    assert certificate["anchor_truth_equal_observations"] is True
    assert contract["unique_time_layer_keys"] == 26_273
    assert contract["duplicate_time_layer_keys"] == 0
    assert contract["unique_timestamps"] == 8_779
    assert contract["full_grid_timestamps"] == 8_784
    assert contract["missing_union_timestamps"] == 5
    assert contract["gap_intervals"] == 2
    assert contract["off_grid_timestamps"] == 0
    assert contract["layer_rows"] == {"2": 8_777, "3": 8_774, "4": 8_722}
    assert contract["layer_missing_from_full_grid"] == {"2": 7, "3": 10, "4": 62}
    assert contract["kst_days"] == 61
    assert materializer.p100_anchor is None
    assert materializer.p100_truth_certificate is None


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


def test_portable_observations_resolver_requires_env_and_exact_relative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    monkeypatch.delenv(runner.P2_DATA_DIR_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="P2_DATA_DIR"):
        runner._resolve_observations_path()

    monkeypatch.setenv(runner.P2_DATA_DIR_ENV, "relative/history")
    with pytest.raises(ValueError, match="absolute"):
        runner._resolve_observations_path()

    root = tmp_path / "portable_history"
    root.mkdir()
    observations = root / "observations.csv"
    observations.write_bytes(b"historical-only\n")
    monkeypatch.setenv(runner.P2_DATA_DIR_ENV, str(root.resolve()))
    assert runner._resolve_observations_path() == observations.resolve()

    resolver_source = RUNNER_PATH.read_text(encoding="utf-8").split(
        "def _resolve_observations_path", 1
    )[1].split("def _parse_aware_datetime", 1)[0]
    assert ".iterdir(" not in resolver_source
    assert ".glob(" not in resolver_source
    assert ".rglob(" not in resolver_source


def test_portable_observations_resolver_rejects_official_token_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    root = tmp_path / "sample_submission_history"
    root.mkdir()
    (root / "observations.csv").write_bytes(b"must-not-open\n")
    monkeypatch.setenv(runner.P2_DATA_DIR_ENV, str(root.resolve()))
    with pytest.raises(ValueError, match="official-path firewall"):
        runner._resolve_observations_path()


def test_portable_observations_resolver_rejects_symlink_escape_if_supported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    root = tmp_path / "root"
    outside = tmp_path / "outside.csv"
    root.mkdir()
    outside.write_bytes(b"outside\n")
    try:
        (root / "observations.csv").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows account")
    monkeypatch.setenv(runner.P2_DATA_DIR_ENV, str(root.resolve()))
    with pytest.raises(ValueError, match="official-path firewall"):
        runner._resolve_observations_path()


def test_no_personal_absolute_observations_path_in_v2r5_static_sources() -> None:
    relatives = [
        "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r5_design.json",
        "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r5_execution.json",
        "scripts/run_p2_public_trajectory_dtw_curvature_transfer_v2r5.py",
    ]
    combined = "\n".join((REPO_ROOT / relative).read_text(encoding="utf-8") for relative in relatives)
    normalized = combined.lower().replace("\\", "/")
    assert "c:/users/" not in normalized
    assert "downloads/" not in normalized
    assert "p2_data_dir" in normalized
    assert "observations.csv" in normalized


def test_pre_exact_source_readiness_never_verifies_p100(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    observed = []

    class LiteralOnlyP100:
        def as_posix(self) -> str:
            return "artifacts/sealed/history/evaluated_oof_100.parquet"

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
    monkeypatch.setattr(runner, "P100_ANCHOR_RELATIVE", p100_path)
    paths = {
        "observations": (tmp_path / "observations.csv", {"bytes": 1, "sha256": "a" * 64}),
        "exact_anchor": (tmp_path / "exact.parquet", {"bytes": 1, "sha256": "b" * 64}),
    }
    p100_literal = {
        "literal_relative_path": p100_path.as_posix(),
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


def test_real_initialization_readiness_precedes_claim_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    snapshot = _synthetic_verified_snapshot(runner)
    seal_digest = snapshot.seal_digest
    events = []
    monkeypatch.setattr(
        runner,
        "_verify_seal",
        lambda: ({}, seal_digest, snapshot),
    )
    monkeypatch.setattr(
        runner,
        "_authorization_state",
        lambda **kwargs: ({"authorized": True}, "a" * 64),
    )
    monkeypatch.setattr(
        runner,
        "_source_readiness",
        lambda seal: events.append("source_hash_readiness"),
    )
    monkeypatch.setattr(
        runner,
        "_real_exact_initialization_readiness",
        lambda seal, observed_snapshot: events.append("real_initialization"),
    )

    def stop_at_claim(authorization_hash: str, observed_seal_digest: str):
        assert authorization_hash == "a" * 64
        assert observed_seal_digest == seal_digest
        assert events == ["source_hash_readiness", "real_initialization"]
        raise RuntimeError("stop-at-claim")

    monkeypatch.setattr(runner, "_acquire_attempt", stop_at_claim)
    with pytest.raises(RuntimeError, match="stop-at-claim"):
        runner._execute_parent()
    assert events == ["source_hash_readiness", "real_initialization"]


def test_claim_creation_failure_performs_no_fit_or_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    _patch_attempt_paths(monkeypatch, runner, tmp_path)

    def fail_claim(path, value):
        del path, value
        raise OSError("claim-build-failure")

    with pytest.raises(OSError, match="claim-build-failure"):
        runner._acquire_attempt("a" * 64, "f" * 64, create_json=fail_claim)
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
        runner._acquire_attempt("b" * 64, "f" * 64, append_event=fail_journal)
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
        seal_digest="f" * 64,
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
            seal_digest="f" * 64,
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
        runner._build_parent_result(
            {"physical_fit_calls": 0},
            attempt_id=attempt_id,
            seal_digest="f" * 64,
        )
    runner._record_terminal_failure(
        caught.value,
        attempt_id=attempt_id,
        phase="RESULT_BUILD",
        seal_digest="f" * 64,
    )
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
            {"status": "aggregate-only", "physical_fit_calls": 0, "seal_sha256": "f" * 64},
            attempt_id="publication-failure",
            seal_digest="f" * 64,
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
        {"status": "aggregate-only", "physical_fit_calls": 0, "seal_sha256": "f" * 64},
        attempt_id=attempt,
        seal_digest="f" * 64,
        sync_directory=logged_sync,
        terminal_link_fn=terminal_link,
    )
    assert io_events[-1] == ("TERMINAL_LINK", "terminal_success.json")
    assert publication["post_terminal_repository_io"] == 0
    assert runner._terminal_state()["logical_state"] == "SUCCESS"
    assert json.loads((runner.FINAL_DIR / "manifest.json").read_text(encoding="utf-8"))[
        "seal_sha256"
    ] == "f" * 64
    assert json.loads(
        (runner.FINAL_DIR / "terminal_success.json").read_text(encoding="utf-8")
    )["seal_sha256"] == "f" * 64
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
            {"status": "aggregate-only", "physical_fit_calls": 0, "seal_sha256": "f" * 64},
            attempt_id=attempt,
            seal_digest="f" * 64,
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
            seal_digest="f" * 64,
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


def test_terminal_success_cannot_cross_bind_a_different_seal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    _patch_attempt_paths(monkeypatch, runner, tmp_path)
    attempt = "terminal-cross-seal"
    ready = runner._event(
        "ATTEMPT_COMMIT_READY",
        attempt,
        seal_sha256="a" * 64,
    )
    runner._append_journal(runner.JOURNAL_PATH, ready)
    runner.FINAL_DIR.mkdir(parents=True)
    runner._exclusive_create_json(
        runner.FINAL_DIR / "terminal_success.json",
        {
            "experiment_id": runner.EXPERIMENT_ID,
            "attempt_id": attempt,
            "seal_sha256": "b" * 64,
            "commit_ready_event_sha256": runner._sha256_bytes(
                runner._canonical_bytes(ready)
            ),
        },
    )
    with pytest.raises(RuntimeError, match="identity differs"):
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
    source = RUNNER_PATH.read_text(encoding="utf-8") + (REPO_ROOT / "src/p2_restore/public_trajectory_dtw_v2r5.py").read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "PHYSICAL_FIT_CALLS = 0" in source


def test_authorized_worker_compiles_retained_module_without_path_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    source = RUNNER_PATH.read_text(encoding="utf-8")
    worker = source[
        source.index("def _load_sealed_numerical_module") :
        source.index("def _run_internal_worker")
    ]
    assert "spec_from_file_location" not in worker
    assert "_held_verified_bytes" not in worker
    assert "snapshot.module_bytes" in worker
    assert "from p2_restore" not in worker
    snapshot = _synthetic_verified_snapshot(runner)
    real_open = Path.open
    module_opens = 0

    def counted_open(path: Path, *args, **kwargs):
        nonlocal module_opens
        if path.resolve() == runner.MODULE_PATH.resolve():
            module_opens += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    loaded = runner._load_sealed_numerical_module(snapshot)
    assert module_opens == 0
    assert loaded.CELLS[0].cell_id == "d1_k3"
    assert Path(loaded.__file__).resolve() == runner.MODULE_PATH.resolve()


def test_module_compilation_ignores_path_mutation_after_seal_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    snapshot = _synthetic_verified_snapshot(runner)
    module_path = tmp_path / "toggled_module.py"
    module_path.write_text("raise RuntimeError('reopened mutated path')\n", encoding="utf-8")
    monkeypatch.setattr(runner, "MODULE_PATH", module_path)
    loaded = runner._load_sealed_numerical_module(snapshot)
    assert loaded.CELLS[-1].cell_id == "d7_k7"
    assert module_path.read_text(encoding="utf-8").startswith("raise RuntimeError")


@pytest.mark.parametrize(
    "corruption",
    ["missing_report", "wrong_bytes", "wrong_sha", "wrong_verdict", "wrong_design", "wrong_seal"],
)
def test_authorization_verifies_real_qa_receipt_and_snapshot_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    corruption: str,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    snapshot = _synthetic_verified_snapshot(runner)
    seal_digest = snapshot.seal_digest
    qa_path = tmp_path / "reports/p2_v2r5_independent_qa.json"
    qa_path.parent.mkdir(parents=True)
    qa_report = {
        "experiment_id": runner.EXPERIMENT_ID,
        "verdict": "PASS",
        "design_sha256": snapshot.design_digest,
        "seal_sha256": seal_digest,
    }
    qa_path.write_text(json.dumps(qa_report, sort_keys=True) + "\n", encoding="utf-8")
    qa_pin = {
        "path": str(qa_path.relative_to(tmp_path)).replace("\\", "/"),
        "bytes": qa_path.stat().st_size,
        "sha256": _sha256(qa_path),
        "verdict": "PASS",
        "design_sha256": snapshot.design_digest,
        "seal_sha256": seal_digest,
    }
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
        "schema_version": "p2_public_trajectory_dtw_v2r5.authorization.v1",
        "status": "AUTHORIZED_AFTER_INDEPENDENT_QA",
        "experiment_id": runner.EXPERIMENT_ID,
        "authorized": True,
        "design_sha256": snapshot.design_digest,
        "trigger_resolution_sha256": runner.TRIGGER_SHA256,
        "seal_sha256": seal_digest,
        "bundle": {
            "static_files": _authorization_static_pins(runner, snapshot),
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
        runner._authorization_state(
            require_authorized=True,
            seal_digest=seal_digest,
            snapshot=snapshot,
        )


def test_authorization_uses_one_snapshot_even_if_auth_path_swaps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    snapshot = _synthetic_verified_snapshot(runner)
    seal_digest = snapshot.seal_digest
    qa_path = tmp_path / "reports/p2_v2r5_independent_qa.json"
    qa_path.parent.mkdir(parents=True)
    qa_report = {
        "experiment_id": runner.EXPERIMENT_ID,
        "verdict": "PASS",
        "design_sha256": snapshot.design_digest,
        "seal_sha256": seal_digest,
    }
    qa_path.write_text(json.dumps(qa_report, sort_keys=True) + "\n", encoding="utf-8")
    qa_pin = {
        "path": str(qa_path.relative_to(tmp_path)).replace("\\", "/"),
        "bytes": qa_path.stat().st_size,
        "sha256": _sha256(qa_path),
        "verdict": "PASS",
        "design_sha256": snapshot.design_digest,
        "seal_sha256": seal_digest,
    }
    authorization = {
        "schema_version": "p2_public_trajectory_dtw_v2r5.authorization.v1",
        "status": "AUTHORIZED_AFTER_INDEPENDENT_QA",
        "experiment_id": runner.EXPERIMENT_ID,
        "authorized": True,
        "design_sha256": snapshot.design_digest,
        "trigger_resolution_sha256": runner.TRIGGER_SHA256,
        "seal_sha256": seal_digest,
        "bundle": {
            "static_files": _authorization_static_pins(runner, snapshot),
            "independent_qa_report": qa_pin,
        },
        "operation_ceiling": _operation_ceiling(),
        "independent_qa": qa_pin,
        "blockers": [],
    }
    auth_path = tmp_path / "authorization.json"
    auth_path.write_text(json.dumps(authorization, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "AUTHORIZATION_PATH", auth_path)
    original_hash = _sha256(auth_path)
    monkeypatch.setenv(runner.AUTHORIZATION_ENV, original_hash)
    real_snapshot = runner._read_byte_snapshot
    snapshot_calls = 0

    def read_once_then_mutate(path: Path):
        nonlocal snapshot_calls
        snapshot_calls += 1
        payload, digest = real_snapshot(path)
        path.write_text('{"tampered_after_snapshot":true}\n', encoding="utf-8")
        return payload, digest

    monkeypatch.setattr(runner, "_read_byte_snapshot", read_once_then_mutate)
    observed, observed_hash = runner._authorization_state(
        require_authorized=True,
        seal_digest=seal_digest,
        snapshot=snapshot,
    )
    assert snapshot_calls == 1
    assert observed["authorized"] is True
    assert observed_hash == original_hash
    assert json.loads(auth_path.read_text(encoding="utf-8")) == {
        "tampered_after_snapshot": True
    }


def test_authorization_code_never_hashes_then_reopens_authorization_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    function = source[source.index("def _authorization_state") : source.index("@contextmanager")]
    assert "_read_byte_snapshot(AUTHORIZATION_PATH)" in function
    assert "_sha256(AUTHORIZATION_PATH)" not in function
    assert "_read_json(AUTHORIZATION_PATH)" not in function
    assert "_sha256(SEAL_PATH)" not in source
    assert "_read_json(SEAL_PATH)" not in source


def test_cross_seal_path_toggle_cannot_change_auth_or_claim_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    _, seal_a_digest, snapshot_a = runner._verify_seal()
    seal_b_path = tmp_path / "seal_b.json"
    seal_b_path.write_text('{"different_unvalidated_seal":"B"}\n', encoding="utf-8")
    seal_b_digest = _sha256(seal_b_path)
    assert seal_b_digest != seal_a_digest
    monkeypatch.setattr(runner, "SEAL_PATH", seal_b_path)

    authorization, authorization_hash = runner._authorization_state(
        require_authorized=False,
        seal_digest=seal_a_digest,
        snapshot=snapshot_a,
    )
    assert authorization["seal_sha256"] == seal_a_digest
    captured_claims = []
    captured_events = []
    monkeypatch.setattr(
        runner,
        "_control_state",
        lambda: {"clean_for_single_attempt": True},
    )

    def capture_claim(path, value):
        del path
        captured_claims.append(dict(value))
        return True

    def capture_event(path, value):
        del path
        captured_events.append(dict(value))

    claim = runner._acquire_attempt(
        authorization_hash,
        seal_a_digest,
        create_json=capture_claim,
        append_event=capture_event,
    )
    assert claim["seal_sha256"] == seal_a_digest
    assert captured_claims[0]["seal_sha256"] == seal_a_digest
    assert captured_events[0]["seal_sha256"] == seal_a_digest
    assert seal_b_digest not in json.dumps([captured_claims, captured_events])


def test_actual_kst_clock_and_synthetic_filesystem_chronology_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    observed_now = datetime.fromisoformat(runner._now_kst())
    assert observed_now.tzinfo is not None
    assert observed_now.utcoffset() == timedelta(hours=9)
    assert observed_now <= datetime.now(UTC).astimezone(ZoneInfo("Asia/Seoul"))

    diagnosis_path = tmp_path / runner.DIAGNOSIS_RELATIVE
    design_path = tmp_path / "design.json"
    execution_path = tmp_path / "execution.json"
    module_path = tmp_path / "module.py"
    runner_path = tmp_path / "runner.py"
    test_path = tmp_path / "test.py"
    closure_path = tmp_path / runner.CLOSURE_MATRIX_RELATIVE
    seal_path = tmp_path / "seal.json"
    paths = [
        diagnosis_path,
        design_path,
        execution_path,
        module_path,
        runner_path,
        test_path,
        closure_path,
        seal_path,
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen\n", encoding="utf-8")

    base_ns = time.time_ns() - 30_000_000_000
    ordered_ns = [base_ns + index * 1_000_000_000 for index in range(len(paths))]
    for path, mtime_ns in zip(paths, ordered_ns, strict=True):
        os.utime(path, ns=(mtime_ns, mtime_ns))

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "DESIGN_PATH", design_path)
    monkeypatch.setattr(runner, "EXECUTION_CONFIG_PATH", execution_path)
    monkeypatch.setattr(runner, "MODULE_PATH", module_path)
    monkeypatch.setattr(runner, "TEST_PATH", test_path)
    monkeypatch.setattr(runner, "SEAL_PATH", seal_path)
    monkeypatch.setattr(runner, "__file__", str(runner_path))

    def to_kst(ns: int) -> str:
        return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC).astimezone(
            ZoneInfo("Asia/Seoul")
        ).isoformat()
    design = {"created_at_kst": to_kst(base_ns + 500_000_000)}
    seal = {
        "chronology": {
            "implementation_completed_at_kst": to_kst(base_ns + 4_500_000_000),
            "tests_completed_at_kst": to_kst(base_ns + 5_500_000_000),
            "seal_created_at_kst": to_kst(base_ns + 6_500_000_000),
            "filesystem_mtime_ns": {
                runner.DIAGNOSIS_RELATIVE: ordered_ns[0],
                runner.DESIGN_RELATIVE: ordered_ns[1],
                runner.EXECUTION_CONFIG_RELATIVE: ordered_ns[2],
                runner.MODULE_RELATIVE: ordered_ns[3],
                runner.RUNNER_RELATIVE: ordered_ns[4],
                runner.TEST_RELATIVE: ordered_ns[5],
                runner.CLOSURE_MATRIX_RELATIVE: ordered_ns[6],
            },
        }
    }
    verified = runner._verify_chronology(design, seal)
    assert verified["nondecreasing"] is True
    assert verified["future_timestamp_count"] == 0
    assert "verified_at_kst" not in verified

    future = json.loads(json.dumps(seal))
    future["chronology"]["seal_created_at_kst"] = (
        datetime.now(UTC) + timedelta(days=1)
    ).astimezone(ZoneInfo("Asia/Seoul")).isoformat()
    with pytest.raises(RuntimeError, match="future-dated"):
        runner._verify_chronology(design, future)

    os.utime(module_path, ns=(ordered_ns[2] + 123, ordered_ns[2] + 123))
    with pytest.raises(RuntimeError, match="mtime pin changed"):
        runner._verify_chronology(design, seal)


def test_seal_transitive_inventory_is_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    seal, seal_digest, snapshot = runner._verify_seal()
    verification = snapshot.verification
    assert seal_digest == snapshot.seal_digest
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    assert seal["transitive_static_inputs"] == design["transitive_static_inputs"]
    assert len(verification["transitive_static_inputs"]) == 22
    assert verification["predecessor_semantics"]["terminal_event"] == "ATTEMPT_TERMINAL_FAILED"
    corrupted = dict(seal)
    corrupted["transitive_static_inputs"] = seal["transitive_static_inputs"][:-1]
    real_snapshot = runner._read_byte_snapshot
    corrupted_bytes = runner._canonical_bytes(corrupted)

    def read_snapshot(path):
        if Path(path).resolve() == runner.SEAL_PATH.resolve():
            return corrupted_bytes, hashlib.sha256(corrupted_bytes).hexdigest()
        return real_snapshot(path)

    monkeypatch.setattr(runner, "_read_byte_snapshot", read_snapshot)
    with pytest.raises(ValueError, match="transitive inventory"):
        runner._verify_seal()


def test_external_authorization_is_false_and_preflight_is_read_only() -> None:
    runner = _load_runner()
    preflight = runner.read_only_preflight()
    assert preflight["status"] == "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
    assert preflight["read_only"] is True
    assert preflight["authorized"] is False
    assert all(value == 0 for value in preflight["operation_counters"].values())
    _, seal_digest, snapshot = runner._verify_seal()
    with pytest.raises(runner.AuthorizationError):
        runner._authorization_state(
            require_authorized=True,
            seal_digest=seal_digest,
            snapshot=snapshot,
        )


def test_read_only_preflight_is_exactly_idempotent_and_mtime_preserving() -> None:
    runner = _load_runner()
    assert os.environ.get(runner.P2_DATA_DIR_ENV), "test requires command-scoped P2_DATA_DIR"
    tracked = [
        runner.DESIGN_PATH,
        runner.EXECUTION_CONFIG_PATH,
        runner.MODULE_PATH,
        RUNNER_PATH,
        runner.TEST_PATH,
        runner.SEAL_PATH,
        runner.AUTHORIZATION_PATH,
        REPO_ROOT / runner.CLOSURE_MATRIX_RELATIVE,
    ]
    before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in tracked}
    control_before = runner._control_state()
    first = runner.read_only_preflight()
    second = runner.read_only_preflight()
    after = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in tracked}
    control_after = runner._control_state()

    assert first == second
    assert before == after
    assert control_before == control_after
    assert first["read_only"] is True
    assert first["authorized"] is False
    assert first["control_state"]["clean_for_single_attempt"] is True
    assert first["historical_source_readiness"]["p100_anchor"]["filesystem_accesses"] == 0
    assert first["real_exact_initialization_readiness"]["status"] == (
        "PASS_REAL_PINNED_INITIALIZATION_ONLY"
    )
    assert first["real_exact_initialization_readiness"]["materialize_calls"] == 0
    assert first["real_exact_initialization_readiness"]["p100_accesses"] == 0
    assert first["static_verification"]["chronology"]["nondecreasing"] is True
    assert all(value == 0 for value in first["operation_counters"].values())
