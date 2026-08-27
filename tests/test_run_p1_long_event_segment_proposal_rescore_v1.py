from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p1_long_event_segment_proposal_rescore_v1.py"
MODULE_PATH = ROOT / "src" / "p1_qc" / "long_event_segment_proposal_rescore.py"
DESIGN_PATH = (
    ROOT / "configs" / "experiments" / "p1_long_event_segment_proposal_rescore_v1_design.json"
)
AMENDMENT_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v2_operational_amendment.json"
)
DESIGN_SHA256 = "31b0bde27d8ef7e2b42135709563cca0bcca61c6ec6fdabefbb3530906869563"
AMENDMENT_SHA256 = "b33f7d386e05cd7ab79976f58e9f4ab752f37cfe6a8856849867ef5f541cb276"


def _load(path: Path, name: str):
    source_root = str(ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def numerical():
    return _load(MODULE_PATH, "p1_qc.long_event_segment_proposal_rescore_test")


@pytest.fixture(scope="module")
def runner():
    return _load(RUNNER_PATH, "p1_long_event_segment_runner_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _design() -> dict:
    return json.loads(DESIGN_PATH.read_text(encoding="utf-8"))


def _amendment() -> dict:
    return json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))


def _metadata(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["G-ORS"] * rows,
            "layer": [1] * rows,
            "time": pd.date_range("2024-01-01", periods=rows, freq="10min", tz="Asia/Seoul").astype(
                str
            ),
        }
    )


def test_frozen_design_and_operational_amendment_hashes_and_contract(numerical) -> None:
    assert _sha(DESIGN_PATH) == DESIGN_SHA256
    assert _sha(AMENDMENT_PATH) == AMENDMENT_SHA256
    design = _design()
    amendment = _amendment()
    numerical.assert_design_contract(design)
    numerical.assert_operational_amendment(amendment)
    contract = numerical.implementation_contract()
    assert contract["fit_budget"] == {
        "inner_seed_cells_per_window": 18,
        "inner_anchor_physical_fits": 9,
        "inner_segment_physical_fits": 54,
        "outer_segment_physical_fits": 9,
        "segment_physical_fits": 63,
        "maximum_lifetime_physical_fits": 72,
        "maximum_feature_or_proposal_materializations": 21,
    }


def test_original_round_b_lineage_is_repeated_unchanged_per_inner_window() -> None:
    amendment = _amendment()
    lineage = amendment["round_b_anchor_lineage"]
    expected_seeds = [20260813, 20260829, 20260847]
    assert lineage["registered_seeds"] == expected_seeds
    assert [window["fit_seeds"] for window in amendment["inner_anchor_fits"]["windows"]] == [
        expected_seeds,
        expected_seeds,
        expected_seeds,
    ]
    assert lineage["fixed_parameters"] | {} == {
        "n_estimators": 700,
        "learning_rate": 0.035,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 60,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "objective": "binary",
        "n_jobs": 8,
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
        "feature_fraction_seed": "fit_seed",
        "bagging_seed": "fit_seed",
        "data_random_seed": "fit_seed",
        "extra_seed": "fit_seed",
        "random_state": "fit_seed",
    }
    postprocess = lineage["postprocess"]
    assert {
        key: postprocess[key]
        for key in ("high_threshold", "low_threshold", "close_gap_rows", "minimum_positive_run")
    } == {
        "high_threshold": 0.2,
        "low_threshold": 0.1,
        "close_gap_rows": 0,
        "minimum_positive_run": 12,
    }


def test_result_can_only_select_frozen_anchor_branch(numerical) -> None:
    required = tuple(f"gate_{index}" for index in range(3))
    metrics = {
        "gate_checks": {required[0]: True, required[1]: False, required[2]: True},
        "passed_all_gates": False,
        "arbitrary_numeric_result": -999999.0,
    }
    result = {
        "status": "COMPLETE_LOCAL_SCREEN_ONLY_PARENT_QA_PENDING",
        "decision": "NO_GO_LOCAL_GATE",
        "passed_all_gates": False,
    }
    assert (
        numerical.predecessor_anchor_branch(result, metrics, required_gate_names=required)
        == "FROZEN_ROUND_B"
    )
    metrics["arbitrary_numeric_result"] = 999999.0
    assert (
        numerical.predecessor_anchor_branch(result, metrics, required_gate_names=required)
        == "FROZEN_ROUND_B"
    )


def test_proposal_api_is_target_free_and_rejects_target_columns(numerical) -> None:
    signature = inspect.signature(numerical.generate_target_free_proposals)
    assert "truth" not in signature.parameters
    assert "anomaly_type" not in signature.parameters
    frame = _metadata(30).assign(
        temp=np.linspace(10.0, 12.0, 30),
        psal=np.linspace(32.0, 33.0, 30),
        depth=10.0,
        label=0,
    )
    with pytest.raises(ValueError, match="target/evaluation"):
        numerical.generate_target_free_proposals(frame, np.full(30, 0.1), (24, 72))


def test_training_target_is_single_connected_nonspike_event_at_80_percent(numerical) -> None:
    metadata = _metadata(40)
    truth = np.zeros(40, dtype=np.int8)
    truth[5:25] = 1
    proposal = numerical.SegmentRecord("p1", "G-ORS", 1, 0, 5, 25, 0.9, 0.9, "synthetic")
    assert numerical.segment_training_targets(truth, [""] * 40, metadata, [proposal]).tolist() == [
        1
    ]
    anomaly_type = [""] * 40
    anomaly_type[10] = "spike"
    assert numerical.segment_training_targets(
        truth, anomaly_type, metadata, [proposal]
    ).tolist() == [0]


def _reference_round_b_weight(metadata: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=np.int8)
    work = metadata.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["__target"] = y
    work["__time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(["station", "layer", "__time", "__position"], inplace=True)
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(600)
    prior = grouped["__target"].shift(1).fillna(0).eq(1)
    starts = work["__target"].eq(1) & (~contiguous | ~prior)
    work["__event"] = starts.cumsum().where(work["__target"].eq(1), -1).astype(np.int64)
    positive = work["__target"].eq(1)
    event_length = work.loc[positive].groupby("__event", sort=False)["__event"].transform("size")
    pos_raw = 1.0 / np.sqrt(event_length.to_numpy(dtype=float))
    pos_raw /= pos_raw.mean()
    day = work["__time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    normal = ~positive
    normal_length = (
        work.loc[normal]
        .assign(__day=day.loc[normal])
        .groupby(["station", "layer", "__day"], sort=False, observed=True)["__day"]
        .transform("size")
    )
    normal_raw = 1.0 / np.sqrt(normal_length.to_numpy(dtype=float))
    normal_raw /= normal_raw.mean()
    ordered = np.empty(len(work), dtype=np.float64)
    ordered[positive.to_numpy()] = pos_raw * math.sqrt(
        max(1, int(normal.sum())) / max(1, int(positive.sum()))
    )
    ordered[normal.to_numpy()] = normal_raw
    work["__weight"] = ordered
    return work.sort_values("__position", kind="mergesort")["__weight"].to_numpy(dtype=np.float32)


def test_round_b_event_day_weight_matches_frozen_reference(numerical) -> None:
    metadata = (
        pd.concat(
            [
                _metadata(24),
                _metadata(16).assign(station="I-ORS", layer=2),
            ],
            ignore_index=True,
        )
        .sample(frac=1.0, random_state=17)
        .reset_index(drop=True)
    )
    target = np.zeros(len(metadata), dtype=np.int8)
    target[[0, 2, 3, 5, 7, 11, 13, 17]] = 1
    np.testing.assert_array_equal(
        numerical.round_b_event_day_weight(metadata, target),
        _reference_round_b_weight(metadata, target),
    )


def test_fit_helpers_reject_unregistered_seeds_before_import_or_fit(
    numerical, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata(4)
    target = np.array([0, 1, 0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="not registered"):
        numerical.fit_round_b_anchor_model(
            np.zeros((4, 2), dtype=np.float32), target, metadata, seed=20260814
        )
    with pytest.raises(ValueError, match="not registered"):
        numerical.fit_segment_model(
            pd.DataFrame({"proposal_id": ["a", "b"], "x": [0.0, 1.0]}),
            [0, 1],
            seed=20260827,
        )


def test_segment_features_never_cross_station_layer_segment_flanks(numerical) -> None:
    first = _metadata(25).assign(temp=np.arange(25, dtype=float), psal=30.0, depth=10.0)
    second = _metadata(25).assign(station="I-ORS", temp=1000.0, psal=1000.0, depth=20.0)
    frame = pd.concat([first, second], ignore_index=True)
    proposal = numerical.SegmentRecord("p", "G-ORS", 1, 0, 5, 24, 0.8, 0.8, "synthetic")
    features = numerical.build_segment_features(
        frame,
        np.full(50, 0.1),
        np.zeros(50, dtype=np.int8),
        [proposal],
        (24, 72),
    )
    assert float(features.loc[0, "temp_24h_post_contrast"]) == -10.0
    crossing = numerical.SegmentRecord("bad", "G-ORS", 1, 0, 20, 30, 0.8, 0.8, "synthetic")
    with pytest.raises(ValueError, match="crosses"):
        numerical.build_segment_features(
            frame,
            np.full(50, 0.1),
            np.zeros(50, dtype=np.int8),
            [crossing],
            (24, 72),
        )


def test_threshold_cell_selection_and_decoder_contract(numerical) -> None:
    assert numerical.select_inner_threshold([0.80, 0.76, 0.90], [1, 0, 1]) == (
        0.85,
        1.0,
    )
    assert numerical.select_inner_threshold([0.80, 0.76, 0.70], [1, 1, 0]) == (
        0.75,
        1.0,
    )
    cells = [
        numerical.InnerCellSummary(
            "b", (24, 72), "CONNECTED_ONLY", 0.75, 0.9, (0.1, 0.1, 0.1), 40, True
        ),
        numerical.InnerCellSummary(
            "a", (48, 168), "CONNECTED_ONLY", 0.75, 0.9, (0.1, 0.1, 0.1), 40, True
        ),
    ]
    assert numerical.select_structure_cell(cells).cell_id == "a"
    anchor = np.zeros(25, dtype=np.int8)
    proposal = numerical.SegmentRecord("p", "G-ORS", 1, 0, 2, 21, 0.9, 0.9, "synthetic")
    connected, connected_audit = numerical.decode_segments(
        anchor,
        [proposal],
        [0.9],
        threshold=0.85,
        decoder_mode="CONNECTED_ONLY",
        spike_protected=np.zeros(25, dtype=bool),
        flatline_protected=np.zeros(25, dtype=bool),
    )
    assert connected.sum() == 0 and connected_audit["accepted_intervals"] == 0
    dual, dual_audit = numerical.decode_segments(
        anchor,
        [proposal],
        [0.9],
        threshold=0.85,
        decoder_mode="DUAL_BOUNDARY_DISCONNECTED_ALLOWED",
        spike_protected=np.zeros(25, dtype=bool),
        flatline_protected=np.zeros(25, dtype=bool),
    )
    assert dual.sum() == 19
    assert dual_audit["accepted_disconnected_intervals"] == 1


def test_attempt_journal_enforces_exact_72_fit_plan_and_21_materializations(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_fsync_directory", lambda _path: None)
    journal = runner.AttemptJournal.begin(tmp_path / "attempt", time.time() + 3600)
    try:
        for window in runner.INNER_WINDOW_IDS:
            for seed in runner.ROUND_B_SEEDS:
                journal.reserve_fit("INNER_ANCHOR", window, "ROUND_B_SHARED", seed)
        for window in runner.INNER_WINDOW_IDS:
            for cell in runner.STRUCTURE_CELL_IDS:
                for seed in runner.SEEDS:
                    journal.reserve_fit("INNER_SEGMENT", window, cell, seed)
        selected = runner.STRUCTURE_CELL_IDS[2]
        for fold in runner.FOLD_ORDER:
            for seed in runner.SEEDS:
                journal.reserve_fit("OUTER_SEGMENT", fold, selected, seed)
        assert journal.fit_reservations == 72
        with pytest.raises(RuntimeError, match="72-fit"):
            journal.reserve_fit("OUTER_SEGMENT", "2025_q4", selected, runner.SEEDS[0])
        for ordinal in range(21):
            journal.reserve_materialization(f"m{ordinal}")
        with pytest.raises(RuntimeError, match="ceiling"):
            journal.reserve_materialization("m21")
    finally:
        journal.close_descriptor()


def test_attempt_journal_rejects_wrong_seed_or_order_before_reservation(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_fsync_directory", lambda _path: None)
    journal = runner.AttemptJournal.begin(tmp_path / "attempt", time.time() + 3600)
    try:
        with pytest.raises(RuntimeError, match="9-fit plan"):
            journal.reserve_fit(
                "INNER_ANCHOR",
                runner.INNER_WINDOW_IDS[0],
                "ROUND_B_SHARED",
                runner.ROUND_B_SEEDS[1],
            )
        assert journal.fit_reservations == 0
    finally:
        journal.close_descriptor()


def test_initialization_fault_retains_claim_and_publishes_terminal(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    original = runner._fsync_directory

    def one_shot(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic first directory flush fault")
        original(path)

    monkeypatch.setattr(runner, "_fsync_directory", one_shot)
    artifact = tmp_path / "attempt"
    with pytest.raises(OSError, match="synthetic"):
        runner.AttemptJournal.begin(artifact, time.time() + 3600)
    assert (artifact / "execution.lock").exists()
    failure = json.loads((artifact / "initialization_failed.json").read_text())
    assert failure["execution_lock_retained"] is True
    assert failure["fit_reservations"] == 0


def test_held_truth_scorer_has_no_path_reopen(runner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("path reopened")),
    )
    score = runner.score_from_held_truth([1, 0, 1, 0], [1, 1, 0, 0])
    assert score["tp"] == 1 and score["fp"] == 1 and score["fn"] == 1


def test_execute_rejects_before_read_claim_or_numerical_import(
    runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched: list[str] = []
    monkeypatch.setattr(runner, "_read_bound_json", lambda *_a, **_k: touched.append("read"))
    monkeypatch.setattr(runner.AttemptJournal, "begin", lambda *_a, **_k: touched.append("claim"))
    with pytest.raises(runner.AuthorizationError, match="NOT_AUTHORIZED"):
        runner.execute_parent()
    assert touched == []


def test_runner_top_level_imports_are_stdlib_and_no_actual_authorization(runner) -> None:
    runner._verify_top_level_stdlib_only(RUNNER_PATH)
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert top_imports
    assert runner.AUTHORIZATION_SHA256 == "<NOT_AUTHORIZED_PENDING_INDEPENDENT_QA>"
    assert not runner.AUTHORIZATION_PATH.exists()


def test_full_static_trust_closure_and_trigger_resolution_pass(runner) -> None:
    checks = runner._static_package_checks()
    assert checks["trigger"]["anchor_branch"] == "FROZEN_ROUND_B"
    assert checks["trigger"]["successor_pooled_f1_delta"] == 0.0
    assert checks["trigger"]["successor_rescued_rows"] == 0
    assert len(checks["dependency_closure"]) == 22
    assert checks["runner_normalized_sha256"] == (
        "ff4c2da62f9e9a227731fd7f1345a23e386d34c2519b3d4e1f9e9d36f39064a8"
    )


def test_strict_data_readiness_is_complete_but_unauthorized(runner) -> None:
    if not os.environ.get("P1_DATA_DIR"):
        pytest.skip("P1_DATA_DIR is required for the held-source readiness probe")
    template, _receipt = runner._read_template()
    readiness = runner._strict_data_preflight(template)
    assert readiness["scientific_input_ready_before_claim"] is True
    assert readiness["execution_ready"] is False
    assert readiness["execution_blocker"] == "NO_INDEPENDENT_EXECUTION_AUTHORIZATION"
    inner = readiness["inner_anchor_construction"]
    assert inner["base_physical_fit_calls"] == 9
    assert inner["registered_seeds_unchanged_per_window"] == [
        20260813,
        20260829,
        20260847,
    ]
    assert inner["fits_observed"] == 0
    assert inner["materializations_observed"] == 0


def test_no_forbidden_official_path_literal_or_materialized_execution_artifacts(runner) -> None:
    lowered = RUNNER_PATH.read_text(encoding="utf-8").lower()
    assert "users\\cedis\\downloads" not in lowered
    assert "official_test_path" not in lowered
    for name in (
        "execution.lock",
        "attempt_journal",
        "result.json",
        "manifest.json",
        "999_completed.json",
    ):
        assert not (runner.ARTIFACT_DIR / name).exists()
