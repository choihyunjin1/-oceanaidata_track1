from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p1_long_event_segment_proposal_rescore_v2.py"
MODULE_PATH = ROOT / "src" / "p1_qc" / "long_event_segment_proposal_rescore_execution_v2.py"
DESIGN_PATH = (
    ROOT / "configs" / "experiments" / "p1_long_event_segment_proposal_rescore_v1_design.json"
)
AMENDMENT_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v2_operational_amendment.json"
)
CLOSURE_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v3_execution_closure_amendment.json"
)


def _load(path: Path, name: str):
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load(RUNNER_PATH, "p1_segment_rescore_v2_runner_test")


@pytest.fixture(scope="module")
def numerical():
    return _load(MODULE_PATH, "p1_qc.long_event_segment_proposal_rescore_execution_v2_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(rows: int = 1200) -> pd.DataFrame:
    times = pd.date_range("2024-06-01", periods=rows, freq="10min", tz="Asia/Seoul")
    base = np.linspace(10.0, 12.0, rows)
    values: list[pd.DataFrame] = []
    for station, offset in (("I-ORS", 0.0), ("S-ORS", 100.0)):
        for layer in (1, 2):
            values.append(
                pd.DataFrame(
                    {
                        "station": station,
                        "year": 2024,
                        "layer": layer,
                        "time": times.astype(str),
                        "temp": base + offset + layer,
                        "psal": 30.0 + 0.1 * base + offset + layer,
                        "depth": float(layer * 10),
                    }
                )
            )
    return pd.concat(values, ignore_index=True)


def test_frozen_authorities_cutoffs_shelves_and_72_fit_contract(runner) -> None:
    assert _sha(DESIGN_PATH) == runner.DESIGN_SHA256
    assert _sha(AMENDMENT_PATH) == runner.AMENDMENT_V2_SHA256
    assert _sha(CLOSURE_PATH) == runner.CLOSURE_V3_SHA256
    contract = json.loads(runner.CONFIG_PATH.read_text(encoding="utf-8"))
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    closure = json.loads(CLOSURE_PATH.read_text(encoding="utf-8"))
    receipt = runner._validate_closure(contract, design, amendment, closure)
    assert receipt["cutoffs_equal_frozen_v2"] is True
    assert receipt["fit_plan"] == {
        "anchor": 9,
        "inner_segment": 54,
        "outer_segment": 9,
    }
    windows = closure["out_of_sample_anchor_surfaces"]["windows"]
    assert [value["anchor_fit_end_inclusive"] for value in windows] == [
        "2024-05-24T23:50:00+09:00",
        "2024-08-24T23:50:00+09:00",
        "2024-11-23T23:50:00+09:00",
    ]
    assert [value["central_grid_slots_per_continuous_station_layer"] for value in windows] == [
        4320,
        4320,
        4464,
    ]
    for value in windows:
        fit_end = pd.Timestamp(value["anchor_fit_end_inclusive"])
        support_start = pd.Timestamp(value["support_surface_start"])
        shelf_start = pd.Timestamp(value["segment_calibration_start"])
        shelf_end = pd.Timestamp(value["segment_calibration_end_exclusive"])
        validation_start = pd.Timestamp(value["inner_validation_start"])
        assert support_start == fit_end + pd.Timedelta(minutes=10)
        assert shelf_start == fit_end + pd.Timedelta(days=7, minutes=10)
        assert shelf_end == validation_start - pd.Timedelta(days=7)


def test_full_oos_support_surface_uses_estimator_without_sentinel_fill(
    numerical, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_end = pd.Timestamp("2024-01-01T23:50:00+09:00")
    support_start = fit_end + pd.Timedelta(minutes=10)
    validation_start = pd.Timestamp("2024-01-20T00:00:00+09:00")
    central_start = fit_end + pd.Timedelta(days=7, minutes=10)
    central_end = validation_start - pd.Timedelta(days=7)
    fit_time = pd.date_range(end=fit_end, periods=12, freq="10min")
    support_time = pd.date_range(
        start=support_start,
        end=validation_start - pd.Timedelta(minutes=10),
        freq="10min",
    )
    validation_time = pd.date_range(start=validation_start, periods=12, freq="10min")
    times = fit_time.append(support_time).append(validation_time)
    positions = np.arange(len(times), dtype=np.int64)
    train = pd.DataFrame(
        {
            "station": "I-ORS",
            "year": times.year,
            "layer": 1,
            "time": times.astype(str),
            "temp": 10.0 + np.sin(positions / 30.0),
            "psal": 30.0 + np.cos(positions / 40.0),
            "depth": 10.0,
            "label": (positions % 11 == 0).astype(np.int8),
            "anomaly_type": "none",
        }
    )
    transformed: list[np.ndarray] = []

    class Encoder:
        def fit(self, _bundle, fit_positions):
            self.fit_positions = np.asarray(fit_positions, dtype=np.int64)
            return self

        def transform(self, _bundle, current_positions):
            current = np.asarray(current_positions, dtype=np.int64)
            transformed.append(current.copy())
            return current.astype(np.float64).reshape(-1, 1)

    class Anchor:
        def __init__(self, seed: int) -> None:
            self.offset = numerical.ROUND_B_SEEDS.index(seed) * 0.01

        def predict_proba(self, matrix):
            current = np.asarray(matrix[:, 0], dtype=np.int64)
            probability = 0.15 + (current % 17) * 0.02 + self.offset
            return np.column_stack((1.0 - probability, probability))

    postprocess_lengths: list[int] = []
    fake_numerical = SimpleNamespace(
        TabularEncoder=Encoder,
        detect_plateaus=lambda frame: pd.Series(np.zeros(len(frame), dtype=bool)),
        detect_singleton_spikes=lambda frame: pd.Series(np.zeros(len(frame), dtype=bool)),
        apply_postprocess=lambda frame, probability, _plateau, _spike, _config: (
            postprocess_lengths.append(len(frame))
            or (np.asarray(probability) >= 0.25).astype(np.int8)
        ),
    )
    monkeypatch.setattr(
        numerical.frozen,
        "fit_round_b_anchor_model",
        lambda _features, _target, _metadata, seed: Anchor(seed),
    )

    class Journal:
        def __init__(self) -> None:
            self.fit_count = 0
            self.completed: list[int] = []

        def reserve_materialization(self, label: str) -> int:
            assert label == "inner_anchor_surface:synthetic"
            return 1

        def reserve_fit(self, phase: str, window: str, cell: str, seed: int) -> int:
            assert (phase, window, cell) == (
                "INNER_ANCHOR",
                "synthetic",
                "ROUND_B_SHARED",
            )
            assert seed in numerical.ROUND_B_SEEDS
            self.fit_count += 1
            return self.fit_count

        def complete_fit(self, ordinal: int) -> None:
            self.completed.append(ordinal)

    closure = {
        "out_of_sample_anchor_surfaces": {
            "uniform_postprocess": {},
            "windows": [
                {
                    "id": "synthetic",
                    "anchor_fit_end_inclusive": fit_end.isoformat(),
                    "support_surface_start": support_start.isoformat(),
                    "support_surface_end_exclusive": validation_start.isoformat(),
                    "segment_calibration_start": central_start.isoformat(),
                    "segment_calibration_end_exclusive": central_end.isoformat(),
                    "inner_validation_start": validation_start.isoformat(),
                    "inner_validation_end_inclusive": validation_time[-1].isoformat(),
                }
            ],
        }
    }
    journal = Journal()
    result = numerical._fit_anchor_surfaces(
        {"train": train, "bundle": object()},
        fake_numerical,
        closure,
        journal,
        time.time() + 3600,
    )
    support = result["synthetic"]["calibration"]
    expected_support_positions = np.flatnonzero(
        (times >= support_start) & (times < validation_start)
    )
    assert journal.fit_count == 3 and journal.completed == [1, 2, 3]
    assert any(np.array_equal(value, expected_support_positions) for value in transformed)
    np.testing.assert_array_equal(
        support.frame[["station", "layer", "time"]].to_numpy(),
        train.iloc[expected_support_positions][["station", "layer", "time"]].to_numpy(),
    )
    expected_probability = (
        0.15 + (expected_support_positions % 17) * 0.02 + np.mean([0.0, 0.01, 0.02])
    )
    np.testing.assert_allclose(support.anchor_probability, expected_probability)
    assert np.all(support.anchor_probability != 1.0)
    assert pd.to_datetime(support.frame["time"], utc=True).min() > fit_end.tz_convert("UTC")
    assert len(support.anchor_probability) in postprocess_lengths

    # A proposal wholly in the noncentral lower buffer proves these estimator
    # values are numerical feature inputs, rather than inert coverage sentinels.
    proposal_start, proposal_stop = 500, 520
    segment_id = int(numerical.frozen.exact_gap_safe_segment_ids(support.frame)[proposal_start])
    proposal = numerical.frozen.SegmentRecord(
        "noncentral",
        "I-ORS",
        1,
        segment_id,
        proposal_start,
        proposal_stop,
        0.9,
        0.9,
        "synthetic",
    )
    before = numerical.build_bounded_segment_features(
        support.frame,
        support.anchor_probability,
        support.anchor_prediction,
        [proposal],
        (24, 72),
    )
    altered = support.anchor_probability.copy()
    altered[proposal_start:proposal_stop] = np.clip(
        altered[proposal_start:proposal_stop] + 0.2,
        0.0,
        1.0,
    )
    after = numerical.build_bounded_segment_features(
        support.frame,
        altered,
        support.anchor_prediction,
        [proposal],
        (24, 72),
    )
    assert before["anchor_probability_mean"].iloc[0] != after["anchor_probability_mean"].iloc[0]


def test_runner_has_stdlib_only_top_level_and_is_not_authorized(runner) -> None:
    runner._verify_top_level_stdlib_only(RUNNER_PATH)
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert imports
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "AUTHORIZATION_SHA256 =" not in source
    assert runner._normalised_runner_sha256() == _sha(RUNNER_PATH)
    assert not runner.AUTHORIZATION_PATH.exists()


def test_authorization_template_matches_exact_runner_schema(runner) -> None:
    template = json.loads(runner.AUTHORIZATION_TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert set(template) == set(runner.AUTHORIZATION_REQUIRED_KEYS)
    assert template["authorized"] is False
    assert template["status"] == "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
    assert set(template["operation_authorization"]) == {
        "single_attempt",
        "maximum_lifetime_physical_fits",
        "maximum_scientific_materializations",
        "outer_scores",
        "candidate_files",
        "uploads",
    }
    assert set(template["independent_qa"]) == {"path", "bytes", "sha256", "verdict"}


def test_execute_rejects_before_any_repository_read_or_claim(
    runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched: list[str] = []
    monkeypatch.delenv(runner.AUTHORIZATION_ENV_VAR, raising=False)
    monkeypatch.setattr(runner, "_read_bound_bytes", lambda *_a, **_k: touched.append("read"))
    monkeypatch.setattr(runner.AttemptJournal, "begin", lambda *_a, **_k: touched.append("claim"))
    with pytest.raises(runner.AuthorizationError, match="external authorization digest"):
        runner.execute_parent()
    assert touched == []


def test_seal_rejects_external_authorization_capability_before_reads(
    runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched: list[str] = []
    monkeypatch.setenv(runner.AUTHORIZATION_ENV_VAR, "a" * 64)
    monkeypatch.setattr(runner, "_static_package_checks", lambda *_a, **_k: touched.append("read"))
    with pytest.raises(runner.AuthorizationError, match="must be absent while sealing"):
        runner.seal()
    assert touched == []


def test_readiness_digest_is_stable_across_live_authorization_transition(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = tmp_path / "authorization.json"
    monkeypatch.setattr(runner, "AUTHORIZATION_PATH", authorization)
    monkeypatch.delenv(runner.AUTHORIZATION_ENV_VAR, raising=False)
    immutable = {
        "schema_version": "synthetic",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "PASS",
        "readiness": {"rows": 1},
        "operation_counters": {"claims": 0, "physical_fits": 0},
    }
    before = runner._attach_stable_verification_and_live_authorization(immutable)
    synthetic = {
        "schema_version": "synthetic-valid-authorization",
        "authorized": True,
        "status": "AUTHORIZED_INDEPENDENT_QA_PASS_ONE_SHOT",
    }
    raw = (json.dumps(synthetic, sort_keys=True) + "\n").encode()
    authorization.write_bytes(raw)
    monkeypatch.setenv(runner.AUTHORIZATION_ENV_VAR, hashlib.sha256(raw).hexdigest())
    after = runner._attach_stable_verification_and_live_authorization(immutable)
    assert before["verification_sha256"] == after["verification_sha256"]
    assert before["live_authorization_inspection"] != after["live_authorization_inspection"]
    assert after["live_authorization_inspection"] == {
        "excluded_from_verification_sha256": True,
        "actual_authorization_exists": True,
        "external_digest_is_64_lower_hex": True,
    }


def test_held_authority_path_a_to_b_swap_fails_without_reparse(runner, tmp_path: Path) -> None:
    path = tmp_path / "seal.json"
    path.write_bytes(b'{"seal":"A"}\n')
    raw, receipt = runner._read_bound_bytes(path)
    held = {"raw": raw, "receipt": receipt}
    path.unlink()
    path.write_bytes(b'{"seal":"B-substituted"}\n')
    with pytest.raises(RuntimeError, match="path identity changed"):
        runner._verify_held_path_identity(path, held)
    assert json.loads(raw) == {"seal": "A"}


def test_external_digest_authorization_chain_is_acyclic_without_runner_mutation(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal_path = tmp_path / "seal.json"
    qa_path = tmp_path / "qa.json"
    auth_path = tmp_path / "authorization.json"
    monkeypatch.setattr(runner, "SEAL_PATH", seal_path)
    monkeypatch.setattr(runner, "R5_QA_PATH", qa_path)
    monkeypatch.setattr(runner, "AUTHORIZATION_PATH", auth_path)
    monkeypatch.setattr(runner, "_relative", lambda path: str(path))
    runner_sha = _sha(RUNNER_PATH)
    preflight_sha = "a" * 64
    seal = {
        "status": "SEALED_STRICT_ZERO_FIT_NOT_AUTHORIZED_PENDING_INDEPENDENT_QA",
        "contract_sha256": runner.CONFIG_SHA256,
        "trust_firewall_v5_sha256": runner.TRUST_FIREWALL_V5_SHA256,
        "runner_sha256": runner_sha,
        "runner_normalized_sha256": runner._normalised_runner_sha256(),
        "execution_module_sha256": runner.EXECUTION_MODULE_SHA256,
        "test_sha256": _sha(Path(__file__)),
        "authorization_template_sha256": _sha(runner.AUTHORIZATION_TEMPLATE_PATH),
        "project_file_sha256": runner.PROJECT_FILE_SHA256,
        "readonly_preflight_verification_sha256": preflight_sha,
    }
    seal_raw = (json.dumps(seal, sort_keys=True) + "\n").encode()
    seal_path.write_bytes(seal_raw)
    seal_sha = hashlib.sha256(seal_raw).hexdigest()
    qa = {
        "verdict": "PASS",
        "experiment_id": runner.EXPERIMENT_ID,
        "preexecution_seal_sha256": seal_sha,
        "contract_sha256": runner.CONFIG_SHA256,
        "trust_firewall_v5_sha256": runner.TRUST_FIREWALL_V5_SHA256,
        "runner_sha256": runner_sha,
        "execution_module_sha256": runner.EXECUTION_MODULE_SHA256,
    }
    qa_raw = (json.dumps(qa, sort_keys=True) + "\n").encode()
    qa_path.write_bytes(qa_raw)
    authorization = {
        "schema_version": "synthetic-r5",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "AUTHORIZED_INDEPENDENT_QA_PASS_ONE_SHOT",
        "authorized": True,
        "contract_sha256": runner.CONFIG_SHA256,
        "trust_firewall_v5_sha256": runner.TRUST_FIREWALL_V5_SHA256,
        "runner_sha256": runner_sha,
        "runner_normalized_sha256": runner._normalised_runner_sha256(),
        "preexecution_seal": {
            "path": str(seal_path),
            "bytes": len(seal_raw),
            "sha256": seal_sha,
        },
        "independent_qa": {
            "path": str(qa_path),
            "bytes": len(qa_raw),
            "sha256": hashlib.sha256(qa_raw).hexdigest(),
            "verdict": "PASS",
        },
        "readonly_preflight_verification_sha256": preflight_sha,
        "zero_prior_state": {
            "claims": 0,
            "physical_fits": 0,
            "scientific_materializations": 0,
            "outer_scores": 0,
            "candidate_files": 0,
        },
        "operation_authorization": {
            "single_attempt": True,
            "maximum_lifetime_physical_fits": 72,
            "maximum_scientific_materializations": 21,
            "outer_scores": 1,
            "candidate_files": 0,
            "uploads": 0,
        },
    }
    auth_raw = (json.dumps(authorization, sort_keys=True) + "\n").encode()
    auth_path.write_bytes(auth_raw)
    auth_sha = hashlib.sha256(auth_raw).hexdigest()
    monkeypatch.setenv(runner.AUTHORIZATION_ENV_VAR, auth_sha)
    before_runner = _sha(RUNNER_PATH)
    receipt = runner._require_execution_authorization_preimport()
    assert receipt["external_authorization_sha256"] == auth_sha
    assert receipt["seal_read_receipt"]["actual_read_sha256"] == seal_sha
    assert set(receipt["held_authorities"]) == {
        str(auth_path),
        str(seal_path),
        str(qa_path),
        str(runner.DESIGN_PATH),
        str(runner.EXECUTION_MODULE_PATH),
        str(runner.TRUST_FIREWALL_V5_PATH),
    }
    assert _sha(RUNNER_PATH) == before_runner


def test_exact_snapshot_inventory_rejects_self_declared_legacy_substitution_preimport(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    sources = {
        runner._relative_literal(runner.LEGACY_RUNNER_PATH): runner.LEGACY_RUNNER_PATH,
        runner._relative_literal(runner.EXECUTION_MODULE_PATH): runner.EXECUTION_MODULE_PATH,
        runner._relative_literal(runner.DESIGN_PATH): runner.DESIGN_PATH,
        runner._relative_literal(runner.TRUST_FIREWALL_V5_PATH): runner.TRUST_FIREWALL_V5_PATH,
    }
    expected = {}
    for relative, source in sources.items():
        raw = source.read_bytes()
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        expected[relative] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    entries = json.loads(json.dumps(expected))
    legacy_relative = runner._relative_literal(runner.LEGACY_RUNNER_PATH)
    substituted = b"raise RuntimeError('crafted legacy runner')\n"
    (root / legacy_relative).write_bytes(substituted)
    entries[legacy_relative] = {
        "bytes": len(substituted),
        "sha256": hashlib.sha256(substituted).hexdigest(),
    }
    imported: list[str] = []
    monkeypatch.setattr(
        runner,
        "_load_module_from_path",
        lambda *_a, **_k: imported.append("imported"),
    )
    with pytest.raises(RuntimeError, match="declared inventory"):
        runner._verify_exact_snapshot_inventory(root, entries, expected)
    assert imported == []


def test_exact_snapshot_inventory_rejects_undeclared_bytecode_and_extra_tree(
    runner, tmp_path: Path
) -> None:
    root = tmp_path / "snapshot"
    source = root / "sealed.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")
    expected = {
        "sealed.py": {
            "bytes": source.stat().st_size,
            "sha256": _sha(source),
        }
    }
    bytecode = root / "__pycache__" / "sealed.cpython-311.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"synthetic timestamp-valid-bytecode probe")
    with pytest.raises(RuntimeError, match="bytecode directory"):
        runner._verify_exact_snapshot_inventory(root, expected, expected)
    bytecode.unlink()
    bytecode.parent.rmdir()
    undeclared_directory = root / "undeclared_directory"
    undeclared_directory.mkdir()
    with pytest.raises(RuntimeError, match="undeclared directory"):
        runner._verify_exact_snapshot_inventory(root, expected, expected)
    undeclared_directory.rmdir()
    extra = root / "undeclared.py"
    extra.write_bytes(b"raise RuntimeError('must never import')\n")
    with pytest.raises(RuntimeError, match="actual snapshot tree differs"):
        runner._verify_exact_snapshot_inventory(root, expected, expected)


def test_attempt_journal_exact_72_fit_and_21_materialization_plan(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_fsync_directory", lambda _path: None)
    journal = runner.AttemptJournal.begin(tmp_path / "attempt", time.time() + 3600)
    try:
        for window in runner.INNER_WINDOW_IDS:
            journal.reserve_materialization(f"inner_anchor_surface:{window}")
            for seed in runner.ROUND_B_SEEDS:
                ordinal = journal.reserve_fit("INNER_ANCHOR", window, "ROUND_B_SHARED", seed)
                journal.complete_fit(ordinal)
        for window in runner.INNER_WINDOW_IDS:
            for bank in runner.CONTEXT_BANK_IDS:
                journal.reserve_materialization(f"inner_context_surface:{window}:{bank}")
                for decoder in runner.DECODER_IDS:
                    cell = f"bank_{bank}__{decoder}"
                    for seed in runner.SEGMENT_SEEDS:
                        ordinal = journal.reserve_fit("INNER_SEGMENT", window, cell, seed)
                        journal.complete_fit(ordinal)
        for fold in runner.FOLD_ORDER:
            for bank in runner.CONTEXT_BANK_IDS:
                journal.reserve_materialization(f"outer_context_surface:{fold}:{bank}")
        selected = runner.STRUCTURE_CELL_IDS[2]
        for fold in runner.FOLD_ORDER:
            for seed in runner.SEGMENT_SEEDS:
                ordinal = journal.reserve_fit("OUTER_SEGMENT", fold, selected, seed)
                journal.complete_fit(ordinal)
        assert (journal.fit_reservations, journal.fits_completed, journal.materializations) == (
            72,
            72,
            21,
        )
        journal.record_outer_freeze({"candidate_sha256": "a" * 64})
        with pytest.raises(RuntimeError, match="72-fit"):
            journal.reserve_fit("OUTER_SEGMENT", "2025_q4", selected, runner.SEGMENT_SEEDS[0])
        with pytest.raises(RuntimeError, match="21-materialization"):
            journal.reserve_materialization("outer_context_surface:2025_q4:24_72")
    finally:
        journal.close_handle_keep_lock()


def test_wrong_fit_order_and_wrong_materialization_reject_before_count(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_fsync_directory", lambda _path: None)
    journal = runner.AttemptJournal.begin(tmp_path / "attempt", time.time() + 3600)
    try:
        with pytest.raises(RuntimeError, match="materialization differs"):
            journal.reserve_materialization("inner_anchor_surface:inner_2024_oct_nov")
        with pytest.raises(RuntimeError, match="fit reservation differs"):
            journal.reserve_fit(
                "INNER_ANCHOR",
                runner.INNER_WINDOW_IDS[0],
                "ROUND_B_SHARED",
                runner.ROUND_B_SEEDS[1],
            )
        assert journal.materializations == 0 and journal.fit_reservations == 0
    finally:
        journal.close_handle_keep_lock()


def test_peer_axis_is_same_station_other_layer_only_and_g_is_neutral(numerical) -> None:
    frame = _frame(30)
    original, count = numerical._same_station_cross_layer_residual(frame, "temp")
    other_station = frame.copy()
    other_station.loc[other_station["station"].eq("S-ORS"), "temp"] += 9999.0
    changed_other, _ = numerical._same_station_cross_layer_residual(other_station, "temp")
    mask_i = frame["station"].eq("I-ORS").to_numpy()
    np.testing.assert_allclose(original[mask_i], changed_other[mask_i], equal_nan=True)
    same_station = frame.copy()
    mask_i2 = same_station["station"].eq("I-ORS") & same_station["layer"].eq(2)
    same_station.loc[mask_i2, "temp"] += 7.0
    changed_peer, _ = numerical._same_station_cross_layer_residual(same_station, "temp")
    mask_i1 = (frame["station"].eq("I-ORS") & frame["layer"].eq(1)).to_numpy()
    assert not np.allclose(original[mask_i1], changed_peer[mask_i1])
    g = frame.loc[frame["station"].eq("I-ORS")].copy()
    g["station"] = "G-ORS"
    residual_g, count_g = numerical._same_station_cross_layer_residual(g, "temp")
    assert np.isnan(residual_g).all() and not count_g.any() and count[mask_i].all()


def test_segment_features_are_invariant_to_raw_outside_max_bank_support(numerical) -> None:
    frame = _frame(1200)
    segment_id = int(numerical.frozen.exact_gap_safe_segment_ids(frame)[500])
    proposal = numerical.frozen.SegmentRecord(
        "p",
        "I-ORS",
        1,
        segment_id,
        500,
        520,
        0.9,
        0.9,
        "synthetic",
    )
    probability = np.full(len(frame), 0.1)
    prediction = np.zeros(len(frame), dtype=np.int8)
    before = numerical.build_bounded_segment_features(
        frame, probability, prediction, [proposal], (24, 72)
    )
    perturbed = frame.copy()
    perturbed.loc[1100, ["temp", "psal"]] = [1e9, -1e9]
    after = numerical.build_bounded_segment_features(
        perturbed, probability, prediction, [proposal], (24, 72)
    )
    pd.testing.assert_frame_equal(before, after)


def test_gap_crossing_proposal_is_rejected(numerical) -> None:
    frame = _frame(80).iloc[:80].copy()
    parsed = pd.to_datetime(frame["time"], utc=True)
    frame.loc[40:, "time"] = (parsed.iloc[40:] + pd.Timedelta(hours=2)).astype(str)
    segment_ids = numerical.frozen.exact_gap_safe_segment_ids(frame)
    proposal = numerical.frozen.SegmentRecord(
        "cross",
        "I-ORS",
        1,
        int(segment_ids[30]),
        30,
        50,
        0.9,
        0.9,
        "synthetic",
    )
    with pytest.raises(ValueError, match="crosses"):
        numerical.build_bounded_segment_features(
            frame,
            np.full(len(frame), 0.1),
            np.zeros(len(frame), dtype=np.int8),
            [proposal],
            (24, 72),
        )


def test_target_free_api_and_old_unbounded_or_cross_station_paths_are_unreachable(
    numerical,
) -> None:
    signature = inspect.signature(numerical.generate_bounded_target_free_proposals)
    assert "truth" not in signature.parameters and "anomaly_type" not in signature.parameters
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "other-station same-layer" not in source
    assert "whole-segment" not in source
    frame = _frame(20).assign(label=0)
    with pytest.raises(ValueError, match="target/evaluation"):
        numerical.generate_bounded_target_free_proposals(
            frame,
            np.full(len(frame), 0.1),
            (24, 72),
        )


def test_outer_truth_is_first_accessed_only_after_prediction_freeze(
    numerical, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert 'state["truth"]' not in inspect.getsource(numerical._fit_outer)
    assert "_accepted_interval_audit" not in inspect.getsource(numerical._fit_outer)
    outer_source = inspect.getsource(numerical._outer_surface)
    assert '.loc[:, ["temp", "psal", "depth"]]' in outer_source
    assert 'state["train"].iloc' not in outer_source
    marker = {"frozen": False, "truth_reads": 0}

    class GuardedState(dict):
        def __getitem__(self, key):
            if key in {
                "frozen_truth_oof_bytes",
                "frozen_truth_oof_sha256",
                "frozen_truth_oof_rows",
            }:
                assert marker["frozen"] is True
                marker["truth_reads"] += 1
            return super().__getitem__(key)

    class Journal:
        fit_reservations = 72
        fits_completed = 72
        materializations = 21

        def record_outer_freeze(self, _freeze) -> None:
            marker["frozen"] = True

    selected = SimpleNamespace(
        cell_id="bank_24_72__connected_only",
        context_bank_hours=(24, 72),
        decoder_mode="CONNECTED_ONLY",
        threshold=0.75,
    )
    monkeypatch.setattr(numerical, "_fit_anchor_surfaces", lambda *_a, **_k: {})
    monkeypatch.setattr(numerical, "_fit_inner_cells", lambda *_a, **_k: ({}, {}))
    monkeypatch.setattr(numerical, "_select_inner_cell", lambda *_a, **_k: (selected, {}))

    def freeze_outer(_state, _num, _surfaces, _contexts, _selected, journal, _deadline):
        journal.record_outer_freeze({"candidate_sha256": "a" * 64})
        return {"freeze": {"candidate_sha256": "a" * 64}}

    monkeypatch.setattr(numerical, "_fit_outer", freeze_outer)
    monkeypatch.setattr(
        numerical,
        "_parse_held_outer_truth_after_freeze",
        lambda state: (
            state["frozen_truth_oof_bytes"],
            pd.DataFrame({"label": [0]}),
        )[1],
    )
    monkeypatch.setattr(
        numerical,
        "_score_outer",
        lambda _outer, truth: (
            {"truth_rows": len(truth)},
            {
                "decision": "NO_GO",
                "RESEARCH_GO": False,
                "SUBMISSION_GO_RESEARCH_ONLY": False,
            },
        ),
    )
    monkeypatch.setattr(
        numerical,
        "_verify_round_b_equivalence_after_freeze",
        lambda *_a, **_k: {"status": "PASS_SYNTHETIC"},
    )
    result = numerical.run_authorized_screen(
        GuardedState(
            frozen_truth_oof_bytes=b"held-once",
            expected_base_metrics={},
        ),
        object(),
        {},
        Journal(),
        time.time() + 3600,
    )
    assert result["decision"] == "NO_GO"
    assert marker == {"frozen": True, "truth_reads": 1}


def test_empty_or_single_class_shelf_fails_before_any_segment_fit(
    numerical, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_surface = SimpleNamespace()
    surfaces = {
        window: {"calibration": fake_surface, "validation": fake_surface}
        for window in numerical.frozen.INNER_WINDOW_IDS
    }
    calls = {"fits": 0, "materializations": 0}

    class Journal:
        def reserve_materialization(self, _label: str) -> None:
            calls["materializations"] += 1

        def reserve_fit(self, *_args) -> int:
            calls["fits"] += 1
            return calls["fits"]

    monkeypatch.setattr(
        numerical,
        "_build_context",
        lambda surface, bank, include_targets: SimpleNamespace(
            surface=surface,
            bank=bank,
            proposals=(),
            features=pd.DataFrame(),
            targets=np.zeros(3, dtype=np.int8) if include_targets else None,
        ),
    )
    with pytest.raises(RuntimeError, match="single-class"):
        numerical._fit_inner_cells(surfaces, Journal(), time.time() + 3600)
    assert calls["fits"] == 0


def test_final_commit_has_no_filesystem_operation_after_lock_unlink(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_fsync_directory", lambda _path: None)
    journal = runner.AttemptJournal.begin(tmp_path / "attempt", time.time() + 3600)
    journal.fit_reservations = 72
    journal.fits_completed = 72
    journal.materializations = 21
    journal.terminal_entry(
        "0999_completed.json",
        {"status": "SUCCESS_ALL_OUTPUTS_VERIFIED_READY_FOR_LOCK_RELEASE"},
    )
    marker = {"unlinked": False}
    original_unlink = Path.unlink

    def guarded_unlink(path: Path, *args, **kwargs):
        if path == journal.lock_path:
            marker["unlinked"] = True
        return original_unlink(path, *args, **kwargs)

    def no_postunlink_fsync(_path: Path) -> None:
        assert marker["unlinked"] is False

    monkeypatch.setattr(Path, "unlink", guarded_unlink)
    monkeypatch.setattr(runner, "_fsync_directory", no_postunlink_fsync)
    runner._final_success_commit(journal)
    assert marker["unlinked"] is True


def test_parent_crash_terminal_is_hash_chained_exact_once_and_retains_lock(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "attempt"
    monkeypatch.setattr(runner, "ARTIFACT_DIR", artifact)
    monkeypatch.setattr(runner, "_fsync_directory", lambda _path: None)
    journal = runner.AttemptJournal.begin(artifact, time.time() + 3600)
    try:
        first = runner._record_parent_failure_if_claimed(
            "PARENT_WORKER_SUPERVISION",
            RuntimeError("worker vanished"),
            provenance={"tree_termination_verified": True},
        )
        second = runner._record_parent_failure_if_claimed(
            "PARENT_WORKER_SUPERVISION",
            RuntimeError("duplicate observer"),
            provenance={"tree_termination_verified": True},
        )
        assert first == second == artifact / "attempt_journal" / "0997_failed.json"
        _last, entries = runner._verify_journal_chain(
            artifact / "attempt_journal", required_last="0997_failed.json"
        )
        assert (
            sum(
                value.get("schema_version") == "p1_segment_rescore.parent_failed_terminal.v2"
                for value in entries
            )
            == 1
        )
        assert entries[-1]["execution_lock_retained"] is True
        assert (artifact / "execution.lock").is_file()
    finally:
        journal.close_handle_keep_lock()


def test_parent_closes_empty_claimed_journal_crash_exact_once(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "attempt"
    artifact.mkdir()
    monkeypatch.setattr(runner, "ARTIFACT_DIR", artifact)
    monkeypatch.setattr(runner, "_fsync_directory", lambda _path: None)
    runner._atomic_create_json(
        artifact / "execution.lock",
        {
            "schema_version": "p1_segment_rescore.execution_lock.v2",
            "experiment_id": runner.EXPERIMENT_ID,
            "attempt_id": "empty-journal-attempt",
        },
    )
    (artifact / "attempt_journal").mkdir()
    first = runner._record_parent_failure_if_claimed(
        "PARENT_WORKER_SUPERVISION", RuntimeError("worker died before started entry")
    )
    second = runner._record_parent_failure_if_claimed(
        "PARENT_WORKER_SUPERVISION", RuntimeError("duplicate observer")
    )
    assert first == second == artifact / "attempt_journal" / "0997_failed.json"
    _last, entries = runner._verify_journal_chain(
        artifact / "attempt_journal", required_last="0997_failed.json"
    )
    assert len(entries) == 1
    assert entries[0]["attempt_id"] == "empty-journal-attempt"
    assert entries[0]["previous_entry_sha256"] is None
    assert (artifact / "execution.lock").is_file()


def test_parent_records_postcompletion_unlink_failure_without_ambiguous_success(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "attempt"
    monkeypatch.setattr(runner, "ARTIFACT_DIR", artifact)
    monkeypatch.setattr(runner, "_fsync_directory", lambda _path: None)
    journal = runner.AttemptJournal.begin(artifact, time.time() + 3600)
    try:
        journal.terminal_entry(
            "0998_worker_terminal.json",
            {"status": "WORKER_SUCCESS_COMMIT_PREPARED"},
        )
        journal.terminal_entry(
            "0999_completed.json",
            {"status": "SUCCESS_ALL_OUTPUTS_VERIFIED_READY_FOR_LOCK_RELEASE"},
        )
        terminal = runner._record_parent_failure_if_claimed(
            "PARENT_WORKER_SUPERVISION",
            OSError("lock unlink failed"),
        )
        assert terminal == artifact / "attempt_journal" / "1000_postcompletion_failed.json"
        _last, entries = runner._verify_journal_chain(
            artifact / "attempt_journal",
            required_last="1000_postcompletion_failed.json",
        )
        assert entries[-1]["status"] == "FAILED_AFTER_COMPLETION_BEFORE_LOCK_RELEASE"
        assert (artifact / "execution.lock").is_file()
    finally:
        journal.close_handle_keep_lock()


def test_parent_recovers_lock_free_durable_success_after_stdout_loss_without_writes(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "attempt"
    journal = artifact / "attempt_journal"
    journal.mkdir(parents=True)
    (artifact / "result.json").write_text("{}\n", encoding="utf-8")
    (artifact / "manifest.json").write_text("{}\n", encoding="utf-8")
    (journal / "0999_completed.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ARTIFACT_DIR", artifact)
    monkeypatch.setattr(
        runner,
        "_parent_read_only_verify",
        lambda path: {
            "status": "PASS_PARENT_READ_ONLY_INDEPENDENT_MANIFEST_QA",
            "result_path": str(path),
        },
    )
    before = {
        str(path.relative_to(artifact)): (path.stat().st_size, _sha(path))
        for path in artifact.rglob("*")
        if path.is_file()
    }
    recovered = runner._recover_committed_success_after_supervision_failure(
        RuntimeError("worker exited after unlink before stdout")
    )
    after = {
        str(path.relative_to(artifact)): (path.stat().st_size, _sha(path))
        for path in artifact.rglob("*")
        if path.is_file()
    }
    assert recovered is not None
    assert recovered[1]["status"] == "PASS_RECOVERED_DURABLE_SUCCESS_AFTER_STDOUT_LOSS"
    assert recovered[1]["recovery_writes"] == 0
    assert before == after


def test_started_entry_publication_fault_gets_valid_initialization_terminal(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "attempt"
    monkeypatch.setattr(runner, "_fsync_directory", lambda _path: None)
    original = runner._atomic_create_json
    injected = {"done": False}

    def publish_then_raise(path: Path, value):
        result = original(path, value)
        if path.name == "0001_started.json" and not injected["done"]:
            injected["done"] = True
            raise OSError("one-shot post-publication fault")
        return result

    monkeypatch.setattr(runner, "_atomic_create_json", publish_then_raise)
    with pytest.raises(OSError, match="post-publication fault"):
        runner.AttemptJournal.begin(artifact, time.time() + 3600)
    assert injected["done"] is True
    assert (artifact / "execution.lock").is_file()
    _last, entries = runner._verify_journal_chain(
        artifact / "attempt_journal", required_last="0997_failed.json"
    )
    assert [entry["schema_version"] for entry in entries] == [
        "p1_segment_rescore.attempt_started.v2",
        "p1_segment_rescore.initialization_failed.v2",
    ]
    assert entries[-1]["execution_lock_retained"] is True


def test_no_forbidden_output_or_official_path_literal(runner) -> None:
    lowered = RUNNER_PATH.read_text(encoding="utf-8").lower()
    assert "users\\cedis\\downloads" not in lowered
    assert "official_test_path" not in lowered
    if runner.ARTIFACT_DIR.exists():
        forbidden = {
            "execution.lock",
            "attempt_journal",
            "metrics.json",
            "report_ko.md",
            "result.json",
            "manifest.json",
        }
        assert not forbidden.intersection(path.name for path in runner.ARTIFACT_DIR.iterdir())


@pytest.mark.skipif(not os.environ.get("P1_DATA_DIR"), reason="P1_DATA_DIR not configured")
def test_read_only_preflight_is_deterministic_and_zero_operation(runner) -> None:
    command = [
        str(ROOT / ".venv-p1" / "Scripts" / "python.exe"),
        str(RUNNER_PATH),
        "--preflight",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    first_run = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    second_run = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    first = json.loads(first_run.stdout)["output"]
    second = json.loads(second_run.stdout)["output"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["operation_counters"] == {
        "claims": 0,
        "physical_fits": 0,
        "scientific_materializations": 0,
        "outer_scores": 0,
        "candidate_files": 0,
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
        "uploads": 0,
    }
    equivalence = first["readiness"]["exact_round_b_equivalence"]
    assert equivalence["status"] == (
        "PASS_TARGET_FREE_KEYS_EXACT_TRUTH_METRICS_DEFERRED_POST_FREEZE"
    )
    assert equivalence["pre_freeze_target_columns_parsed"] == 0
    assert equivalence["pre_freeze_parsed_columns"] == [
        "station",
        "year",
        "layer",
        "time",
        "fold",
    ]
