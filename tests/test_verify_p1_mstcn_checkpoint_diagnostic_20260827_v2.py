from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "verify_p1_mstcn_checkpoint_diagnostic_20260827_v2.py"


def _load_verifier():
    name = "p1_checkpoint_diagnostic_v2_verifier_tested"
    spec = importlib.util.spec_from_file_location(name, VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_absent_or_unsealed_namespace_returns_wait_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    verifier = _load_verifier()
    absent = tmp_path / "absent"
    assert verifier.main(["--artifact-dir", str(absent), "--project-root", str(tmp_path)]) == 3
    assert json.loads(capsys.readouterr().out)["result"] == "WAIT_INCOMPLETE"
    assert not absent.exists()
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    before = set(incomplete.iterdir())
    assert verifier.main(["--artifact-dir", str(incomplete), "--project-root", str(tmp_path)]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["result"] == "WAIT_INCOMPLETE"
    assert result["writes_performed"] == 0
    assert set(incomplete.iterdir()) == before


def _build_manifest_fixture(verifier, artifact_dir: Path) -> None:
    artifact_dir.mkdir()
    for name in sorted(verifier._expected_artifact_names()):
        path = artifact_dir / name
        if path.suffix == ".json":
            _write_json(path, {})
        else:
            path.write_bytes((name + "\n").encode("utf-8"))
    entries = [
        verifier._identity(artifact_dir / name)
        for name in sorted(verifier._expected_artifact_names())
    ]
    _write_json(
        artifact_dir / "manifest.json",
        {
            "schema_version": "p1.mstcn_checkpoint_diagnostic.manifest.v1",
            "experiment_id": verifier.EXPERIMENT_ID,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "file_count_excluding_manifest": len(entries),
            "files": entries,
            "official_interface_reads": 0,
            "submission_created": False,
            "upload_performed": False,
        },
    )


def test_manifest_fixture_hashes_exact_inventory_and_detects_tamper(tmp_path: Path) -> None:
    verifier = _load_verifier()
    artifact_dir = tmp_path / "fixture"
    _build_manifest_fixture(verifier, artifact_dir)
    manifest, paths = verifier._read_and_verify_manifest(artifact_dir)
    assert manifest["file_count_excluding_manifest"] == 34
    assert set(paths) == verifier._expected_artifact_names()
    target = artifact_dir / "q3_width_512_seed_20260827_epoch_145_state.pt"
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(verifier.VerificationError, match="identity changed"):
        verifier._read_and_verify_manifest(artifact_dir)


def _build_phase_fixture(verifier, artifact_dir: Path, phase: str = "q3") -> tuple[Path, Path]:
    artifact_dir.mkdir()
    recipe_path = artifact_dir / "selected_recipe.json"
    _write_json(recipe_path, {"epoch": 150})
    fits = []
    for seed in verifier.SEEDS:
        history = []
        for epoch in range(1, 151):
            row = {"epoch": epoch}
            if epoch in verifier.CHECKPOINT_EPOCHS:
                row["blind_checkpoint_captured"] = True
            if epoch in verifier.STATE_EPOCHS:
                row["state_saved"] = True
            history.append(row)
        history_path = artifact_dir / f"{phase}_width_512_seed_{seed}_training_history.json"
        _write_json(history_path, history)
        states = []
        for epoch in verifier.STATE_EPOCHS:
            state_path = artifact_dir / f"{phase}_width_512_seed_{seed}_epoch_{epoch}_state.pt"
            state_path.write_bytes(f"fixture:{phase}:{seed}:{epoch}".encode())
            states.append({**verifier._identity(state_path), "epoch": epoch})
        fits.append(
            {
                "phase": phase,
                "width": 512,
                "seed": seed,
                "fresh_refit": True,
                "epochs_trained": 150,
                "source_schedule_horizon_epochs": 300,
                "checkpoint_epochs": list(verifier.CHECKPOINT_EPOCHS),
                "saved_state_epochs": list(verifier.STATE_EPOCHS),
                "history_artifact": verifier._identity(history_path),
                "state_artifacts": states,
            }
        )
    rows = 4
    arrays = {
        "epochs": np.asarray(verifier.CHECKPOINT_EPOCHS, dtype=np.int16),
        "row_probability": np.full((5, rows), 0.25, dtype=np.float32),
        "boundary_probability": np.full((5, rows, 2), 0.5, dtype=np.float32),
        "type_probability": np.full((5, rows, 5), 0.2, dtype=np.float32),
        "proposal": np.zeros((5, rows), dtype=np.int8),
        "candidate": np.zeros((5, rows), dtype=np.int8),
    }
    npz_path = artifact_dir / f"{phase}_blind_checkpoint_curve.npz"
    np.savez_compressed(npz_path, **arrays)
    receipt = {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.phase_blind.v1",
        "experiment_id": verifier.EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": phase,
        "fold": verifier.FOLDS[phase],
        "score_path": npz_path.name,
        "score_bytes": npz_path.stat().st_size,
        "score_sha256": verifier._sha256(npz_path),
        "config_sha256": verifier.EXPECTED_CONFIG_SHA256,
        "recipe_sha256": verifier._sha256(recipe_path),
        "ordered_holdout_key_sha256": "a" * 64,
        "holdout_rows": rows,
        "array_inventory": verifier._array_inventory(arrays),
        "checkpoint_epochs": list(verifier.CHECKPOINT_EPOCHS),
        "scientific_metric_epoch": 150,
        "same_truth_oracle_diagnostic_epochs": list(verifier.ORACLE_EPOCHS),
        "same_truth_oracle_promotion_evidence": False,
        "same_truth_oracle_recipe_mutation_allowed": False,
        "fit_receipts": fits,
        "same_fold_holdout_truth_columns_opened_before_receipt": 0,
        "prior_fold_metrics_computed_before_both_phase_seals": False,
        "official_interface_reads": 0,
    }
    receipt_path = artifact_dir / f"{phase}_blind_checkpoint_curve_receipt.json"
    _write_json(receipt_path, receipt)
    return receipt_path, recipe_path


def test_phase_fixture_replays_npz_history_and_state_hashes(tmp_path: Path) -> None:
    verifier = _load_verifier()
    artifact_dir = tmp_path / "phase"
    receipt_path, recipe_path = _build_phase_fixture(verifier, artifact_dir)
    receipt, arrays = verifier._verify_phase(
        artifact_dir,
        receipt_path,
        recipe_path,
        phase="q3",
    )
    assert receipt["holdout_rows"] == 4
    assert arrays["candidate"].shape == (5, 4)
    state = artifact_dir / "q3_width_512_seed_20260839_epoch_150_state.pt"
    state.write_bytes(b"changed")
    with pytest.raises(verifier.VerificationError, match="identity changed"):
        verifier._verify_phase(artifact_dir, receipt_path, recipe_path, phase="q3")


def test_encoder_fixture_uses_registered_74_numeric_to_165_runtime_width(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    split_path = tmp_path / "q3_split.json"
    encoder_path = tmp_path / "q3_encoder.json"
    receipt = {"holdout_rows": 4, "ordered_holdout_key_sha256": "a" * 64}
    _write_json(
        split_path,
        {
            "schema_version": "p1.mstcn_asrf.phase_split.v1",
            "phase": "q3",
            "fold": "2025_q3",
            "holdout_rows": 4,
            "holdout_key_sha256": "a" * 64,
            "holdout_membership_sha256": "a" * 64,
            "split_before_windowing": True,
            "cross_split_window_count": 0,
            "holdout_rows_used_to_fit_preprocessing": 0,
            "holdout_rows_used_to_train": 0,
            "holdout_truth_columns_read": 0,
            "runtime_input_features": verifier.RUNTIME_INPUT_FEATURE_COUNT,
            "feature_non_overlap_slack_hours": 1.0,
            "actual_separation_hours": 2.0,
            "required_feature_non_overlap_hours": 1.0,
        },
    )
    encoder = {
        "center": [0.0] * verifier.MODEL_NUMERIC_FEATURE_COUNT,
        "scale": [1.0] * verifier.MODEL_NUMERIC_FEATURE_COUNT,
        "station_vocab": ["A"],
        "layer_vocab": ["1"],
        "depth_regime_vocab": ["shallow"],
        "numeric_names": [
            f"feature_{index}" for index in range(verifier.MODEL_NUMERIC_FEATURE_COUNT)
        ],
        "fit_ids_sha256": "b" * 64,
        "uses_supplied_depth_regime": False,
        "depth_thresholds": [1.0, 2.0],
        "preprocessing_fit_uses_holdout_rows": False,
    }
    _write_json(encoder_path, encoder)
    verifier._verify_split_and_encoder(split_path, encoder_path, receipt, phase="q3")
    encoder["numeric_names"].extend(["forbidden_1", "forbidden_2", "forbidden_3"])
    encoder["center"].extend([0.0, 0.0, 0.0])
    encoder["scale"].extend([1.0, 1.0, 1.0])
    _write_json(encoder_path, encoder)
    with pytest.raises(verifier.VerificationError, match="numeric encoder width"):
        verifier._verify_split_and_encoder(split_path, encoder_path, receipt, phase="q3")


def _synthetic_surfaces_and_arrays(verifier):
    surfaces = {
        "q3": {
            "keys": pd.DataFrame({"station": ["A", "A", "B"]}),
            "truth": np.asarray([1, 0, 1], dtype=np.int8),
            "anchor": np.asarray([1, 0, 0], dtype=np.int8),
        },
        "q4": {
            "keys": pd.DataFrame({"station": ["A", "B"]}),
            "truth": np.asarray([0, 1], dtype=np.int8),
            "anchor": np.asarray([0, 1], dtype=np.int8),
        },
    }
    q3_candidates = np.asarray(
        [
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.int8,
    )
    q4_candidates = np.asarray(
        [
            [0, 1],
            [0, 1],
            [1, 1],
            [0, 1],
            [0, 1],
        ],
        dtype=np.int8,
    )
    return surfaces, {"q3": {"candidate": q3_candidates}, "q4": {"candidate": q4_candidates}}


def _fixed_report(verifier, surfaces, arrays):
    truth_parts = []
    anchor_parts = []
    candidate_parts = []
    station_parts = []
    folds = {}
    for phase in verifier.PHASES:
        truth = surfaces[phase]["truth"]
        anchor = surfaces[phase]["anchor"]
        candidate = arrays[phase]["candidate"][4]
        a = verifier._binary_metrics(truth, anchor)
        c = verifier._binary_metrics(truth, candidate)
        folds[phase] = {"anchor": a, "candidate": c, "delta_f1": c["f1"] - a["f1"]}
        truth_parts.append(truth)
        anchor_parts.append(anchor)
        candidate_parts.append(candidate)
        station_parts.append(surfaces[phase]["keys"]["station"].to_numpy())
    truth = np.concatenate(truth_parts)
    anchor = np.concatenate(anchor_parts)
    candidate = np.concatenate(candidate_parts)
    stations = np.concatenate(station_parts)
    a = verifier._binary_metrics(truth, anchor)
    c = verifier._binary_metrics(truth, candidate)
    added = (candidate == 1) & (anchor == 0)
    by_station = {}
    for station in sorted(set(stations)):
        mask = stations == station
        sa = verifier._binary_metrics(truth[mask], anchor[mask])
        sc = verifier._binary_metrics(truth[mask], candidate[mask])
        by_station[station] = {"anchor": sa, "candidate": sc, "delta_f1": sc["f1"] - sa["f1"]}
    return {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.fixed_metrics.v1",
        "scientific_metric_epoch": 150,
        "truth_scored_epochs": [150],
        "same_truth_oracle_epochs_pending": list(verifier.ORACLE_EPOCHS),
        "folds": folds,
        "pooled": {
            "rows": len(truth),
            "anchor": a,
            "candidate": c,
            "delta_f1": c["f1"] - a["f1"],
            "added_rows": int(added.sum()),
            "added_row_precision": float(truth[added].mean()) if added.any() else 0.0,
            "anchor_positive_removed_rows": int(np.sum((anchor == 1) & (candidate == 0))),
        },
        "by_station": by_station,
        "bootstrap": {},
        "fixed_recipe_improved_pooled": c["f1"] > a["f1"],
        "both_fold_deltas_positive": all(row["delta_f1"] > 0 for row in folds.values()),
        "official_probe_authorized": False,
        "three_official_points_claimed": False,
    }


def _oracle_report(verifier, surfaces, arrays, recipe_path: Path, decision_path: Path):
    rows = []
    for epoch in verifier.ORACLE_EPOCHS:
        index = verifier.CHECKPOINT_EPOCHS.index(epoch)
        truth_parts = []
        anchor_parts = []
        candidate_parts = []
        folds = {}
        for phase in verifier.PHASES:
            truth = surfaces[phase]["truth"]
            anchor = surfaces[phase]["anchor"]
            candidate = arrays[phase]["candidate"][index]
            a = verifier._binary_metrics(truth, anchor)
            c = verifier._binary_metrics(truth, candidate)
            folds[phase] = {
                "anchor_f1": a["f1"],
                "candidate_f1": c["f1"],
                "delta_f1": c["f1"] - a["f1"],
            }
            truth_parts.append(truth)
            anchor_parts.append(anchor)
            candidate_parts.append(candidate)
        truth = np.concatenate(truth_parts)
        anchor = np.concatenate(anchor_parts)
        candidate = np.concatenate(candidate_parts)
        a = verifier._binary_metrics(truth, anchor)
        c = verifier._binary_metrics(truth, candidate)
        added = (candidate == 1) & (anchor == 0)
        rows.append(
            {
                "epoch": epoch,
                "folds": folds,
                "pooled_anchor_f1": a["f1"],
                "pooled_candidate_f1": c["f1"],
                "pooled_delta_f1": c["f1"] - a["f1"],
                "added_rows": int(added.sum()),
                "added_precision": float(truth[added].mean()) if added.any() else 0.0,
                "anchor_positive_removed_rows": int(np.sum((anchor == 1) & (candidate == 0))),
            }
        )
    best = max(rows, key=lambda row: (row["pooled_delta_f1"], -row["epoch"]))
    recipe_sha = verifier._sha256(recipe_path)
    return {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.same_truth_oracle.v1",
        "fixed_scientific_decision": verifier._identity(decision_path),
        "fixed_recipe_sha256_before": recipe_sha,
        "fixed_recipe_sha256_after": recipe_sha,
        "recipe_mutated": False,
        "scientific_decision_mutated": False,
        "promotion_evidence": False,
        "official_probe_authorized": False,
        "oracle_epochs": list(verifier.ORACLE_EPOCHS),
        "rows": rows,
        "same_truth_oracle_best": {
            "epoch": best["epoch"],
            "pooled_delta_f1": best["pooled_delta_f1"],
        },
    }


def test_synthetic_fixed_and_oracle_metrics_are_recomputed(tmp_path: Path) -> None:
    verifier = _load_verifier()
    surfaces, arrays = _synthetic_surfaces_and_arrays(verifier)
    fixed = _fixed_report(verifier, surfaces, arrays)
    recomputed = verifier._recompute_fixed_metrics(fixed, surfaces, arrays)
    assert recomputed["epoch"] == 150
    recipe_path = tmp_path / "selected_recipe.json"
    decision_path = tmp_path / "fixed_epoch_150_decision.json"
    _write_json(recipe_path, {"epoch": 150})
    _write_json(decision_path, {"sealed": True})
    oracle = _oracle_report(verifier, surfaces, arrays, recipe_path, decision_path)
    replay = verifier._recompute_oracle(
        oracle,
        surfaces,
        arrays,
        recipe_path=recipe_path,
        decision_path=decision_path,
    )
    assert replay["best_epoch"] in verifier.ORACLE_EPOCHS
    fixed["pooled"]["delta_f1"] += 0.01
    with pytest.raises(verifier.VerificationError, match="pooled delta"):
        verifier._recompute_fixed_metrics(fixed, surfaces, arrays)


def test_order_fixture_rejects_decision_after_oracle(tmp_path: Path) -> None:
    verifier = _load_verifier()
    artifact_dir = tmp_path / "order"
    artifact_dir.mkdir()
    names = [
        "q2_plateau_revalidation.json",
        "selected_recipe.json",
        "q3_blind_checkpoint_curve_receipt.json",
        "q4_blind_checkpoint_curve_receipt.json",
        "blind_semantic_replays.json",
        "fixed_epoch_150_metrics.json",
        "fixed_epoch_150_decision.json",
        "same_truth_oracle_diagnostic.json",
        "terminal_result.json",
        "manifest.json",
    ]
    base = datetime(2026, 8, 27, tzinfo=UTC)
    for index, name in enumerate(names):
        path = artifact_dir / name
        path.write_text("{}\n", encoding="utf-8")
        stamp = int((base + timedelta(seconds=index)).timestamp() * 1_000_000_000)
        os.utime(path, ns=(stamp, stamp))
    receipts = {
        "q3": {"created_at_utc": (base + timedelta(seconds=2, microseconds=500000)).isoformat()},
        "q4": {"created_at_utc": (base + timedelta(seconds=3, microseconds=500000)).isoformat()},
    }
    terminal = {
        "started_at_utc": base.isoformat(),
        "completed_at_utc": (base + timedelta(seconds=8)).isoformat(),
    }
    manifest = {"created_at_utc": (base + timedelta(seconds=9)).isoformat()}
    assert verifier._verify_ordering(artifact_dir, manifest, receipts, terminal)[
        "post_hoc_limit"
    ].startswith("BLIND_CHRONOLOGY")
    decision = artifact_dir / "fixed_epoch_150_decision.json"
    oracle = artifact_dir / "same_truth_oracle_diagnostic.json"
    decision_stamp = int((base + timedelta(seconds=8)).timestamp() * 1_000_000_000)
    oracle_stamp = int((base + timedelta(seconds=7)).timestamp() * 1_000_000_000)
    os.utime(decision, ns=(decision_stamp, decision_stamp))
    os.utime(oracle, ns=(oracle_stamp, oracle_stamp))
    with pytest.raises(verifier.VerificationError, match="filesystem ordering"):
        verifier._verify_ordering(artifact_dir, manifest, receipts, terminal)


def test_control_flags_reject_official_or_upload_claims() -> None:
    verifier = _load_verifier()
    verifier._verify_control_flags(
        {"official_interface_reads": 0, "submission_created": False, "upload_performed": False},
        label="fixture",
    )
    with pytest.raises(verifier.VerificationError, match="upload_performed"):
        verifier._verify_control_flags({"upload_performed": True}, label="fixture")
