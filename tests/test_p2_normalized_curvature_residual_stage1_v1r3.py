from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_restore.normalized_curvature_residual import (
    PUBLIC_LAYERS,
    build_normalized_curvature_design,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts/run_p2_normalized_curvature_residual_stage1_v1r3.py"
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/p2_normalized_curvature_residual_lgbm_stage1_v1r3.json"
)
V1R2_CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/p2_normalized_curvature_residual_lgbm_stage1_v1r2.json"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("ncr_stage1_v1r3_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature_frame() -> pd.DataFrame:
    rows = 4
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-08-01", periods=rows, freq="h", tz="UTC"),
            "layer": [2, 3, 4, 2],
            "target": [9.6, 9.1, 8.2, 9.4],
            "baseline": [9.5, 9.0, 8.0, 9.2],
            "target_depth": [5.0, 10.0, 15.0, 5.0],
            "public_temp_range": [2.0, 0.2, 3.0, 1.5],
            "public_temp_count": [5, 4, 5, 5],
            "doy_sin": [0.1, 0.2, 0.3, 0.4],
            "doy_cos": [0.9, 0.8, 0.7, 0.6],
            "hour_sin": [0.0, 0.2, 0.4, 0.6],
            "hour_cos": [1.0, 0.8, 0.6, 0.4],
            "m2_sin": [0.2, 0.3, 0.4, 0.5],
            "m2_cos": [0.8, 0.7, 0.6, 0.5],
        }
    )
    nominal = {1: 0.0, 5: 20.0, 6: 30.0, 7: 40.0, 8: 50.0}
    temperatures = {
        1: [10.5, np.nan, 10.0, 10.0],
        5: [8.5, 8.8, 7.0, 8.5],
        6: [8.0, 8.9, 6.9, 8.4],
        7: [7.8, 8.7, 6.8, 8.3],
        8: [7.5, 8.6, 6.7, 8.2],
    }
    for public_layer in PUBLIC_LAYERS:
        frame[f"temp_{public_layer}"] = temperatures[public_layer]
        frame[f"psal_{public_layer}"] = np.asarray(
            [34.0, 34.1, 34.2, 34.3], dtype=float
        ) + public_layer * 0.001
        frame[f"nominal_{public_layer}"] = nominal[public_layer]
        frame[f"depth_{public_layer}"] = nominal[public_layer] + np.asarray(
            [0.0, 0.1, -0.1, 0.2]
        )
    return frame


def _dummy_event_base() -> dict[str, str]:
    return {
        "contract_sha256": "c" * 64,
        "bundle_sha256": "b" * 64,
        "attempt_token_sha256": "t" * 64,
    }


def test_strict_preflight_pins_full_v1r3_bundle_runtime_and_literal_data() -> None:
    runner = _load_runner_module()
    config, bundle = runner._verify_static_bundle(CONFIG_PATH)
    assert config["experiment_id"].endswith("_v1r3")
    assert runner._sha256(CONFIG_PATH) == runner.EXPECTED_CONFIG_SHA256
    assert len(bundle["implementation_pins"]) == 5
    assert len(bundle["superseded_lineage_pins"]) == 8
    assert len(bundle["immutable_references"]) == 3
    packages = bundle["runtime_pins"]["packages"]
    assert packages["scipy"] == "1.18.0"
    assert packages["joblib"] == "1.5.3"
    assert packages["threadpoolctl"] == "3.6.0"
    assert len(bundle["runtime_pins"]["lightgbm_native_files"]) == 2
    readiness = runner._literal_source_readiness(
        config,
        bundle,
        require_environment=False,
        environ={},
    )
    assert readiness["bytes"] == 49058719
    assert readiness["sha256"] == (
        "cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a"
    )
    assert readiness["permanent_claim_created"] is False
    state = runner._inspect_control_state(config)
    assert state["claim_exists"] is False
    assert state["journal_exists"] is False
    assert state["final_exists"] is False
    assert state["orphan_or_active_staging_count"] == 0


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({}, "P2_DATA_DIR is required"),
        ({"P2_DATA_DIR": r"C:\wrong\p2\directory"}, "differs from the approved"),
    ],
)
def test_missing_or_wrong_execution_environment_fails_before_claim(
    environment: dict[str, str], message: str
) -> None:
    runner = _load_runner_module()
    config, bundle = runner._verify_static_bundle(CONFIG_PATH)
    claim_calls: list[object] = []

    def forbidden_claim(*args):
        claim_calls.append(args)
        raise AssertionError("claim must not be reached")

    with pytest.raises(RuntimeError, match=message):
        runner._prepare_readiness_then_acquire(
            config,
            bundle,
            environ=environment,
            claim_fn=forbidden_claim,
        )
    assert claim_calls == []


def test_wrong_literal_observations_path_fails_before_claim() -> None:
    runner = _load_runner_module()
    config, bundle = runner._verify_static_bundle(CONFIG_PATH)
    altered = json.loads(json.dumps(config))
    approved_parent = altered["data_contract"]["approved_literal_data_directory"]
    altered["data_contract"]["approved_literal_observations_path"] = str(
        Path(approved_parent) / "wrong_observations.csv"
    )
    claim_calls: list[object] = []

    def forbidden_claim(*args):
        claim_calls.append(args)
        raise AssertionError("claim must not be reached")

    with pytest.raises((FileNotFoundError, RuntimeError)):
        runner._prepare_readiness_then_acquire(
            altered,
            bundle,
            environ={"P2_DATA_DIR": approved_parent},
            claim_fn=forbidden_claim,
        )
    assert claim_calls == []


def test_runner_imports_no_numerical_module_before_preflight() -> None:
    code = f"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('ncr_v1r3_isolation', {str(RUNNER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
blocked = sorted(name for name in sys.modules if name in {{'numpy', 'pandas', 'lightgbm'}} or name.startswith('p2_restore'))
print(json.dumps(blocked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == []


def test_actual_design_columns_equal_sealed_allow_list_exactly() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    design = build_normalized_curvature_design(_feature_frame())
    expected = config["feature_contract"]["allowed_feature_columns"]
    assert list(design.features.columns) == expected
    assert {
        "log1p_profile_scale",
        "log1p_psal_scale",
        "log1p_depth_scale",
    }.issubset(expected)
    assert any("profile_scale^-2" in item for item in config["scientific_limitations"])


def test_v1r3_preserves_v1r2_scientific_model_feature_seed_and_gate_contract() -> None:
    old = json.loads(V1R2_CONFIG_PATH.read_text(encoding="utf-8"))
    new = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for name in (
        "stage1_split",
        "target_contract",
        "model",
        "metrics",
        "stage1_gate",
        "stage2_gate_contract_not_executed_by_this_runner",
    ):
        assert new[name] == old[name]
    assert new["selection_policy"]["stage1_model_fit_count"] == old["selection_policy"][
        "stage1_model_fit_count"
    ]
    for name in (
        "public_layers",
        "target_layers",
        "forbidden_features",
        "salinity_scale_floor",
        "depth_scale_floor_m",
    ):
        assert new["feature_contract"][name] == old["feature_contract"][name]
    assert new["implementation_pins"]["normalized_curvature_module"]["sha256"] == old[
        "implementation_pins"
    ]["normalized_curvature_module"]["sha256"]
    assert new["feature_contract"] == old["feature_contract"]
    assert new["superseded_execution_forbidden"] == {
        "enforced": True,
        "experiment_ids": [
            "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1",
            "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r2",
        ],
        "policy": (
            "Only this sealed v1r3 runner can become execution-eligible; "
            "v1 and v1r2 must remain unexecuted."
        ),
    }


def test_exclusive_claim_has_exactly_one_concurrent_winner(tmp_path: Path) -> None:
    runner = _load_runner_module()
    claim = tmp_path / "stable.claim.json"
    barrier = threading.Barrier(2)

    def attempt(index: int) -> str:
        barrier.wait(timeout=5)
        try:
            runner._exclusive_create_json(claim, {"winner": index})
        except FileExistsError:
            return "LOST_FAIL_CLOSED"
        return "WON"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, (1, 2)))
    assert sorted(outcomes) == ["LOST_FAIL_CLOSED", "WON"]
    assert json.loads(claim.read_text(encoding="utf-8"))["winner"] in {1, 2}


def test_crash_after_durable_reservation_forbids_rerun_and_duplicate_slot(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    paths = {
        "final": tmp_path / "final",
        "claim": tmp_path / "stable.claim.json",
        "journal": tmp_path / "attempt.ndjson",
    }
    runner._exclusive_create_json(paths["claim"], {"permanent": True})
    initial = runner._journal_event(
        "ATTEMPT_CLAIMED",
        **_dummy_event_base(),
        physical_fit_slots=[
            {"slot": slot, "seed": 100 + slot, "status": "UNRESERVED"}
            for slot in (1, 2, 3)
        ],
    )
    runner._exclusive_create_bytes(paths["journal"], runner._journal_line(initial))
    runner._reserve_fit_slot(
        paths["journal"], slot=1, seed=101, **_dummy_event_base()
    )
    state = runner._inspect_paths(paths)
    assert state["eligible"] is False
    assert state["status"] == "INCOMPLETE_OR_CONCURRENT_ATTEMPT_NO_RERUN"
    assert runner._fit_slot_states(runner._read_journal(paths["journal"]))[1] == "RESERVED"
    with pytest.raises(RuntimeError, match="already consumed"):
        runner._reserve_fit_slot(
            paths["journal"], slot=1, seed=101, **_dummy_event_base()
        )


def test_torn_journal_and_orphan_staging_are_reported_fail_closed(tmp_path: Path) -> None:
    runner = _load_runner_module()
    final = tmp_path / "result_bundle"
    claim = tmp_path / "claim"
    journal = tmp_path / "journal.ndjson"
    claim.write_text("claim", encoding="utf-8")
    journal.write_bytes(b'{"event":"ATTEMPT_CLAIMED"')
    (tmp_path / f".{final.name}.staging-orphan").mkdir()
    state = runner._inspect_paths({"final": final, "claim": claim, "journal": journal})
    assert state["eligible"] is False
    assert state["status"] == "CORRUPT_OR_TORN_JOURNAL_NO_RERUN"
    assert state["journal_parse_error"] is not None
    assert state["orphan_or_active_staging_count"] == 1


def test_hash_read_binding_keeps_verified_bytes_after_path_mutation(tmp_path: Path) -> None:
    runner = _load_runner_module()
    source = tmp_path / "observations.csv"
    original = b"x,y\n1,2\n"
    source.write_bytes(original)
    expected_sha = hashlib.sha256(original).hexdigest()
    with runner._held_verified_bytes(
        source, expected_bytes=len(original), expected_sha256=expected_sha
    ) as (handle, captured, digest):
        source.write_bytes(b"mutated path bytes")
        assert handle.closed is False
        assert io.BytesIO(captured).read() == original
        assert digest == {"bytes": len(original), "sha256": expected_sha}
        runner._verify_captured_source(
            captured,
            expected_bytes=len(original),
            expected_sha256=expected_sha,
            label="synthetic",
        )
    assert source.read_bytes() != captured


def test_parent_timeout_uses_only_remaining_wall_and_fails_closed() -> None:
    runner = _load_runner_module()
    observed: dict[str, float] = {}

    def fake_run(command, **kwargs):
        observed["timeout"] = float(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(runner.HardWallTimeout):
        runner._run_subprocess_with_deadline(
            ["synthetic-worker"],
            environment={},
            deadline_epoch=105.0,
            run_fn=fake_run,
            now_fn=lambda: 100.0,
        )
    assert observed["timeout"] == pytest.approx(5.0)


def test_three_slots_can_each_be_reserved_and_completed_only_once(tmp_path: Path) -> None:
    runner = _load_runner_module()
    journal = tmp_path / "attempt.ndjson"
    initial = runner._journal_event("ATTEMPT_CLAIMED", **_dummy_event_base())
    runner._exclusive_create_bytes(journal, runner._journal_line(initial))
    for slot in (1, 2, 3):
        runner._reserve_fit_slot(journal, slot=slot, seed=100 + slot, **_dummy_event_base())
        runner._complete_fit_slot(
            journal,
            slot=slot,
            seed=100 + slot,
            elapsed_seconds=0.01,
            **_dummy_event_base(),
        )
    assert runner._fit_slot_states(runner._read_journal(journal)) == {
        1: "COMPLETED",
        2: "COMPLETED",
        3: "COMPLETED",
    }
    with pytest.raises(RuntimeError):
        runner._reserve_fit_slot(journal, slot=3, seed=103, **_dummy_event_base())
    with pytest.raises(RuntimeError):
        runner._reserve_fit_slot(journal, slot=4, seed=104, **_dummy_event_base())


def test_atomic_publish_exposes_only_complete_two_file_final(tmp_path: Path) -> None:
    runner = _load_runner_module()
    final = tmp_path / "final_artifact"
    durability = runner._publish_aggregate(
        final,
        {"status": "SYNTHETIC"},
        {"schema_version": "synthetic.manifest.v1"},
    )
    assert final.is_dir()
    assert sorted(path.name for path in final.iterdir()) == ["manifest.json", "result.json"]
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["result"]["sha256"] == runner._sha256(final / "result.json")
    assert not list(tmp_path.glob(".final_artifact.staging-*"))
    assert set(durability) == {
        "staging_directory_fsync_complete_supported",
        "final_parent_directory_fsync_supported",
    }
