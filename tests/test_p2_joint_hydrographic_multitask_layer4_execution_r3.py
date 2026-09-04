from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from p2_restore import joint_hydrographic_multitask_layer4_contract_r2 as r2_guard
from p2_restore import joint_hydrographic_multitask_layer4_contract_r3 as guard
from p2_restore import joint_hydrographic_multitask_layer4_execution_r3 as engine

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PERSONAL_ROOT = r"[a-z]:" + r"[\\/]" + r"(?:users|documents|downloads)[\\/]"
POSIX_PERSONAL_ROOT = "/" + "home" + r"/[^/]+/"
DATA_DIR_LITERAL_PATTERN = re.compile(
    rf"(?i)(?:{WINDOWS_PERSONAL_ROOT}|{POSIX_PERSONAL_ROOT})"
)


def _config() -> dict[str, Any]:
    return guard.load_canonical_config(ROOT)


def _dummy_pin(path: str, token: str = "0") -> dict[str, Any]:
    return {"path": path, "sha256": token * 64, "bytes": 1}


def _reference_frame(fold: str = "outer_2024_sep_oct") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [fold, fold, fold],
            "station": ["S-ORS"] * 3,
            "layer": [2, 3, 4],
            "time": pd.to_datetime(
                [
                    "2024-09-01T00:00:00+09:00",
                    "2024-09-01T00:00:00+09:00",
                    "2024-09-01T00:00:00+09:00",
                ],
                utc=True,
            ),
        }
    )


def test_canonical_config_and_frozen_scientific_pins_are_exact() -> None:
    config_path = ROOT / guard.CONFIG_RELATIVE
    assert guard.sha256_file(config_path) == guard.CONFIG_SHA256
    config = _config()
    guard.validate_config(config)
    expected = {
        "design": "7c8fd3de7300995e73a3baeafd2af7a3c5cf33cb9cc93d636cb4ce1b13bd74f2",
        "pure_model": "8d667256fe37e5b6bf72e09f233efd17b514b40dae1597a047c0c343a344a314",
        "pure_model_tests": "695bd872965377692799b659a40459c12771ca7ae3efdf5c8f38015eb234d2ca",
        "static_preflight": "25ed914813b10478caabe63e89adaf744dec0d436d10a43df58e643e7e830cf6",
        "execution_design": "06c9d1183a7ac523a34d0f4d072cc154b9daccc0611c9020de29da19ec503ee8",
    }
    assert {
        role: config["scientific_surface"][role]["sha256"] for role in expected
    } == expected
    assert config["scientific_surface"]["scientific_changes_from_frozen_design"] == 0


def test_transitive_and_stage_a_pins_verify_without_numerical_import() -> None:
    config = _config()
    before = {
        name
        for name in sys.modules
        if name.split(".")[0] in {"numpy", "pandas", "scipy", "sklearn", "torch"}
    }
    surface = guard.verify_scientific_surface(ROOT, config)
    sources = guard.verify_pin_map(ROOT, config["source_pins"], label="test source")
    stage_a = guard.verify_stage_a_reference(ROOT, config)
    after = {
        name
        for name in sys.modules
        if name.split(".")[0] in {"numpy", "pandas", "scipy", "sklearn", "torch"}
    }
    assert surface["checks"]["static_steps"]
    assert set(sources) == {"PACKAGE_INIT", "DEEP_DATA", "FEATURES", "DATA", "CURVE_GATE"}
    assert stage_a["status"] == "PASS_EXACT_STAGE_A_V3_REFERENCE"
    # This test module already imported numpy/pandas; the guard itself adds none.
    assert after == before


def test_prelock_guard_import_in_clean_process_loads_no_numerical_stack() -> None:
    script = """
import importlib.util, json, sys
from pathlib import Path
root = Path.cwd()
name = 'p2_restore.joint_hydrographic_multitask_layer4_contract_r3'
spec = importlib.util.spec_from_file_location(name, root / 'src/p2_restore/joint_hydrographic_multitask_layer4_contract_r3.py')
guard = importlib.util.module_from_spec(spec)
sys.modules[name] = guard
spec.loader.exec_module(guard)
prefixes = ('numpy','pandas','scipy','sklearn','torch')
print(json.dumps(sorted(n for n in sys.modules if any(n == p or n.startswith(p + '.') for p in prefixes))))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout.strip()) == []


def test_execution_plan_is_fold_major_45_fits_and_6132_steps() -> None:
    plan = engine.build_execution_plan(_config())
    assert plan["fold_major_order"] == [
        "outer_2024_sep_oct",
        "outer_2025_may_jun",
        "outer_2025_jul_aug",
    ]
    assert plan["fit_cells"] == plan["blind_prediction_arrays"] == 45
    assert plan["fold_commitments"] == 3
    assert plan["optimizer_steps"] == 6132
    assert plan["prior_v1_optimizer_steps_not_reused"] == 56
    assert plan["prior_r2_optimizer_steps_not_reused"] == 392
    assert plan["fresh_r3_optimizer_steps"] == 6132
    assert plan["bootstrap_replicates_per_fraction"] == 5000
    assert plan["candidate_predictions"] == plan["test_predictions"] == plan["uploads"] == 0


def test_o_excl_binary_writer_is_full_and_duplicate_fails(tmp_path: Path) -> None:
    path = tmp_path / "one-shot.bin"
    payload = bytes(range(256)) * 8193
    guard.exclusive_bytes(path, payload)
    assert path.read_bytes() == payload
    with pytest.raises(FileExistsError):
        guard.exclusive_bytes(path, b"forgery")
    assert path.read_bytes() == payload
    source = (ROOT / guard.IMPLEMENTATION_ROLES["GUARD"]).read_text(encoding="utf-8")
    assert "os.O_EXCL" in source
    assert 'getattr(os, "O_BINARY", 0)' in source
    assert "while written < len(view)" in source


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value":1,"value":2}', encoding="utf-8")
    with pytest.raises(guard.Layer4ContractError, match="duplicate JSON key"):
        guard.strict_json_object(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(guard.Layer4ContractError, match="non-finite JSON"):
        guard.strict_json_object(nonfinite)


def test_stdlib_npy_verifier_recomputes_exact_float64_payload(tmp_path: Path) -> None:
    values = np.array([-5.0, 0.25, 45.0], dtype="<f8")
    path = tmp_path / "prediction.npy"
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    payload, rows = guard._npy_little_endian_float64_payload(path)
    assert rows == 3
    assert payload == values.tobytes(order="C")
    forged = tmp_path / "nonfinite.npy"
    with forged.open("wb") as stream:
        np.save(stream, np.array([np.nan], dtype="<f8"), allow_pickle=False)
    with pytest.raises(guard.Layer4ContractError, match="non-finite"):
        guard._npy_little_endian_float64_payload(forged)


def test_config_mutations_fail_closed() -> None:
    config = _config()
    mutations = []
    changed = copy.deepcopy(config)
    changed["official_promotion_allowed"] = True
    mutations.append(changed)
    changed = copy.deepcopy(config)
    changed["curve_protocol"]["total_optimizer_steps"] = 6131
    mutations.append(changed)
    changed = copy.deepcopy(config)
    changed["execution_policy"]["active_fold_target_temp_psal_scalar_decode_before_commitment"] = 1
    mutations.append(changed)
    changed = copy.deepcopy(config)
    changed["prefix_pins"][0]["timestamp_order_sha256"] = "0" * 64
    mutations.append(changed)
    changed = copy.deepcopy(config)
    changed["output_contract"]["candidate_generated"] = True
    mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(guard.Layer4ContractError):
            guard.validate_config(mutation)


def test_control_apis_accept_no_caller_supplied_hashes() -> None:
    assert set(inspect.signature(guard.consume_attempt_lock).parameters) == {
        "root",
        "data_dir",
        "config",
    }
    assert set(inspect.signature(guard.issue_execution_capability).parameters) == {
        "root",
        "data_dir",
        "config",
    }
    runner = (ROOT / guard.IMPLEMENTATION_ROLES["RUNNER"]).read_text(encoding="utf-8")
    assert "qa_sha256=" not in runner
    assert "authorization_sha256=" not in runner
    assert "lock_sha256=" not in runner


def test_forged_capability_rejected_before_fit_or_truth_access(tmp_path: Path) -> None:
    config = _config()
    fold = config["curve_protocol"]["folds"][0]
    with pytest.raises(PermissionError, match="canonical live post-lock"):
        engine._fit_predict_cell(
            object(),
            workspace=ROOT,
            output=tmp_path,
            config=config,
            fold=fold,
            fraction=0.4,
            seed=20260823,
            panel=None,
            reference_fold=None,
            fold_audit={},
            numerical=None,
            progress=None,
        )
    with pytest.raises(PermissionError, match="canonical live post-lock"):
        engine._load_metric_truth_after_commitment(
            object(),
            root=ROOT,
            config=config,
            observations_path=tmp_path / "must-not-open.csv",
            pd_module=None,
            np_module=None,
        )


def test_capability_state_machine_rejects_arbitrary_order_and_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    capability = guard.ExecutionCapability(
        root=str(ROOT),
        config_sha256=guard.CONFIG_SHA256,
        preflight_summary_sha256="1" * 64,
        attempt_lock_sha256="2" * 64,
        qa_sha256="3" * 64,
        authorization_sha256="4" * 64,
        implementation_pins_sha256="5" * 64,
        nonce="6" * 64,
    )
    monkeypatch.setattr(guard, "_LIVE_CAPABILITY", capability)
    monkeypatch.setattr(guard, "_LIVE_PHASE", "EXECUTION_ACTIVE")
    monkeypatch.setattr(guard, "_COMPLETED_CELLS", [])
    monkeypatch.setattr(guard, "_ACTIVE_CELL", None)
    monkeypatch.setattr(guard, "_FOLD_COMMITMENTS", [])
    monkeypatch.setattr(guard, "_AGGREGATE_COMMITMENT", None)
    monkeypatch.setattr(guard, "_require_core", lambda *args, **kwargs: capability)
    with pytest.raises(PermissionError, match="arbitrary or replayed"):
        guard.claim_cell(
            capability,
            root=ROOT,
            config=config,
            fold="outer_2025_may_jun",
            fraction=0.4,
            seed=20260823,
        )
    guard.claim_cell(
        capability,
        root=ROOT,
        config=config,
        fold="outer_2024_sep_oct",
        fraction=0.4,
        seed=20260823,
    )
    with pytest.raises(PermissionError, match="phase is unavailable"):
        guard.claim_cell(
            capability,
            root=ROOT,
            config=config,
            fold="outer_2024_sep_oct",
            fraction=0.4,
            seed=20260823,
        )
    monkeypatch.setattr(guard, "_LIVE_PHASE", "EXECUTION_ACTIVE")
    monkeypatch.setattr(guard, "_ACTIVE_CELL", None)
    monkeypatch.setattr(guard, "_COMPLETED_CELLS", [("outer_2024_sep_oct", 0.4, 20260823)] * 14)
    with pytest.raises(PermissionError, match="15 registered cells"):
        guard.claim_fold_commitment(
            capability,
            root=ROOT,
            config=config,
            fold="outer_2024_sep_oct",
        )


def test_active_fold_target_poison_is_never_decoded(tmp_path: Path) -> None:
    observations = tmp_path / "observations.csv"
    observations.write_bytes(
        b"station,year,layer,time,temp,psal,depth,nominal_depth\n"
        b"S-ORS,2024,1,2024-09-01T00:00:00+09:00,20.0,31.0,3.0,3.0\n"
        b"S-ORS,2024,2,2024-08-01T00:00:00+09:00,19.0,31.1,6.0,6.0\n"
        b"S-ORS,2024,2,2024-09-01T00:00:00+09:00,POISON_TEMP,POISON_PSAL,6.0,6.0\n"
    )
    frame, audit = engine._load_fold_blind_observations(
        observations,
        fold={
            "name": "outer_2024_sep_oct",
            "start_kst": "2024-09-01T00:00:00+09:00",
            "stop_kst": "2024-11-01T00:00:00+09:00",
        },
        embargo_days=7,
        verified_prior_fold_commitments=0,
        expected_prior_fold_commitments=0,
        pd_module=pd,
        np_module=np,
    )
    poisoned = frame.loc[
        frame["time"].eq("2024-09-01T00:00:00+09:00") & frame["layer"].eq(2)
    ]
    assert poisoned[["temp", "psal"]].isna().all().all()
    assert audit["active_fold_target_rows"] == 1
    assert audit["active_fold_target_temp_psal_scalar_fields_decoded_or_converted"] == 0
    assert audit["withheld_target_temp_psal_scalar_fields_decoded_or_converted"] == 0


def test_selective_decoder_accepts_registered_empty_numeric_fields(tmp_path: Path) -> None:
    observations = tmp_path / "observations.csv"
    observations.write_bytes(
        b"station,year,layer,time,temp,psal,depth,nominal_depth\n"
        b"S-ORS,2024,1,2024-09-01T00:00:00+09:00,20.0,,3.0,3.0\n"
        b"S-ORS,2024,2,2024-08-01T00:00:00+09:00,19.0,,6.0,6.0\n"
    )
    frame, _audit = engine._load_fold_blind_observations(
        observations,
        fold={
            "name": "outer_2024_sep_oct",
            "start_kst": "2024-09-01T00:00:00+09:00",
            "stop_kst": "2024-11-01T00:00:00+09:00",
        },
        embargo_days=7,
        verified_prior_fold_commitments=0,
        expected_prior_fold_commitments=0,
        pd_module=pd,
        np_module=np,
    )
    assert np.isnan(frame.loc[frame["layer"].eq(1), "psal"].iloc[0])
    assert np.isnan(frame.loc[frame["layer"].eq(2), "psal"].iloc[0])


def test_previous_fold_history_loader_requires_verified_commitment(tmp_path: Path) -> None:
    observations = tmp_path / "observations.csv"
    observations.write_bytes(
        b"station,year,layer,time,temp,psal,depth,nominal_depth\n"
        b"S-ORS,2024,1,2024-09-01T00:00:00+09:00,20,31,3,3\n"
    )
    with pytest.raises(PermissionError, match="previous fold commitment"):
        engine._load_fold_blind_observations(
            observations,
            fold={
                "name": "outer_2025_may_jun",
                "start_kst": "2025-05-01T00:00:00+09:00",
                "stop_kst": "2025-07-01T00:00:00+09:00",
            },
            embargo_days=7,
            verified_prior_fold_commitments=0,
            expected_prior_fold_commitments=1,
            pd_module=pd,
            np_module=np,
        )


def test_frozen_timestamp_digest_uses_little_endian_nanoseconds() -> None:
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(
            ["2024-01-01T00:00:00+09:00", "2024-01-01T00:10:00+09:00"],
            utc=True,
        )
    )
    expected = hashlib.sha256(
        timestamps.to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes()
    ).hexdigest()
    assert engine._timestamp_order_sha256(timestamps, np_module=np) == expected


def test_csv_float_roundtrip_is_in_memory_and_finite() -> None:
    values = np.array([-5.0, 1.0 / 3.0, 44.99999999999999], dtype=np.float64)
    observed = engine._csv_float_roundtrip(values, pd_module=pd, np_module=np)
    assert observed.shape == values.shape
    assert np.isfinite(observed).all()
    assert observed.min() >= -5.0 and observed.max() <= 45.0


def test_blind_predictions_attach_by_fold_not_stage_a_block_order() -> None:
    folds = [
        "outer_2024_sep_oct",
        "outer_2025_may_jun",
        "outer_2025_jul_aug",
    ]
    # The sealed Stage-A OOF byte order is 2024, Jul/Aug, May/Jun.
    frame = pd.DataFrame(
        {
            "fold": [folds[0], folds[2], folds[2], folds[1]],
            "row": [0, 0, 1, 0],
        }
    )
    seed = 20260823
    predictions = {
        (folds[0], 1.0, seed): np.array([10.0]),
        (folds[1], 1.0, seed): np.array([20.0]),
        (folds[2], 1.0, seed): np.array([30.0, 31.0]),
    }
    observed = engine._attach_blind_seed_predictions(
        frame,
        fraction=1.0,
        fold_order=folds,
        seeds=[seed],
        blind_predictions=predictions,
        np_module=np,
    )
    assert observed[f"challenger_seed_{seed}"].tolist() == [10.0, 30.0, 31.0, 20.0]


def test_bootstrap_uses_registered_fold_order_not_oof_block_order() -> None:
    folds = [
        "outer_2024_sep_oct",
        "outer_2025_may_jun",
        "outer_2025_jul_aug",
    ]
    rows = []
    for fold_index, fold in enumerate(folds):
        for day in range(2):
            for layer in (2, 3, 4):
                truth = float(layer + day)
                rows.append(
                    {
                        "fold": fold,
                        "layer": layer,
                        "time": pd.Timestamp(f"2025-01-0{day + 1}T00:00:00Z"),
                        "truth": truth,
                        "reference": truth + 0.2,
                        "challenger": truth + 0.1 * (fold_index + day + 1),
                    }
                )
    frame = pd.DataFrame(rows)
    reordered = pd.concat(
        [
            frame.loc[frame["fold"].eq(folds[0])],
            frame.loc[frame["fold"].eq(folds[2])],
            frame.loc[frame["fold"].eq(folds[1])],
        ],
        ignore_index=True,
    )
    kwargs = {
        "reference_column": "reference",
        "challenger_column": "challenger",
        "counts": {"2": 1, "3": 1, "4": 1},
        "fold_order": folds,
        "replicates": 200,
        "seed": 20260823,
        "pd_module": pd,
        "np_module": np,
    }
    first = engine._paired_kst_day_bootstrap(frame, **kwargs)
    second = engine._paired_kst_day_bootstrap(reordered, **kwargs)
    assert first == second
    assert list(first[1]["fold_cluster_counts"]) == folds


def test_fold_commitment_binds_exact_order_keys_models_predictions_and_receipts() -> None:
    config = _config()
    references = {
        float(fraction): _reference_frame() for fraction in config["curve_protocol"]["prefix_fractions"]
    }
    cells = []
    index = 0
    for fraction in config["curve_protocol"]["prefix_fractions"]:
        for seed in config["curve_protocol"]["seed_ids"]:
            index += 1
            cells.append(
                {
                    "fold": "outer_2024_sep_oct",
                    "fraction": float(fraction),
                    "seed": int(seed),
                    "model_bundle": _dummy_pin(f"model-{index}.pt", "1"),
                    "blind_prediction_array": _dummy_pin(f"prediction-{index}.npy", "2"),
                    "cell_receipt": _dummy_pin(f"receipt-{index}.json", "3"),
                    "prediction_values_sha256": "4" * 64,
                    "model_state_sha256": "5" * 64,
                    "optimizer_steps": 56,
                    "blindness": {
                        "active_fold_target_temp_psal_scalar_fields_decoded_or_converted": 0,
                        "outer_truth_used_for_fit_or_epoch_selection": False,
                        "future_target_truth_used_for_fit": False,
                        "layer2_and_layer3_exact_stage_a_seed_values": True,
                        "only_layer4_temperature_replaced": True,
                        "prediction_persisted_before_active_fold_truth_decode": True,
                    },
                }
            )
    audit = {
        "fold": "outer_2024_sep_oct",
        "verified_prior_fold_commitments": 0,
        "active_fold_target_temp_psal_scalar_fields_decoded_or_converted": 0,
        "withheld_target_temp_psal_scalar_fields_decoded_or_converted": 0,
        "anomaly_or_hidden_target_proxy_reads": 0,
    }
    payload = engine._fold_commitment_payload(
        workspace=ROOT,
        config=config,
        fold="outer_2024_sep_oct",
        reference_by_fraction=references,
        cells=cells,
        prior_fold_commitments=[],
        fold_audit=audit,
    )
    expected_combined = hashlib.sha256(
        bytes.fromhex(payload["validation_key_order_sha256"])
        + bytes.fromhex(payload["cells_sha256"])
    ).hexdigest()
    assert payload["cell_prediction_count"] == 15
    assert payload["combined_fold_commitment_sha256"] == expected_combined
    assert payload["active_fold_target_temp_psal_scalar_decodes_before_commitment"] == 0
    forged = cells[::-1]
    with pytest.raises(ValueError, match="15-cell order"):
        engine._fold_commitment_payload(
            workspace=ROOT,
            config=config,
            fold="outer_2024_sep_oct",
            reference_by_fraction=references,
            cells=forged,
            prior_fold_commitments=[],
            fold_audit=audit,
        )
    forged_audit = {**audit, "active_fold_target_temp_psal_scalar_fields_decoded_or_converted": 1}
    with pytest.raises(PermissionError, match="blindness audit"):
        engine._fold_commitment_payload(
            workspace=ROOT,
            config=config,
            fold="outer_2024_sep_oct",
            reference_by_fraction=references,
            cells=cells,
            prior_fold_commitments=[],
            fold_audit=forged_audit,
        )


def test_aggregate_commitment_binds_all_45_arrays_and_three_folds() -> None:
    config = _config()
    cells = []
    predictions = {}
    index = 0
    for fold in config["curve_protocol"]["fold_major_order"]:
        for fraction in config["curve_protocol"]["prefix_fractions"]:
            for seed in config["curve_protocol"]["seed_ids"]:
                index += 1
                key = (fold, float(fraction), int(seed))
                predictions[key] = np.array([index, index + 0.5], dtype=np.float64)
                prediction_sha = hashlib.sha256(
                    predictions[key].astype("<f8").tobytes()
                ).hexdigest()
                cells.append(
                    {
                        "fold": fold,
                        "fraction": float(fraction),
                        "seed": int(seed),
                        "model_bundle": _dummy_pin(f"model-{index}.pt", "1"),
                        "blind_prediction_array": _dummy_pin(f"prediction-{index}.npy", "2"),
                        "cell_receipt": _dummy_pin(f"receipt-{index}.json", "3"),
                        "prediction_values_sha256": prediction_sha,
                    }
                )
    folds = [_dummy_pin(f"fold-{index}.json", str(index + 1)) for index in range(3)]
    payload = engine._aggregate_commitment_payload(
        workspace=ROOT,
        config=config,
        fold_commitments=folds,
        cells=cells,
        blind_predictions=predictions,
        numerical=type("Numerical", (), {"np": np})(),
    )
    expected = hashlib.sha256(
        bytes.fromhex(payload["prediction_values_sha256"])
        + bytes.fromhex(payload["cell_artifacts_sha256"])
        + bytes.fromhex(payload["fold_commitments_sha256"])
    ).hexdigest()
    assert payload["cell_prediction_count"] == 45
    assert payload["fold_commitment_count"] == 3
    assert payload["combined_prediction_commitment_sha256"] == expected
    assert payload["validation_truth_scalar_decodes_before_commitment"] == 0


def test_output_allowlist_is_exact_and_has_no_candidate_test_or_upload() -> None:
    config = _config()
    files = guard.expected_output_files(config)
    directories = guard.expected_output_directories(config)
    assert len(files) == 148
    assert len(directories) == 68
    assert sum(path.endswith("model.pt") for path in files) == 45
    assert sum(path.endswith("prediction.npy") for path in files) == 45
    assert sum(path.endswith("/receipt.json") for path in files) == 45
    assert sum(path.endswith("fold_commitment.json") for path in files) == 3
    assert not any(
        token in path.lower() for path in files for token in ("candidate", "submission", "upload")
    )


def test_new_identity_contains_no_absolute_personal_path_literal() -> None:
    paths = [ROOT / relative for relative in guard.IMPLEMENTATION_ROLES.values()]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert DATA_DIR_LITERAL_PATTERN.search(text) is None, path
    config = _config()
    assert config["data_contract"]["data_directory_is_runtime_injected"] is True
    assert config["data_contract"]["personal_absolute_path_in_code_or_config"] is False


def test_check_only_is_default_and_run_never_creates_controls() -> None:
    runner = (ROOT / guard.IMPLEMENTATION_ROLES["RUNNER"]).read_text(encoding="utf-8")
    assert 'choices=("check-only", "run"), default="check-only"' in runner
    assert "exclusive_json" not in runner
    assert "exclusive_bytes" not in runner
    assert "consume_attempt_lock" in runner
    assert runner.index("consume_attempt_lock") < runner.index("import_module(ENGINE_MODULE)")


def test_authorized_runner_rejects_preloaded_engine_before_consuming_lock() -> None:
    path = ROOT / guard.IMPLEMENTATION_ROLES["RUNNER"]
    spec = importlib.util.spec_from_file_location("layer4_runner_prelock_test", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with pytest.raises(PermissionError, match="clean pre-lock process"):
        runner.run_authorized(root=ROOT, data_dir=ROOT)


def test_v1_failure_receipt_and_tombstone_are_exact_and_v1_is_frozen() -> None:
    config = _config()
    evidence = guard.verify_v1_failure_evidence(ROOT, config)
    assert evidence["status"] == "PASS_FROZEN_V1_FAILURE_TOMBSTONE"
    assert evidence["output_inventory"] == {
        "recursive_directories": 68,
        "persisted_files": 0,
        "relative_directory_listing_sha256": (
            "5251a17d479f8fe452770cf9d313a1695ba567b7be1d51162307bd509507594a"
        ),
    }
    assert all(evidence["checks"].values())
    assert evidence["pins"] == {
        "V1_FAILURE_RECEIPT": config["v1_failure_evidence"]["failure_receipt"],
        "V1_FAILURE_TOMBSTONE": config["v1_failure_evidence"]["failure_tombstone"],
        "V1_ATTEMPT_LOCK": config["v1_failure_evidence"]["consumed_v1_attempt_lock"],
    }


def test_forged_in_memory_v1_failure_proof_is_rejected() -> None:
    forged = copy.deepcopy(_config())
    forged["v1_failure_evidence"]["failure_receipt"]["sha256"] = "0" * 64
    with pytest.raises(guard.Layer4ContractError, match="pin drift"):
        guard.verify_v1_failure_evidence(ROOT, forged)


def test_r3_control_is_distinct_and_missing_qa_fails_closed() -> None:
    config = _config()
    paths = guard.stage_paths(ROOT, config)
    assert paths["control"].is_dir()
    assert "execution_r3_control" in paths["control"].as_posix()
    assert "execution_r2_control" not in paths["control"].as_posix()
    assert (paths["control"] / "r2_failure_receipt.json").is_file()
    assert (paths["control"] / "r2_failure_tombstone.json").is_file()
    assert not paths["attempt_lock"].exists()
    assert not paths["output"].exists()
    if not paths["pre_execution_qa"].exists():
        with pytest.raises(PermissionError, match="QA receipt is missing"):
            guard.verify_pre_execution_qa(ROOT, config)


def test_irrelevant_full_panel_nan_is_allowed_and_audited() -> None:
    physical = np.arange(15, dtype=np.float64).reshape(5, 3)
    physical[0, :] = np.nan
    physical[2, 0] = np.nan
    observed, audit = engine._validate_required_layer4_physical_prediction(
        physical,
        panel_rows=5,
        required_layer4_positions=np.array([1, 3], dtype=np.int64),
        np_module=np,
    )
    assert np.array_equal(observed[1:, 2], physical[1:, 2])
    assert audit == {
        "validation_domain": "REGISTERED_STAGE_A_OOF_LAYER4_POSITIONS_ONLY",
        "panel_rows": 5,
        "panel_physical_values": 15,
        "required_layer4_positions": 2,
        "required_layer4_values_finite": True,
        "nonrequired_nonfinite_physical_values": 4,
        "global_full_panel_finiteness_required": False,
        "validated_before_any_persistence": True,
    }


def test_required_layer4_oof_nan_is_rejected_before_persistence() -> None:
    physical = np.ones((5, 3), dtype=np.float64)
    physical[0, :] = np.nan
    physical[3, 2] = np.nan
    with pytest.raises(ValueError, match="required Layer-4 OOF physical prediction"):
        engine._validate_required_layer4_physical_prediction(
            physical,
            panel_rows=5,
            required_layer4_positions=np.array([1, 3], dtype=np.int64),
            np_module=np,
        )


def test_required_domain_validation_is_wired_before_first_persistence() -> None:
    fit_source = inspect.getsource(engine._fit_predict_cell)
    predict_source = inspect.getsource(engine._predict_panel_temperature)
    assert "required_layer4_positions=positions[layer4]" in fit_source
    assert "_validate_required_layer4_physical_prediction" in predict_source
    assert "_validate_assembled_layer4_prediction" in fit_source
    assert fit_source.index("required_layer4_positions=positions[layer4]") < fit_source.index(
        "guard.exclusive_bytes"
    )
    assert fit_source.index("_validate_assembled_layer4_prediction") < fit_source.index(
        "guard.exclusive_bytes"
    )
    assert guard.REQUIRED_LAYER4_ROWS_BY_FOLD == {
        "outer_2024_sep_oct": 8671,
        "outer_2025_may_jun": 8437,
        "outer_2025_jul_aug": 8913,
    }


def test_r3_correction_scope_mutations_fail_closed() -> None:
    config = _config()
    changed = copy.deepcopy(config)
    changed["implementation_correction"]["scientific_surface_changes"] = 1
    with pytest.raises(guard.Layer4ContractError, match="implementation-only correction"):
        guard.validate_config(changed)
    changed = copy.deepcopy(config)
    changed["v1_failure_evidence"]["v1_completed_optimizer_steps"] = 55
    with pytest.raises(guard.Layer4ContractError, match="failure evidence changed"):
        guard.validate_config(changed)


def test_r2_failure_receipt_tombstone_and_partial_output_are_exact() -> None:
    config = _config()
    evidence = guard.verify_r2_failure_evidence(ROOT, config)
    assert evidence["status"] == "PASS_FROZEN_R2_FAILURE_TOMBSTONE"
    assert evidence["output_inventory"] == {
        "recursive_directories": 68,
        "persisted_files": 18,
        "persisted_file_bytes": 25919649,
        "canonical_inventory_sha256": (
            "63408872ba7fa67929fa4e1007c4d2868da01ca8ae046b8901f62f3ac03a1fb9"
        ),
    }
    assert all(evidence["checks"].values())
    assert evidence["pins"] == {
        "R2_FAILURE_RECEIPT": config["r2_failure_evidence"]["failure_receipt"],
        "R2_FAILURE_TOMBSTONE": config["r2_failure_evidence"]["failure_tombstone"],
        "R2_ATTEMPT_LOCK": config["r2_failure_evidence"]["consumed_r2_attempt_lock"],
    }
    assert guard.FAILURE_EVIDENCE_ROLES == {
        "V1_FAILURE_RECEIPT",
        "V1_FAILURE_TOMBSTONE",
        "V1_ATTEMPT_LOCK",
        "R2_FAILURE_RECEIPT",
        "R2_FAILURE_TOMBSTONE",
        "R2_ATTEMPT_LOCK",
    }


def test_consumed_r2_identity_fails_closed_against_rerun_or_resume() -> None:
    r2_config = r2_guard.load_canonical_config(ROOT)
    with pytest.raises(FileExistsError, match="output already exists"):
        r2_guard.verify_execution_authorization(ROOT, r2_config)


def test_forged_in_memory_r2_failure_proof_and_inventory_are_rejected() -> None:
    forged = copy.deepcopy(_config())
    forged["r2_failure_evidence"]["failure_receipt"]["sha256"] = "0" * 64
    with pytest.raises(guard.Layer4ContractError, match="pin drift"):
        guard.verify_r2_failure_evidence(ROOT, forged)
    forged = copy.deepcopy(_config())
    forged["r2_failure_evidence"]["r2_completed_optimizer_steps"] = 391
    with pytest.raises(guard.Layer4ContractError, match="r2 failure evidence changed"):
        guard.validate_config(forged)


def test_scientific_training_fold_seed_and_gate_surface_is_deep_equal_to_v1_r2() -> None:
    r3 = _config()
    r2 = json.loads(
        (ROOT / "configs/experiments/p2_joint_hydrographic_multitask_layer4_execution_r2.json")
        .read_text(encoding="utf-8")
    )
    v1 = json.loads(
        (ROOT / "configs/experiments/p2_joint_hydrographic_multitask_layer4_execution_v1.json")
        .read_text(encoding="utf-8")
    )
    for key in (
        "scientific_surface",
        "source_pins",
        "stage_a_reference",
        "data_contract",
        "curve_protocol",
        "prefix_pins",
        "model_and_training",
        "metric_and_gates",
        "runtime_contract",
        "output_contract",
        "resource_estimate",
    ):
        assert r3[key] == r2[key] == v1[key]


def test_preserved_finite_non_layer4_outside_layer4_bounds_is_accepted_exactly() -> None:
    reference = np.array([45.995698912768603, 18.770901641638364, 2.5], dtype=np.float64)
    prediction = reference.copy()
    prediction[2] = 45.0
    layer4 = np.array([False, False, True])
    observed = engine._validate_assembled_layer4_prediction(
        prediction,
        reference,
        layer4,
        clip_bounds=(-5.0, 45.0),
        np_module=np,
    )
    assert observed[0] == 45.995698912768603
    assert observed[~layer4].astype("<f8").tobytes() == reference[~layer4].astype("<f8").tobytes()


def test_sealed_oof_070_offending_key_is_preserved_and_allowed() -> None:
    path = ROOT / _config()["stage_a_reference"]["artifacts"]["OOF_070"]["path"]
    frame = pd.read_csv(path, dtype={"fold": "string", "station": "string", "time": "string"})
    row = frame.loc[
        frame["fold"].eq("outer_2024_sep_oct")
        & frame["station"].eq("S-ORS")
        & frame["layer"].eq(2)
        & frame["time"].eq("2024-09-09T06:20:00+09:00")
    ]
    assert len(row) == 1
    value = row["seed_20260823"].to_numpy(dtype=np.float64)[0]
    assert value == 45.9956989127686
    reference = np.array([value, 20.0, 3.0], dtype=np.float64)
    prediction = np.array([value, 20.0, 45.0], dtype=np.float64)
    observed = engine._validate_assembled_layer4_prediction(
        prediction,
        reference,
        np.array([False, False, True]),
        clip_bounds=(-5.0, 45.0),
        np_module=np,
    )
    assert observed[0] == value


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -5.0000001, 45.0000001])
def test_nonfinite_or_out_of_range_modified_layer4_is_rejected(poison: float) -> None:
    reference = np.array([46.0, 20.0, 3.0], dtype=np.float64)
    prediction = np.array([46.0, 20.0, poison], dtype=np.float64)
    with pytest.raises(AssertionError, match="modified Layer-4"):
        engine._validate_assembled_layer4_prediction(
            prediction,
            reference,
            np.array([False, False, True]),
            clip_bounds=(-5.0, 45.0),
            np_module=np,
        )


def test_non_layer4_mutation_or_nonfinite_is_rejected() -> None:
    reference = np.array([46.0, 20.0, 3.0], dtype=np.float64)
    layer4 = np.array([False, False, True])
    for prediction in (
        np.array([45.0, 20.0, 3.0], dtype=np.float64),
        np.array([46.0, float("nan"), 3.0], dtype=np.float64),
    ):
        with pytest.raises(AssertionError, match="Stage-A Layer-2/3"):
            engine._validate_assembled_layer4_prediction(
                prediction,
                reference,
                layer4,
                clip_bounds=(-5.0, 45.0),
                np_module=np,
            )
