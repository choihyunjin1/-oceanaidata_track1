from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from p1_qc.pipeline import apply_postprocess
from p1_qc.rules import apply_hard_rules

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts/bootstrap_p1_multiscale_cross_layer_offset_drift_unary_v6r2.py"
CONFIG = ROOT / "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r2.json"
PROJECTION = ROOT / (
    "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r2_science_projection.json"
)
SCIENCE = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift_v6r2.py"
CONTRACT = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift_contract_v6r2.py"
ENGINE = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift_execution_v6r2.py"
RUNNER = ROOT / "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6r2.py"
TESTS = Path(__file__).resolve()

FROZEN_V6 = {
    "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6.json": (
        21120,
        "132dd95a91687d47212ff4af653a9b2d2b3264705dce0821aaa2be73c4933838",
    ),
    "src/p1_qc/multiscale_cross_layer_offset_drift.py": (
        37082,
        "434bce024437d5dea77a4827bfdde95f0726659f901993fd4c033df82ab605d8",
    ),
    "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6.py": (
        26293,
        "cbaf1ec413d449dff00a1ab5b3e93935ecaab21e41e6e24cb8db664b1b8d3a33",
    ),
    "tests/test_run_p1_multiscale_cross_layer_offset_drift_unary_v6.py": (
        14137,
        "323e06c2c588db1bea20f2d1d58849d12384e2d58b0f7d4192587508f63998b3",
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _load_science() -> ModuleType:
    context: dict[str, Any] = {
        "all_owner_roles_authenticated": True,
        "require_engine_capability": lambda capability, entry: (
            None
            if capability is _CAPABILITY and isinstance(entry, str)
            else (_ for _ in ()).throw(PermissionError("capability differs"))
        ),
        "verify_numerical_runtime": lambda: {"all_origins_record_authenticated": True},
    }
    module = ModuleType("_p1_v6r2_test_science")
    module.__file__ = str(SCIENCE)
    module.__dict__["_P1_V6R2_BOOTSTRAP_CONTEXT"] = context
    sys.modules[module.__name__] = module
    try:
        exec(compile(SCIENCE.read_text(encoding="utf-8"), str(SCIENCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    return module


_CAPABILITY = object()


def _state_snapshot() -> dict[str, Any]:
    config = _json(CONFIG)
    paths = [ROOT / relative for relative in config["static_expected_absence"]] + [
        ROOT / config["v9_binding"]["path"]
    ]
    return {
        str(path): None if not os.path.lexists(path) else (path.stat().st_size, _sha(path))
        for path in paths
    }


def _check_only() -> subprocess.CompletedProcess[str]:
    raw_data = os.environ.get("P1_DATA_DIR")
    if not raw_data or not Path(raw_data).is_absolute():
        raise RuntimeError("P1_DATA_DIR must be an absolute environment-only test input")
    env = os.environ.copy()
    env["P1_WORKSPACE_ROOT"] = str(ROOT)
    env["P1_DATA_DIR"] = raw_data
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    pycache = ROOT / "artifacts/p1_v6r2_forbidden_pycache"
    assert not os.path.lexists(pycache)
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={pycache}",
            str(BOOTSTRAP),
            "--check-only",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )


def test_frozen_v6_bytes_are_unchanged() -> None:
    for relative, (size, digest) in FROZEN_V6.items():
        path = ROOT / relative
        assert path.stat().st_size == size
        assert _sha(path) == digest


def test_v6_owner_no_go_and_tombstone_are_exact() -> None:
    config = _json(CONFIG)
    lineage = config["append_only_lineage"]
    owner = _json(ROOT / lineage["owner_no_go"]["path"])
    tombstone = _json(ROOT / lineage["execution_tombstone"]["path"])
    assert owner["verdict"] == "P0=0_P1=2_NO_GO"
    assert type(owner["p0_count"]) is int and owner["p0_count"] == 0
    assert type(owner["p1_count"]) is int and owner["p1_count"] == 2
    assert tombstone["status"] == "PERMANENTLY_TOMBSTONED_NEVER_EXECUTE"
    assert tombstone["owner_no_go_receipt"] == lineage["owner_no_go"]
    for role, pin in lineage["superseded_v6"].items():
        del role
        path = ROOT / pin["path"]
        assert (path.stat().st_size, _sha(path)) == (pin["bytes"], pin["sha256"])


def test_corrected_science_and_resource_contract_are_fixed() -> None:
    config = _json(CONFIG)
    projection = _json(PROJECTION)
    train_gate = projection["train_only_gate"]
    final_gate = projection["final_curve_gate"]
    assert train_gate["minimum_worst_station_layer_f1_delta"] == 0.0
    assert final_gate["worst_station_layer_f1_delta_at_least"] == 0.0
    assert train_gate["minimum_spike_f1_delta"] == 0.0
    assert train_gate["required_inner_blocks"] == 3
    assert train_gate["minimum_nondegrading_inner_blocks"] == 3
    assert train_gate["count_domain"] == "non_boolean_integral_json_number_exactly_3"
    assert projection["search_forbidden"] == {name: True for name in projection["search_forbidden"]}
    resource = config["resource_ceiling"]
    assert resource["maximum_label_free_baseline_fit_calls"] == 60
    assert resource["maximum_supervised_unary_fit_calls"] == 60
    assert resource["maximum_top_level_fit_calls"] == 120
    assert resource["maximum_total_iterative_steps"] == 11520
    assert resource["maximum_vram_bytes"] == 0
    assert resource["maximum_artifact_disk_bytes"] == 1_073_741_824


def test_strict_gates_reject_coercion_and_any_worst_group_regression() -> None:
    module = _load_science()
    passing = {
        "micro_f1_delta": 0.01,
        "offset_recall_delta": 0.05,
        "drift_recall_delta": 0.05,
        "spike_f1_delta": 0.0,
        "worst_station_layer_f1_delta": 0.0,
        "normal_fp_relative_increase": 0.0,
        "nondegrading_inner_blocks": 3,
        "inner_block_count": 3,
        "both_slow_types_observed": True,
        "spike_observed": True,
        "all_required_station_layers_observed": True,
        "blind_predictions_sealed_before_gate_labels": True,
    }
    assert module.strict_inner_gate(passing)["passed"] is True
    for value in (True, "3", 3.0, 3.9):
        probe = dict(passing, inner_block_count=value)
        assert module.strict_inner_gate(probe)["passed"] is False
    assert (
        module.strict_inner_gate(
            dict(passing, worst_station_layer_f1_delta=-np.nextafter(0.0, 1.0))
        )["passed"]
        is False
    )


def test_final_gate_rejects_coercible_bootstrap_count_and_late_regression() -> None:
    module = _load_science()
    points = [
        {
            "fraction": fraction,
            "micro_f1_delta": 0.03,
            "ci90": [0.01, 0.04],
            "offset_recall_delta": 0.05,
            "drift_recall_delta": 0.05,
            "spike_f1_delta": 0.0,
            "worst_station_layer_f1_delta": 0.0,
            "bootstrap_replicates": 5000,
            "offset_observed": True,
            "drift_observed": True,
            "spike_observed": True,
            "all_required_station_layers_observed": True,
        }
        for fraction in module.FRACTIONS
    ]
    passing = {
        "fraction_metrics": points,
        "fold_full_micro_f1_deltas": {fold: 0.03 for fold in module.FOLDS},
        "all_leakage_checks": True,
        "all_reproducibility_checks": True,
        "all_commitments_verified": True,
    }
    assert module.strict_final_curve_gate(passing)["passed"] is True
    coerced = {**passing, "fraction_metrics": [dict(item) for item in points]}
    coerced["fraction_metrics"][-1]["bootstrap_replicates"] = 5000.0
    assert module.strict_final_curve_gate(coerced)["passed"] is False
    regression = {**passing, "fraction_metrics": [dict(item) for item in points]}
    regression["fraction_metrics"][-1]["worst_station_layer_f1_delta"] = -1.0e-12
    assert module.strict_final_curve_gate(regression)["passed"] is False


def test_failed_gate_fallback_ignores_invalid_challenger_and_preserves_exact_bytes() -> None:
    module = _load_science()
    probability = np.ascontiguousarray(np.asarray([0.1, 0.9, 0.2], dtype=np.float32))
    prediction = np.ascontiguousarray(np.asarray([0, 1, 0], dtype=np.int8))
    output_probability, output_prediction, additions = module.protected_incumbent_union(
        capability=_CAPABILITY,
        incumbent_probability=probability,
        incumbent_prediction=prediction,
        gate_passed=False,
        slow_probability="intentionally invalid and never inspected",
        segment_ids="intentionally invalid and never inspected",
    )
    assert output_probability.dtype == probability.dtype
    assert output_prediction.dtype == prediction.dtype
    assert output_probability.tobytes(order="C") == probability.tobytes(order="C")
    assert output_prediction.tobytes(order="C") == prediction.tobytes(order="C")
    assert not additions.any()


def test_segment_proof_rejects_fabrication_and_target_carrying_frames() -> None:
    module = _load_science()
    frame = pd.DataFrame(
        {
            "station": ["A"] * 2200,
            "year": [2024] * 2200,
            "layer": np.ones(2200, dtype=np.int64),
            "time": pd.date_range("2024-01-01", periods=2200, freq="10min", tz="UTC").astype(str),
            "temp": np.linspace(1.0, 2.0, 2200),
            "psal": np.linspace(30.0, 31.0, 2200),
            "depth": np.ones(2200),
        }
    )
    train = np.arange(0, 500, dtype=np.int64)
    holdout = np.arange(1600, 1700, dtype=np.int64)
    exact = module.exact_gap_safe_segment_ids(frame)
    assert (
        module.verify_dependency_closed_split(
            capability=_CAPABILITY,
            frame=frame,
            train_ids=train,
            holdout_ids=holdout,
            segment_ids=exact,
        )["passed"]
        is True
    )
    with pytest.raises(module.ScienceContractError):
        module.verify_dependency_closed_split(
            capability=_CAPABILITY,
            frame=frame,
            train_ids=train,
            holdout_ids=holdout,
            segment_ids=np.arange(len(frame), dtype=np.int64),
        )
    carrying = frame.assign(label=0)
    with pytest.raises(module.ScienceContractError):
        module.verify_dependency_closed_split(
            capability=_CAPABILITY,
            frame=carrying,
            train_ids=train,
            holdout_ids=holdout,
            segment_ids=exact,
        )


def test_fixed_incumbent_postprocess_matches_frozen_reference() -> None:
    module = _load_science()
    rows = 96
    values = np.linspace(10.0, 11.0, rows)
    values[12:20] = 10.25
    values[35] += 3.0
    values[60:75] = 10.8
    frame = pd.DataFrame(
        {
            "station": ["G-ORS"] * rows,
            "year": [2024] * rows,
            "layer": np.ones(rows, dtype=np.int64),
            "time": pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC").astype(str),
            "temp": values,
            "psal": np.full(rows, 34.0),
            "depth": np.full(rows, 10.0),
        }
    )
    probability = np.linspace(0.01, 0.25, rows, dtype=np.float32)
    projection = _json(PROJECTION)["inner_incumbent"]
    golden = projection["golden_fixture"]
    assert (
        hashlib.sha256(
            values.astype("<f8", copy=False).tobytes(order="C")
            + probability.astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest()
        == golden["sha256"]
    )
    hard = apply_hard_rules(frame)
    parameters = {
        "2025_q2": {
            "high_threshold": 0.15,
            "low_threshold": 0.075,
            "close_gap_rows": 0,
            "minimum_positive_run": 12,
        },
        "2025_q3": {
            "high_threshold": 0.2,
            "low_threshold": 0.1,
            "close_gap_rows": 0,
            "minimum_positive_run": 12,
        },
        "2025_q4": {
            "high_threshold": 0.15,
            "low_threshold": 0.075,
            "close_gap_rows": 6,
            "minimum_positive_run": 6,
        },
    }
    for fold, fixed in parameters.items():
        expected = apply_postprocess(
            frame,
            probability,
            hard.plateau.to_numpy(bool),
            hard.singleton_spike.to_numpy(bool),
            fixed,
        ).astype(np.int8)
        observed = module.fixed_incumbent_postprocess(
            capability=_CAPABILITY, frame=frame, probabilities=probability, fold=fold
        )
        assert observed.dtype == np.int8
        assert observed.tobytes(order="C") == expected.tobytes(order="C")
        assert (
            hashlib.sha256(observed.tobytes(order="C")).hexdigest()
            == golden["output_int8_sha256_by_fold"][fold]
        )


def test_inner_split_matches_frozen_implementation() -> None:
    from p1_qc.incumbent_residual_tcn import build_three_block_inner_splits as frozen_split

    module = _load_science()
    frame = pd.DataFrame(
        {"time": pd.date_range("2023-01-01", periods=5000, freq="10min", tz="UTC").astype(str)}
    )
    ids = np.arange(len(frame), dtype=np.int64)
    expected = frozen_split(frame, ids, purge_days=7)
    observed = module.build_three_block_inner_splits(
        capability=_CAPABILITY, metadata=frame, outer_prefix_ids=ids
    )
    assert [item.as_audit() for item in observed] == [
        {
            "block": item.block,
            "train_rows": len(item.teacher_train_ids),
            "prediction_rows": len(item.teacher_prediction_ids),
            "train_ids_sha256": item.train_ids_sha256,
            "prediction_ids_sha256": item.prediction_ids_sha256,
            "train_end_utc": item.train_end_utc,
            "prediction_start_utc": item.prediction_start_utc,
            "prediction_end_utc": item.prediction_end_utc,
            "purge_days": item.purge_days,
        }
        for item in expected
    ]


@pytest.mark.parametrize("path", [SCIENCE, CONTRACT, ENGINE, RUNNER])
def test_direct_owner_module_load_fails(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"_direct_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="authenticated bootstrap"):
        spec.loader.exec_module(module)


def test_all_owner_roles_are_ast_valid_and_have_no_top_level_numerical_imports() -> None:
    for path in (SCIENCE, CONTRACT, ENGINE, RUNNER):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                assert not any(
                    name.split(".")[0] in {"numpy", "pandas", "sklearn", "torch"} for name in names
                )


def test_bootstrap_pin_map_matches_every_subordinate_role() -> None:
    tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "PINNED_OWNER_ROLES"
    )
    pin_map = ast.literal_eval(assignment.value)
    expected_paths = {
        "CONFIG": CONFIG,
        "SCIENCE_PROJECTION": PROJECTION,
        "SCIENCE": SCIENCE,
        "CONTRACT": CONTRACT,
        "ENGINE": ENGINE,
        "RUNNER": RUNNER,
        "TESTS": TESTS,
    }
    for role, path in expected_paths.items():
        pin = pin_map[role]
        assert ROOT / pin["path"] == path
        assert pin["bytes"] == path.stat().st_size
        assert pin["sha256"] == _sha(path)


def test_bootstrap_rejects_noncanonical_python_flags_before_path_imports() -> None:
    env = os.environ.copy()
    env["P1_WORKSPACE_ROOT"] = str(ROOT)
    raw_data = env.get("P1_DATA_DIR")
    if not raw_data or not Path(raw_data).is_absolute():
        raise RuntimeError("P1_DATA_DIR must be an absolute environment-only test input")
    completed = subprocess.run(
        [sys.executable, "-B", str(BOOTSTRAP), "--check-only"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode != 0
    assert "requires python -I -S -B" in completed.stderr


def test_bootstrap_declares_same_buffer_json_and_authenticated_source_loading() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "sys.addaudithook(_earliest_audit_hook)" in text
    assert text.index("sys.addaudithook(_earliest_audit_hook)") < text.index("import hashlib")
    assert "_AuthenticatedSourceLoader" in text
    assert "return _RUNTIME_FILE_BYTES[normalized]" in text
    assert "bytecode loading is forbidden" in text
    assert "parse_json_text(qa_raw.decode" in text
    assert "parse_json_text(auth_raw.decode" in text
    assert "python -I -S -B" in text


def test_authenticated_numerical_runtime_import_smoke_is_read_only() -> None:
    raw_data = os.environ.get("P1_DATA_DIR")
    if not raw_data or not Path(raw_data).is_absolute():
        raise RuntimeError("P1_DATA_DIR must be an absolute environment-only test input")
    pycache = ROOT / "artifacts/p1_v6r2_forbidden_pycache"
    assert not os.path.lexists(pycache)
    before = _state_snapshot()
    probe = "\n".join(
        (
            "import json, pathlib, sys",
            f"bootstrap = pathlib.Path({str(BOOTSTRAP)!r})",
            "sys.argv = [str(bootstrap), '--check-only']",
            "scope = {'__name__': '_p1_v6r2_runtime_smoke', '__file__': str(bootstrap)}",
            "exec(compile(bootstrap.read_bytes(), str(bootstrap), 'exec'), scope)",
            f"audit = scope['_authenticate_numerical_runtime'](pathlib.Path({str(ROOT)!r}))",
            "import joblib, narwhals, numpy, pandas, psutil, pyarrow, scipy, six, sklearn.linear_model, threadpoolctl",
            "origins = scope['_verify_loaded_numerical_origins']()",
            "print(json.dumps({'audit': audit, 'origins': origins}, sort_keys=True))",
        )
    )
    env = os.environ.copy()
    env["P1_WORKSPACE_ROOT"] = str(ROOT)
    env["P1_DATA_DIR"] = raw_data
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={pycache}",
            "-c",
            probe,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["audit"]["verified_unique_files"] > 0
    assert report["audit"]["held_native_files"] > 0
    assert report["origins"]["loaded_modules"] > 0
    assert report["origins"]["all_origins_record_authenticated"] is True
    assert _state_snapshot() == before
    assert not os.path.lexists(pycache)


def test_contract_declares_exact_192_file_fold_major_output_sequence() -> None:
    context = {
        "all_owner_roles_authenticated": True,
        "config": _json(CONFIG),
        "science_projection": _json(PROJECTION),
    }
    module = ModuleType("_p1_v6r2_test_contract")
    module.__file__ = str(CONTRACT)
    module.__dict__["_P1_V6R2_BOOTSTRAP_CONTEXT"] = context
    sys.modules[module.__name__] = module
    try:
        exec(compile(CONTRACT.read_text(encoding="utf-8"), str(CONTRACT), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    sequence = module._expected_output_sequence()
    assert len(sequence) == 192
    assert sequence[:9] == [
        "models/inner_01.json",
        "inner_predictions/inner_01.bin",
        "blind_commitments/inner_01.json",
        "models/inner_02.json",
        "inner_predictions/inner_02.bin",
        "blind_commitments/inner_02.json",
        "models/inner_03.json",
        "inner_predictions/inner_03.bin",
        "blind_commitments/inner_03.json",
    ]
    assert sequence[-2:] == ["manifest.json", "manifest.sha256"]
    counters = module._exact_completion_counters()
    assert counters["scores"] == 45 + 5 + 3 == 53
    assert counters["predictions"] == 45 + 15 == 60
    assert counters["inner_commitments"] == 45
    assert counters["cell_commitments"] == 15
    assert counters["fold_commitments"] == 3
    assert counters["predictions_complete"] == 1
    assert counters["files_written"] == len(sequence) == 192
    assert (
        module._validate_completion_counters(dict(counters), expected_file_count=len(sequence))
        == counters
    )
    wrong = dict(counters, scores=60)
    with pytest.raises(module.ContractError, match="resource/commitment"):
        module._validate_completion_counters(wrong, expected_file_count=len(sequence))


def test_canonical_check_only_is_read_only_and_engine_free() -> None:
    before = _state_snapshot()
    completed = _check_only()
    after = _state_snapshot()
    assert completed.returncode == 0, completed.stderr
    assert before == after
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    summary = report.pop("summary_sha256")
    assert hashlib.sha256(_canonical(report)).hexdigest() == summary
    assert report["status"] == "P1_MULTISCALE_CROSS_LAYER_OFFSET_DRIFT_V6R2_STATIC_CHECK_PASS"
    assert report["engine_loaded"] is False
    assert report["numerical_modules_loaded_by_check_only"] == []
    assert report["operation_counts"] == {
        "independent_qa_receipts_created": 0,
        "execution_authorizations_created": 0,
        "attempt_locks_created": 0,
        "fits": 0,
        "predictions": 0,
        "target_decodes": 0,
        "scores": 0,
        "outputs": 0,
        "test_value_reads": 0,
        "candidates": 0,
        "ledger_appends": 0,
        "uploads": 0,
    }


def test_execute_without_external_trust_documents_fails_before_state() -> None:
    before = _state_snapshot()
    env = os.environ.copy()
    env["P1_WORKSPACE_ROOT"] = str(ROOT)
    raw_data = env.get("P1_DATA_DIR")
    if not raw_data or not Path(raw_data).is_absolute():
        raise RuntimeError("P1_DATA_DIR must be an absolute environment-only test input")
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={ROOT / 'artifacts/p1_v6r2_forbidden_pycache'}",
            str(BOOTSTRAP),
            "--execute",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode != 0
    assert _state_snapshot() == before
    assert not os.path.lexists(ROOT / _json(CONFIG)["canonical_paths"]["attempt_lock"])


def test_static_surface_contains_full_future_execution_chain() -> None:
    science_text = SCIENCE.read_text(encoding="utf-8")
    contract_text = CONTRACT.read_text(encoding="utf-8")
    engine_text = ENGINE.read_text(encoding="utf-8")
    required = (
        "SelectiveTargetAccessorV6R2",
        "BlindCommitmentLedger",
        "commit_inner",
        "commit_cell",
        "commit_fold",
        "predictions_complete",
        "build_three_block_inner_splits",
        "verify_dependency_closed_split",
        "fit_robust_seasonal_graph_state",
        "fit_fixed_slow_unary_head",
        "saved model reload inference is not byte-exact",
        "paired_bootstrap_f1_delta_ci90",
        "strict_final_curve_gate",
        'candidate_created": False',
        'test_prediction_created": False',
        'ledger_appended": False',
        'uploaded": False',
    )
    combined = science_text + contract_text + engine_text
    assert all(item in combined for item in required)


def test_v9_and_all_future_paths_remain_frozen_or_absent() -> None:
    config = _json(CONFIG)
    v9 = ROOT / config["v9_binding"]["path"]
    assert (v9.stat().st_size, _sha(v9)) == (
        config["v9_binding"]["bytes"],
        config["v9_binding"]["sha256"],
    )
    assert config["v9_binding"]["head_seq"] == 5
    assert config["v9_binding"]["semantic_upload_count"] == 0
    assert all(
        not os.path.lexists(ROOT / relative) for relative in config["static_expected_absence"]
    )
