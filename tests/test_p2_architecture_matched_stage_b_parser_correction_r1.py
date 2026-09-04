from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_restore import architecture_matched_stage_b_contract_r1 as guard
from p2_restore import architecture_matched_stage_b_csv_r1 as parser_boundary
from p2_restore import architecture_matched_stage_b_execution_r1 as engine
from p2_restore import architecture_matched_stage_b_execution_v3 as base_engine


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((_root() / guard.CONFIG_RELATIVE).read_text(encoding="utf-8"))


def _write_observations(path: Path, rows: list[str]) -> None:
    path.write_text(
        "station,year,layer,time,temp,psal,depth,nominal_depth\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def test_config_and_failure_ancestry_are_exact() -> None:
    root = _root()
    config_path = root / guard.CONFIG_RELATIVE
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == guard.CONFIG_SHA256
    config = _config()
    failure = config["base_v3"]["failure_receipt"]
    failure_path = root / failure["path"]
    assert failure_path.stat().st_size == failure["bytes"]
    assert hashlib.sha256(failure_path.read_bytes()).hexdigest() == failure["sha256"]
    receipt = json.loads(failure_path.read_text(encoding="utf-8"))
    assert receipt["classification"] == "INFRASTRUCTURE_FAILURE_NO_EVALUATION"
    assert receipt["execution_audit"]["challenger_fit_count"] == 0
    assert receipt["execution_audit"]["challenger_prediction_count"] == 0
    assert receipt["field_access_audit"]["withheld_validation_target_scalar_decode_count"] == 0
    addendum = config["base_v3"]["failure_forensic_addendum"]
    addendum_path = root / addendum["path"]
    assert addendum_path.stat().st_size == addendum["bytes"]
    assert hashlib.sha256(addendum_path.read_bytes()).hexdigest() == addendum["sha256"]
    addendum_receipt = json.loads(addendum_path.read_text(encoding="utf-8"))
    assert (
        addendum_receipt["classification"]
        == "SECONDARY_PREEXECUTION_INPUT_ROUTING_DEFECT_NO_EVALUATION"
    )
    assert addendum_receipt["discovery"]["actual_v3_attempt_reached_this_defect"] is False


def test_scientific_projection_is_deep_equal_to_base_v3() -> None:
    root = _root()
    correction = _config()
    base = json.loads(
        (root / correction["base_v3"]["config"]["path"]).read_text(encoding="utf-8")
    )
    sections = correction["scientific_contract"]["base_v3_sections"]
    projected = {key: base[key] for key in sections}
    observed = hashlib.sha256(guard.canonical_json_bytes(projected)).hexdigest()
    assert observed == correction["scientific_contract"]["canonical_sha256"]
    assert correction["scientific_contract"][
        "model_feature_fold_seed_prefix_bootstrap_postprocess_gate_changes"
    ] == 0


def test_decoder_accepts_unquoted_and_quoted_empty_scalars() -> None:
    line, spans = parser_boundary.csv_field_spans(b"a,,\"\",\"x,y\",z\n", expected_fields=5)
    decoded = [parser_boundary.decode_csv_field(line, span) for span in spans]
    assert decoded == ["a", "", "", "x,y", "z"]


def test_fold_loader_routes_all_public_layers_and_never_decodes_withheld_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observations = tmp_path / "observations.csv"
    rows = [
        "S,2024,1,2024-09-01T00:00:00+09:00,11.0,,1.0,1.0",
        "S,2024,5,2024-09-01T00:00:00+09:00,15.0,35.0,5.0,5.0",
        "S,2024,6,2024-09-01T00:00:00+09:00,16.0,,6.0,6.0",
        "S,2024,7,2024-09-01T00:00:00+09:00,17.0,37.0,7.0,7.0",
        "S,2024,8,2024-09-01T00:00:00+09:00,18.0,38.0,,8.0",
        "S,2024,2,2024-08-24T23:50:00+09:00,12.0,32.0,2.0,2.0",
        "S,2024,2,2024-08-25T00:00:00+09:00,TRUTH_POISON,TRUTH_POISON,2.0,2.0",
        "S,2024,3,2024-09-01T00:00:00+09:00,TRUTH_POISON,TRUTH_POISON,3.0,3.0",
        "S,2024,4,2024-09-01T00:00:00+09:00,TRUTH_POISON,TRUTH_POISON,4.0,4.0",
    ]
    _write_observations(observations, rows)
    real_decode = engine.decode_csv_field
    decoded_tokens: list[bytes] = []

    def poison_guard(raw_line: bytes, span: tuple[int, int]) -> str:
        selected = raw_line[slice(*span)]
        assert b"TRUTH_POISON" not in selected
        decoded_tokens.append(selected)
        return real_decode(raw_line, span)

    monkeypatch.setattr(engine, "decode_csv_field", poison_guard)
    frame, audit = engine._load_fold_blind_observations(
        observations,
        outer_start=pd.Timestamp("2024-09-01T00:00:00+09:00").tz_convert("UTC"),
        embargo_days=7,
        pd_module=pd,
        np_module=np,
    )
    public = frame.loc[frame["layer"].isin(parser_boundary.PUBLIC_LAYERS)]
    withheld = frame.loc[
        frame["layer"].isin(parser_boundary.TARGET_LAYERS)
        & pd.to_datetime(frame["time"], utc=True).ge(pd.Timestamp("2024-08-24T15:00:00Z"))
    ]
    assert set(public["layer"]) == set(parser_boundary.PUBLIC_LAYERS)
    assert np.isnan(public.loc[public["layer"].eq(1), "psal"]).all()
    assert withheld[["temp", "psal"]].isna().all().all()
    assert audit["allowed_training_target_rows"] == 1
    assert audit["withheld_target_rows"] == 3
    assert audit["withheld_target_scalar_fields_decoded_or_converted"] == 0
    assert audit["public_layers_loaded_at_all_times"] == [1, 5, 6, 7, 8]
    assert not any(b"TRUTH_POISON" in token for token in decoded_tokens)


def test_postcommit_truth_loader_accepts_missing_scalars_and_skips_hidden_poison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observations = tmp_path / "observations.csv"
    _write_observations(
        observations,
        [
            "S,2024,2,2024-09-01T00:00:00+09:00,12.0,,2.0,2.0",
            "S,2024,3,2024-09-01T00:10:00+09:00,13.0,33.0,3.0,3.0",
            "S,2025,2,2025-09-01T00:00:00+09:00,HIDDEN_POISON,HIDDEN_POISON,2.0,2.0",
        ],
    )
    config = guard.load_canonical_config(_root())
    decoded_tokens: list[bytes] = []
    real_decode = parser_boundary.decode_csv_field

    def track(raw_line: bytes, span: tuple[int, int]) -> str:
        selected = raw_line[slice(*span)]
        decoded_tokens.append(selected)
        return real_decode(raw_line, span)

    monkeypatch.setattr(engine, "decode_csv_field", track)
    with engine._bound_base_engine():
        truth, audit = base_engine._load_validation_truth_after_commitment(
            observations,
            config=config,
            pd_module=pd,
            np_module=np,
        )
    assert truth[["layer", "truth"]].to_dict("records") == [{"layer": 3, "truth": 13.0}]
    assert audit["validation_target_scalars_converted_after_commitment"] == 4
    assert audit["hidden_test_target_scalars_converted"] == 0
    assert not any(b"HIDDEN_POISON" in token for token in decoded_tokens)


def test_parser_boundary_imports_no_numerical_or_model_packages() -> None:
    source = (_root() / "src/p2_restore/architecture_matched_stage_b_csv_r1.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(
        {"numpy", "pandas", "scipy", "sklearn", "lightgbm", "torch", "p2_restore"}
    )


def test_canonical_root_rejects_a_different_workspace(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="canonical module workspace"):
        guard.load_canonical_config(tmp_path)


def test_new_one_shot_paths_are_distinct_from_consumed_v3() -> None:
    config = guard.load_canonical_config(_root())
    correction = _config()
    assert config["canonical_paths"] == correction["canonical_paths"]
    assert config["canonical_paths"]["output"] != correction["base_v3"]["empty_output"]["path"]
    assert config["execution_policy"]["rerun_allowed"] is False
    assert config["execution_policy"]["resume_allowed"] is False
    assert config["execution_policy"]["candidate_or_test_prediction_allowed"] is False


def test_engine_adapter_rebinds_only_control_and_parser_boundaries() -> None:
    source = inspect.getsource(engine._bound_base_engine)
    expected = {
        "CONFIG_RELATIVE",
        "CONFIG_SHA256",
        "FRACTION_ROLES",
        "MODE",
        "TARGET_LAYERS",
        "_csv_field_spans",
        "_decode_csv_field",
        "_load_fold_blind_observations",
        "implementation_pins",
        "load_canonical_config",
        "stage_paths",
        "static_preflight",
        "verify_consumed_attempt_lock",
        "verify_execution_authorization",
        "verify_pre_execution_qa",
        "verify_stage_a_reference",
    }
    tree = ast.parse(source)
    literal_keys = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert literal_keys == expected


def test_runner_orders_parser_preflight_qa_lock_and_late_engine_import() -> None:
    from scripts import run_p2_architecture_matched_stage_b_parser_correction_r1 as runner

    body = inspect.getsource(runner.run_once)
    positions = [
        body.index("preflight = guard.static_preflight("),
        body.index("guard.verify_pre_execution_qa("),
        body.index("guard.verify_execution_authorization("),
        body.index("lock = guard.consume_attempt_lock("),
        body.index("engine = importlib.import_module(ENGINE_MODULE)"),
        body.index("result = execute("),
    ]
    assert positions == sorted(positions)
    assert "guard.write_run_failure_receipt" in body


def test_check_only_cannot_create_qa_authorization_lock_output_or_upload() -> None:
    from scripts import run_p2_architecture_matched_stage_b_parser_correction_r1 as runner

    body = inspect.getsource(runner.check_only)
    forbidden = (
        "exclusive_json",
        "consume_attempt_lock",
        "execute_stage_b",
        "upload(",
    )
    assert all(token not in body for token in forbidden)
    assert '"fits": 0' in body
    assert '"predictions": 0' in body
    assert '"scores": 0' in body
    assert '"uploads": 0' in body


def test_implementation_roles_and_paths_have_no_personal_absolute_path() -> None:
    correction = _config()
    assert set(correction["implementation_roles"]) == guard.IMPLEMENTATION_ROLES
    for relative in (
        *correction["implementation_roles"].values(),
        *correction["canonical_paths"].values(),
    ):
        path = Path(relative)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert "cedis" not in relative.casefold()
