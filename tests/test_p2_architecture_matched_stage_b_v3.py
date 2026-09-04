from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from p2_restore import architecture_matched_stage_b_contract_v3 as guard


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _engine() -> object:
    return importlib.import_module("p2_restore.architecture_matched_stage_b_execution_v3")


def test_canonical_config_sha_copy_rejection_and_structural_identity(tmp_path: Path) -> None:
    root = _root()
    config = guard.load_canonical_config(root)
    assert guard.sha256_file(root / guard.CONFIG_RELATIVE) == guard.CONFIG_SHA256
    assert config["problem"] == "P2"
    assert config["comparison_mode"] == guard.MODE
    assert config["hypothesis"]["id"] == ("H1_SEEDED_CONDITIONAL_ANALOG_RANK2_PROFILE_MANIFOLD")
    assert config["hypothesis"]["not_a_blend_weight_or_alpha_tweak"] is True
    assert config["hypothesis"]["catboost_used"] is False
    assert config["hypothesis"]["failed_generation_v1_reused"] is False
    assert config["hypothesis"]["hyperparameter_searches"] == 0
    copied = tmp_path / "copied_config.json"
    copied.write_bytes((root / guard.CONFIG_RELATIVE).read_bytes())
    with pytest.raises(guard.StageBContractError, match="canonical Stage-B v3 config path"):
        guard.load_canonical_config(root, copied)


@pytest.mark.parametrize("problem", ["P1", "P3"])
def test_architecture_matched_stage_b_is_rejected_for_p1_and_p3(problem: str) -> None:
    config = deepcopy(guard.load_canonical_config(_root()))
    config["problem"] = problem
    with pytest.raises(guard.StageBContractError, match="P2-only"):
        guard.validate_config(config)


def test_non_exact_research_only_and_output_firewall_fail_closed() -> None:
    root = _root()
    for key, value in (
        ("exact_official_incumbent_comparison", True),
        ("official_promotion_allowed", True),
        ("upload_allowed", True),
        ("explicitly_not_exact_official_incumbent", False),
        ("local_qualification_only", False),
        ("research_only", False),
    ):
        config = deepcopy(guard.load_canonical_config(root))
        config[key] = value
        with pytest.raises(guard.StageBContractError):
            guard.validate_config(config)
    config = deepcopy(guard.load_canonical_config(root))
    config["output_contract"]["candidate_generated"] = True
    with pytest.raises(guard.StageBContractError, match="output firewall"):
        guard.validate_config(config)


def test_exact_implementation_roles_and_all_source_pins_are_current() -> None:
    root = _root()
    config = guard.load_canonical_config(root)
    assert list(config["implementation_roles"]) == [
        "CONFIG",
        "GUARD",
        "ENGINE",
        "RUNNER",
        "TESTS",
    ]
    assert set(guard.implementation_pins(root)) == set(guard.IMPLEMENTATION_ROLES)
    assert set(config["source_pins"]) == {
        "STAGE_A_V3_GUARD",
        "STAGE_A_V3_ENGINE",
        "STAGE_A_V2_ENGINE",
        "DEEP_DATA",
        "PROFILE_PROJECTION",
        "FINAL_INFERENCE",
        "DATA",
        "CURVE_GATE",
        "PACKAGE_INIT",
        "MODEL_MODULE",
        "CENTRAL_V3_EVALUATOR",
        "CENTRAL_V3_CONTRACT",
    }
    assert (
        guard.verify_pin_map(root, config["source_pins"], label="source") == config["source_pins"]
    )


def test_stage_a_v3_reference_is_exactly_pinned_and_deeply_sealed() -> None:
    root = _root()
    config = guard.load_canonical_config(root)
    reference = guard.verify_stage_a_reference(root, config)
    assert reference["status"] == "PASS_EXACT_STAGE_A_V3_REFERENCE_BINDING"
    assert reference["full_reference_rmse_c"] == 1.0109798870010898
    assert reference["cells"] == 45
    assert reference["deep_training_jobs"] == 720
    assert reference["router_training_jobs"] == 180
    assert reference["challenger_fit_or_score_count_before_seal"] == 0
    assert reference["uploads"] == 0
    assert reference["pins"] == config["stage_a_reference"]["artifacts"]
    observed_folds = pd.read_csv(
        root / config["stage_a_reference"]["artifacts"]["OOF_100"]["path"],
        usecols=["fold"],
    )["fold"].drop_duplicates()
    expected_folds = [fold["name"] for fold in config["curve_protocol"]["outer_folds"]]
    assert set(observed_folds) == set(expected_folds)


def test_source_or_stage_a_drift_fails_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _root()
    config = guard.load_canonical_config(root)
    real_sha = guard.sha256_file
    target = (root / config["source_pins"]["MODEL_MODULE"]["path"]).resolve()

    def drift(path: Path) -> str:
        if path.resolve() == target:
            return "0" * 64
        return real_sha(path)

    monkeypatch.setattr(guard, "sha256_file", drift)
    with pytest.raises(guard.StageBContractError, match="MODEL_MODULE"):
        guard.static_preflight(root, tmp_path)


def test_output_containment_and_o_excl_rerun_guard(tmp_path: Path) -> None:
    output = tmp_path / "stage_b"
    assert guard.contained_path(output, "seal.json") == (output / "seal.json").resolve()
    with pytest.raises(guard.StageBContractError):
        guard.contained_path(output, "../escape.json")
    path = tmp_path / "control" / "attempt.lock"
    guard.exclusive_bytes(path, b"first")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        guard.exclusive_bytes(path, b"second")
    assert path.read_bytes() == original


def test_post_run_seal_verifier_rejects_unregistered_or_nonaggregate_outputs() -> None:
    source = (_root() / guard.IMPLEMENTATION_ROLES["GUARD"]).read_text(encoding="utf-8")
    body = source[source.index("def verify_stage_b_seal(") :]
    assert "observed_children != expected_children" in body
    assert '"manifest_artifact_roles"' in body
    assert '"commitment_before_truth"' in body
    assert '"commitment_exact_keys"' in body
    assert '"commitment_combined_digest"' in body
    assert '"commitment_key_order"' in body
    assert '"evidence_commitment"' in body
    assert '"evidence_leakage_checks"' in body
    assert '"seal_commitment"' in body
    assert '"evidence_firewall"' in body
    assert '"decision_never_promotes"' in body
    assert '"receipt_no_full_fit"' in body
    assert '"receipt_no_candidate"' in body
    assert '"receipt_no_test"' in body
    assert '"receipt_no_upload"' in body
    assert '"receipt_no_withheld_decode"' in body


def test_execution_plan_is_complete_not_scaffolding_and_never_builds_candidate() -> None:
    config = guard.load_canonical_config(_root())
    engine = _engine()
    plan = engine.build_execution_plan(config)
    assert callable(engine.execute_stage_b)
    assert plan["outer_prefix_cells"] == 15
    assert plan["challenger_fits"] == 45
    assert plan["challenger_outer_predictions"] == 45
    assert plan["complete_pipeline_seeds"] == [20260823, 20260824, 20260825]
    assert plan["paired_kst_day_bootstrap_replicates_per_fraction"] == 5000
    assert plan["full_fit_jobs"] == 0
    assert plan["test_predictions"] == 0
    assert plan["candidate_files"] == 0
    assert plan["uploads"] == 0


def test_rank2_projection_geometry_and_seeded_cpu_model_are_deterministic() -> None:
    engine = _engine()
    neighbors = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]]])
    weights = torch.full((1, 4), 0.25)
    anchor = torch.tensor([[0.25, 0.5, 4.0]])
    projected = engine._weighted_rank2_projection(
        neighbors,
        weights,
        anchor,
        torch_module=torch,
    )
    assert abs(float(projected[0, 0] + projected[0, 1] - projected[0, 2])) < 1e-5
    on_plane = torch.tensor([[0.25, 0.5, 0.75]])
    unchanged = engine._weighted_rank2_projection(
        neighbors,
        weights,
        on_plane,
        torch_module=torch,
    )
    assert torch.allclose(unchanged, on_plane, atol=1e-5, rtol=0.0)

    rng = np.random.default_rng(11)
    train_x = rng.normal(size=(40, 6))
    train_y = np.column_stack((train_x[:, 0], train_x[:, 1], train_x[:, 0] + train_x[:, 1]))
    query_x = rng.normal(size=(5, 6))
    anchors = rng.normal(size=(5, 3))
    kwargs = {
        "seed": 123,
        "projection_dimensions": 4,
        "nearest_neighbors": 12,
        "batch_size": 3,
        "standardized_clip": 12.0,
        "np_module": np,
        "torch_module": torch,
        "device": torch.device("cpu"),
    }
    first, first_receipt = engine._fit_predict_rank2(
        train_x,
        train_y,
        query_x,
        anchors,
        **kwargs,
    )
    second, second_receipt = engine._fit_predict_rank2(
        train_x,
        train_y,
        query_x,
        anchors,
        **kwargs,
    )
    assert np.array_equal(first, second)
    assert first_receipt == second_receipt
    assert first_receipt["local_manifold_rank"] == 2
    assert first_receipt["nearest_neighbors"] == 12


def test_paired_kst_day_bootstrap_is_deterministic_exactly_5000_and_negative() -> None:
    engine = _engine()
    rows: list[dict[str, object]] = []
    for fold in ("f1", "f2", "f3"):
        for day in ("2024-09-01", "2024-09-02"):
            for layer in (2, 3, 4):
                rows.append(
                    {
                        "fold": fold,
                        "layer": layer,
                        "time": f"{day}T00:00:00+09:00",
                        "truth": 0.0,
                        "reference": 1.0,
                        "challenger": 0.5,
                    }
                )
    frame = pd.DataFrame(rows)
    kwargs = {
        "reference_column": "reference",
        "challenger_column": "challenger",
        "counts": {"2": 8713, "3": 8712, "4": 8636},
        "replicates": 5000,
        "seed": 20260823,
        "pd_module": pd,
        "np_module": np,
    }
    first, receipt = engine._paired_kst_day_bootstrap(frame, **kwargs)
    second, _ = engine._paired_kst_day_bootstrap(frame, **kwargs)
    assert first == second
    assert first[1] < 0.0
    assert receipt["replicates"] == 5000
    assert receipt["cluster"] == "KST_day"
    assert receipt["paired_reference_and_challenger"] is True


def test_fold_blind_loader_never_decodes_current_validation_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = _engine()
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "\n".join(
            (
                "station,year,layer,time,temp,psal,depth,nominal_depth",
                "S,2024,1,2024-09-01 00:00:00+09:00,11.0,31.0,0.0,0.0",
                "S,2024,2,2024-08-24 23:59:59+09:00,12.0,32.0,10.0,10.0",
                "S,2024,2,2024-08-25 00:00:00+09:00,TRUTH_POISON,TRUTH_POISON,10.0,10.0",
                "S,2024,3,2024-09-01T00:00:00+09:00,TRUTH_POISON,TRUTH_POISON,20.0,20.0",
                "S,2024,5,2024-09-01T00:00:00+09:00,15.0,35.0,40.0,40.0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    real_decode = engine._decode_csv_field
    decoded_fields: list[bytes] = []

    def poison_guard(raw_line: bytes, span: tuple[int, int]) -> str:
        selected = raw_line[slice(*span)]
        assert b"TRUTH_POISON" not in selected
        decoded_fields.append(selected)
        return real_decode(raw_line, span)

    monkeypatch.setattr(engine, "_decode_csv_field", poison_guard)
    frame, audit = engine._load_fold_blind_observations(
        observations,
        outer_start=pd.Timestamp("2024-09-01T00:00:00+09:00").tz_convert("UTC"),
        embargo_days=7,
        pd_module=pd,
        np_module=np,
    )
    public = frame[frame["layer"].isin((1, 5))]
    allowed = frame[(frame["layer"] == 2) & frame["time"].str.startswith("2024-08-24")]
    withheld = frame[frame["layer"].isin((2, 3)) & ~frame.index.isin(allowed.index)]
    assert public[["temp", "psal"]].notna().all().all()
    assert allowed[["temp", "psal"]].to_numpy(float).tolist() == [[12.0, 32.0]]
    assert withheld[["temp", "psal"]].isna().all().all()
    assert audit["allowed_training_target_rows"] == 1
    assert audit["withheld_target_rows"] == 2
    assert audit["withheld_target_scalar_fields_decoded_or_converted"] == 0
    assert audit["validation_target_temp_psal_strings_converted"] == 0
    assert audit["validation_truth_columns_read_by_challenger"] == 0
    assert not any(b"TRUTH_POISON" in value for value in decoded_fields)


def test_truth_loader_runs_after_commitment_and_skips_hidden_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = _engine()
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "\n".join(
            (
                "station,year,layer,time,temp,psal,depth,nominal_depth",
                "S,2024,2,2024-09-01 00:00:00+09:00,12.0,32.0,10.0,10.0",
                "S,2025,3,2025-05-01T00:00:00+09:00,13.0,33.0,20.0,20.0",
                "S,2025,2,2025-09-01T00:00:00+09:00,HIDDEN_POISON,HIDDEN_POISON,10.0,10.0",
                "S,2024,1,2024-09-01T00:00:00+09:00,PUBLIC_POISON,PUBLIC_POISON,0.0,0.0",
                "S,2024,4,2024-01-01T00:00:00+09:00,OUTSIDE_POISON,OUTSIDE_POISON,30.0,30.0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    real_decode = engine._decode_csv_field
    decoded_fields: list[bytes] = []

    def track_decode(raw_line: bytes, span: tuple[int, int]) -> str:
        selected = raw_line[slice(*span)]
        decoded_fields.append(selected)
        return real_decode(raw_line, span)

    monkeypatch.setattr(engine, "_decode_csv_field", track_decode)
    truth, audit = engine._load_validation_truth_after_commitment(
        observations,
        config=guard.load_canonical_config(_root()),
        pd_module=pd,
        np_module=np,
    )
    assert truth[["station", "layer", "truth"]].to_dict("records") == [
        {"station": "S", "layer": 2, "truth": 12.0},
        {"station": "S", "layer": 3, "truth": 13.0},
    ]
    assert audit["validation_truth_rows"] == 2
    assert audit["validation_target_scalars_converted_after_commitment"] == 4
    assert audit["nonvalidation_target_scalars_converted"] == 0
    assert audit["hidden_test_target_scalars_converted"] == 0
    assert not any(b"HIDDEN_POISON" in value for value in decoded_fields)
    assert not any(b"OUTSIDE_POISON" in value for value in decoded_fields)


def test_prediction_commitment_is_aggregate_o_excl_and_detects_memory_drift(
    tmp_path: Path,
) -> None:
    engine = _engine()
    root = _root()
    config = guard.load_canonical_config(root)
    seeds = config["curve_protocol"]["seed_ids"]
    blind_by_fraction: dict[float, pd.DataFrame] = {}
    for index, registered in enumerate(config["curve_protocol"]["prefix_fractions"]):
        fraction = float(registered)
        frame = pd.DataFrame(
            {
                "fold": ["outer_2024_sep_oct", "outer_2025_may_jun"],
                "station": ["STATION_SENTINEL_A", "STATION_SENTINEL_B"],
                "layer": [2, 3],
                "time": pd.to_datetime(
                    ["2024-09-01T00:00:00+09:00", "2025-05-01T00:00:00+09:00"],
                    utc=True,
                ),
            }
        )
        for seed_offset, seed in enumerate(seeds):
            frame[f"seed_{seed}"] = [1.0 + seed_offset, 2.0 + seed_offset]
            frame[f"challenger_seed_{seed}"] = [
                10.0 + index + seed_offset,
                20.0 + index + seed_offset,
            ]
        frame["prediction_mean"] = frame[[f"seed_{seed}" for seed in seeds]].mean(axis=1)
        frame["challenger_mean"] = frame[[f"challenger_seed_{seed}" for seed in seeds]].mean(axis=1)
        frame = frame.loc[
            :,
            [
                "fold",
                "station",
                "layer",
                "time",
                *(f"seed_{seed}" for seed in seeds),
                "prediction_mean",
                *(f"challenger_seed_{seed}" for seed in seeds),
                "challenger_mean",
            ],
        ]
        blind_by_fraction[fraction] = frame
    cells = []
    cell_index = 0
    for fraction in config["curve_protocol"]["prefix_fractions"]:
        for fold in config["curve_protocol"]["outer_folds"]:
            for seed in seeds:
                cell_index += 1
                cells.append(
                    {
                        "fraction": float(fraction),
                        "fold": fold["name"],
                        "pipeline_seed": seed,
                        "derived_model_seed": cell_index,
                        "diagnostics": {
                            "outer_rows": 2,
                            "prediction_sha256": f"{cell_index:064x}",
                        },
                    }
                )
    numerical = SimpleNamespace(np=np)
    output = tmp_path / "stage_b"
    output.mkdir()
    commitment, pin = engine._write_prediction_commitment(
        workspace=root,
        output=output,
        config=config,
        blind_by_fraction=blind_by_fraction,
        cell_receipts=cells,
        numerical=numerical,
    )
    serialized = json.dumps(commitment, sort_keys=True)
    assert commitment["cell_prediction_count"] == 45
    assert commitment["total_rows"] == 10
    assert commitment["key_columns_in_order"] == ["fraction", "fold", "station", "layer", "time"]
    assert commitment["row_level_predictions_persisted"] is False
    assert "STATION_SENTINEL" not in serialized
    assert "challenger_seed_20260823" in serialized
    assert (
        engine._reverify_prediction_commitment(
            workspace=root,
            output=output,
            config=config,
            blind_by_fraction=blind_by_fraction,
            cell_receipts=cells,
            numerical=numerical,
            expected_pin=pin,
        )
        == commitment
    )
    with pytest.raises(FileExistsError):
        engine._write_prediction_commitment(
            workspace=root,
            output=output,
            config=config,
            blind_by_fraction=blind_by_fraction,
            cell_receipts=cells,
            numerical=numerical,
        )
    blind_by_fraction[0.4].loc[0, "challenger_seed_20260823"] += 0.25
    with pytest.raises((RuntimeError, ValueError)):
        engine._reverify_prediction_commitment(
            workspace=root,
            output=output,
            config=config,
            blind_by_fraction=blind_by_fraction,
            cell_receipts=cells,
            numerical=numerical,
            expected_pin=pin,
        )


def test_fixed_numeric_gate_thresholds() -> None:
    from p2_restore.meaningful_learning_curve import numeric_curve_gate

    points = [
        {
            "fraction": fraction,
            "incumbent": 1.0,
            "challenger": 0.96 if fraction < 1.0 else 0.97,
            "delta_ci90": [-0.05, -0.01],
        }
        for fraction in (0.4, 0.55, 0.7, 0.85, 1.0)
    ]
    gates = numeric_curve_gate(
        points,
        fold_deltas=[-0.04, -0.02, 0.01],
        slice_deltas={
            "layer_2": -0.02,
            "layer_3": -0.01,
            "layer_4": 0.0075,
            "2024_sep_oct": 0.0,
        },
    )
    assert all(gates.values())
    points[-1]["challenger"] = 0.9700000001
    assert (
        numeric_curve_gate(
            points,
            fold_deltas=[-0.04, -0.02, 0.01],
            slice_deltas={
                "layer_2": -0.02,
                "layer_3": -0.01,
                "layer_4": 0.0075,
                "2024_sep_oct": 0.0,
            },
        )["full_effect_meets_absolute_threshold"]
        is False
    )


def test_direct_engine_ignores_forged_preflight_and_stops_before_late_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = _engine()
    config = guard.load_canonical_config(_root())
    order: list[str] = []

    def stop_at_fresh(*_args: object, **kwargs: object) -> dict[str, object]:
        order.append("fresh_preflight")
        assert kwargs["supplied_config"] is config
        raise guard.StageBContractError("stage-a/source drift")

    monkeypatch.setattr(engine, "static_preflight", stop_at_fresh)
    monkeypatch.setattr(
        engine,
        "_load_numerical_stack",
        lambda *_args: order.append("late_numerical_import"),
    )
    output = _root() / config["canonical_paths"]["output"]
    assert not output.exists()
    with pytest.raises(guard.StageBContractError, match="stage-a/source drift"):
        engine.execute_stage_b(
            root=_root(),
            data_dir=tmp_path,
            config=config,
            preflight={"status": "FORGED_PASS"},
            attempt_lock=tmp_path / "forged.lock",
        )
    assert order == ["fresh_preflight"]
    assert not output.exists()


def test_engine_orders_fresh_preflight_and_pins_before_output_or_fit() -> None:
    source = (_root() / guard.ENGINE_RELATIVE).read_text(encoding="utf-8")
    body = source[source.index("def execute_stage_b(") :]
    positions = [
        body.index("fresh_preflight = static_preflight("),
        body.index("verified_reference = verify_stage_a_reference("),
        body.index("numerical = _load_numerical_stack("),
        body.index("runtime = numerical.engine._verify_runtime("),
        body.index("data_pins = numerical.engine._verify_data_pins("),
        body.index("verify_consumed_attempt_lock("),
        body.index("os.mkdir(output)"),
        body.index(
            "blind_by_fraction, cells, fold_blind_audits, input_names = _blind_predictions("
        ),
        body.index("commitment, commitment_pin = _write_prediction_commitment("),
        body.index("truth, truth_access_audit = _load_validation_truth_after_commitment("),
        body.index("_reverify_prediction_commitment("),
        body.index("evidence = _evaluate_curve_after_commitment("),
    ]
    assert positions == sorted(positions)
    assert "del preflight" in body[: positions[0]]
    assert "load_p2_data(" not in body


def test_no_personal_absolute_data_path_is_embedded_in_stage_b_tests() -> None:
    source = (_root() / guard.IMPLEMENTATION_ROLES["TESTS"]).read_text(encoding="utf-8")
    assert re.search(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]", source) is None


def test_runner_verifies_stage_a_and_qa_before_lock_and_engine_import() -> None:
    source = (_root() / guard.IMPLEMENTATION_ROLES["RUNNER"]).read_text(encoding="utf-8")
    body = source[source.index("def run_authorized(") :]
    positions = [
        body.index("guard.verify_stage_a_reference("),
        body.index("preflight = guard.static_preflight("),
        body.index("guard.verify_pre_execution_qa("),
        body.index("guard.verify_execution_authorization("),
        body.index("lock = guard.consume_attempt_lock("),
        body.index("engine = importlib.import_module(ENGINE_MODULE)"),
    ]
    assert positions == sorted(positions)


def test_runner_missing_qa_rejects_before_lock_or_engine_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import run_p2_architecture_matched_stage_b_v3 as runner

    config = guard.load_canonical_config(_root())
    order: list[str] = []
    fake_guard = type(
        "FakeGuard",
        (),
        {
            "__file__": str(_root() / runner.GUARD_RELATIVE),
            "load_canonical_config": staticmethod(lambda *_args, **_kwargs: config),
            "verify_stage_a_reference": staticmethod(
                lambda *_args, **_kwargs: order.append("stage_a") or {}
            ),
            "static_preflight": staticmethod(
                lambda *_args, **_kwargs: order.append("preflight") or {"implementation_pins": {}}
            ),
            "verify_pre_execution_qa": staticmethod(
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    PermissionError("missing Stage-B QA")
                )
            ),
        },
    )

    def fake_import(name: str) -> object:
        order.append(f"import:{name}")
        if name == runner.GUARD_MODULE:
            return fake_guard
        raise AssertionError("Stage-B engine import must not occur")

    monkeypatch.setattr(runner.importlib, "import_module", fake_import)
    with pytest.raises(PermissionError, match="missing Stage-B QA"):
        runner.run_authorized(root=_root(), data_dir=tmp_path)
    assert order == [f"import:{runner.GUARD_MODULE}", "stage_a", "preflight"]


def test_check_only_parent_import_boundary_is_truthful_and_isolated() -> None:
    root = _root()
    script = r"""
import json
import sys
from pathlib import Path
from scripts import run_p2_architecture_matched_stage_b_v3 as runner
before = sorted(sys.modules)
runner._isolated_preflight = lambda **kwargs: {
    "status": "PASS_STATIC_IMPLEMENTATION_ONLY_STAGE_A_SEALED",
    "preflight_process_loaded_numerical_modules": ["numpy", "pandas", "torch"],
    "challenger_engine_imported": False,
}
result = runner.check_only(root=Path(sys.argv[1]), data_dir=Path(sys.argv[1]))
print(json.dumps({
    "result": result,
    "guard_loaded": runner.GUARD_MODULE in sys.modules,
    "engine_loaded": runner.ENGINE_MODULE in sys.modules,
    "new_modules": sorted(set(sys.modules) - set(before)),
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "src")))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(completed.stdout)
    parent = observed["result"]["check_only_parent_process"]
    assert observed["guard_loaded"] is False
    assert observed["engine_loaded"] is False
    assert parent["new_numerical_modules"] == []
    assert parent["guard_imported_after"] is False
    assert parent["engine_imported_after"] is False
    assert observed["result"]["execution_engine_imported_by_parent"] is False


def test_static_state_has_no_qa_authorization_lock_output_or_candidate() -> None:
    root = _root()
    config = guard.load_canonical_config(root)
    paths = guard.stage_paths(root, config)
    assert {key: path.exists() for key, path in paths.items()} == {
        "output": False,
        "control": False,
        "pre_execution_qa": False,
        "authorization": False,
        "attempt_lock": False,
    }
    output = root / config["canonical_paths"]["output"]
    forbidden = (
        output / "candidate",
        output / "submission.csv",
        output / "test_predictions.csv",
    )
    assert not any(path.exists() for path in forbidden)
    assert config["output_contract"]["full_fit_performed"] is False
    assert config["output_contract"]["candidate_generated"] is False
    assert config["output_contract"]["test_prediction_generated"] is False
    assert config["output_contract"]["upload_performed"] is False
