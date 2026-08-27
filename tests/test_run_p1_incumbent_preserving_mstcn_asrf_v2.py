from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_p1_incumbent_preserving_mstcn_asrf_v2.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("p1_mstcn_asrf_runner_tested", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mode_parser_has_exactly_three_mutually_exclusive_modes() -> None:
    runner = _load_runner()
    assert runner._parse_args(["--check-only"]).check_only
    assert runner._parse_args(["--smoke"]).smoke
    assert runner._parse_args(["--execute-protocol"]).execute_protocol
    with pytest.raises(SystemExit):
        runner._parse_args([])
    with pytest.raises(SystemExit):
        runner._parse_args(["--smoke", "--execute-protocol"])


def test_runner_contains_no_protected_interface_filename_literals() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8").casefold()
    protected = [
        "".join(map(chr, (116, 101, 115, 116, 46, 99, 115, 118))),
        "".join(
            map(
                chr,
                (
                    115,
                    97,
                    109,
                    112,
                    108,
                    101,
                    95,
                    115,
                    117,
                    98,
                    109,
                    105,
                    115,
                    115,
                    105,
                    111,
                    110,
                    46,
                    99,
                    115,
                    118,
                ),
            )
        ),
    ]
    assert all(value not in source for value in protected)


def test_external_execution_attestation_covers_full_local_implementation() -> None:
    runner = _load_runner()
    identities = runner._implementation_identity()
    assert set(identities) == {
        "runner",
        "config",
        "package_init",
        "model",
        "data",
        "current_router_anchor_builder",
        "capacity_calibration_builder",
    }
    attestation = {
        "verified_by": runner.EXTERNAL_LAUNCHER_NAME,
        "launcher_sha256": runner._sha256(
            runner.ROOT / "scripts" / runner.EXTERNAL_LAUNCHER_NAME
        ),
        "externally_expected_launcher_sha256": runner._sha256(
            runner.ROOT / "scripts" / runner.EXTERNAL_LAUNCHER_NAME
        ),
        "external_launcher_hash_acknowledged": True,
        "identities": identities,
        "all_pins_matched": True,
    }
    assert runner._verify_external_implementation_attestation(
        attestation, launcher_capability=runner._SEALED_LAUNCHER_CAPABILITY
    )[
        "all_pins_matched"
    ]
    mutated = json.loads(json.dumps(attestation))
    mutated["identities"]["model"]["sha256"] = "0" * 64
    with pytest.raises(runner.ContractError, match="differs"):
        runner._verify_external_implementation_attestation(
            mutated, launcher_capability=runner._SEALED_LAUNCHER_CAPABILITY
        )
    with pytest.raises(runner.ContractError, match="capability"):
        runner._verify_external_implementation_attestation(attestation)
    with pytest.raises(runner.ContractError, match="direct runner API"):
        runner.execute_protocol(implementation_attestation=attestation)


def test_full_config_seal_and_external_launcher_literal_pins() -> None:
    runner = _load_runner()
    assert runner.EXPECTED_CONFIG_SHA256 == runner._sha256(runner.CONFIG_PATH)
    launcher_path = runner.ROOT / "scripts" / runner.EXTERNAL_LAUNCHER_NAME
    spec = importlib.util.spec_from_file_location("p1_mstcn_launcher_tested", launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    identities = launcher.verify_literal_pins()
    assert identities["runner"]["sha256"] == runner._sha256(RUNNER_PATH)
    assert identities["config"]["sha256"] == runner.EXPECTED_CONFIG_SHA256


def test_v2_has_no_stale_inner_validation_protocol() -> None:
    runner = _load_runner()
    runner_source = RUNNER_PATH.read_text(encoding="utf-8").casefold()
    config_source = runner.CONFIG_PATH.read_text(encoding="utf-8").casefold()
    assert "inner_" not in runner_source
    assert "inner_" not in config_source


def test_decoder_snaps_boundaries_and_anchor_union_never_removes_a_one() -> None:
    runner = _load_runner()
    source = str(Path(__file__).resolve().parents[1] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from p1_qc.ms_tcn_asrf_data import SegmentLayout

    rows = 100
    times = pd.date_range("2025-01-01", periods=rows, freq="10min", tz="Asia/Seoul")
    layout = SegmentLayout.from_aligned(
        ["S"] * rows,
        [2025] * rows,
        [1] * rows,
        times.astype(str),
    )
    probability = np.zeros(rows, dtype=np.float32)
    probability[30:50] = 0.8
    probability[28:54] = np.maximum(probability[28:54], 0.4)
    boundary = np.zeros((rows, 2), dtype=np.float32)
    boundary[28, 0] = 1.0
    boundary[53, 1] = 1.0
    proposal = runner.decode_long_event_segments(
        probability,
        boundary,
        layout,
        high_threshold=0.7,
        low_threshold=0.35,
        minimum_rows=19,
        maximum_rows=60,
    )
    assert proposal[28:54].all()
    assert int(proposal.sum()) == 26
    anchor = np.zeros(rows, dtype=np.int8)
    anchor[[2, 70]] = 1
    candidate = runner.anchor_preserving_union(anchor, proposal)
    assert candidate[2] == 1 and candidate[70] == 1
    assert not np.any((anchor == 1) & (candidate == 0))


def test_binary_and_event_metrics_are_exact() -> None:
    runner = _load_runner()
    score = runner.binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert score["tp"] == 1
    assert score["fp"] == 1
    assert score["fn"] == 1
    assert score["f1"] == pytest.approx(0.5)
    event = runner.event_metrics(
        [0, 1, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 0],
    )
    assert event["event_recall_iou_0_70"] == 1.0
    assert event["median_event_iou"] == 1.0
    even = runner.event_metrics(
        [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    )
    assert even["median_event_iou"] == pytest.approx(0.875)


def test_all_fifteen_immutable_inputs_are_verified_by_size_and_hash(tmp_path: Path) -> None:
    runner = _load_runner()
    names = (
        "feature_cache",
        "feature_metadata",
        "feature_key_sidecar",
        "training_labels",
        "frozen_truth_and_folds",
        "frozen_round_b_anchor",
        "frozen_current_router_components",
        "frozen_current_router_anchor",
        "frozen_current_router_manifest",
        "capacity_calibration_receipt",
        "capacity_calibration_builder",
        "current_router_anchor_builder",
        "model_implementation",
        "data_implementation",
        "package_init_implementation",
    )
    records = {}
    for index, name in enumerate(names):
        path = tmp_path / f"input_{index}.bin"
        payload = f"pin-{index}".encode("ascii")
        path.write_bytes(payload)
        records[name] = {
            "path": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    verified = runner.verify_immutable_inputs({"immutable_inputs": records}, root=tmp_path)
    assert set(verified) == set(names)
    (tmp_path / "input_3.bin").write_bytes(b"changed")
    with pytest.raises(runner.ContractError, match="training_labels"):
        runner.verify_immutable_inputs({"immutable_inputs": records}, root=tmp_path)


def test_delayed_immutable_read_detects_mutation_during_reader(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "delayed.bin"
    path.write_bytes(b"before")
    config = {
        "immutable_inputs": {
            "training_labels": {
                "path": path.name,
                "bytes": 6,
                "sha256": hashlib.sha256(b"before").hexdigest(),
            }
        }
    }
    with pytest.raises(runner.ContractError, match="training_labels"):
        with runner._verified_immutable_read(
            config, "training_labels", root=tmp_path
        ) as verified:
            assert verified == path.resolve()
            path.write_bytes(b"during")


def test_every_delayed_critical_reader_uses_pre_post_identity_guard() -> None:
    runner = _load_runner()
    blind_source = inspect.getsource(runner.load_blind_surfaces)
    for name in (
        "feature_metadata",
        "feature_cache",
        "feature_key_sidecar",
        "training_labels",
        "frozen_truth_and_folds",
        "frozen_current_router_components",
        "frozen_current_router_anchor",
    ):
        assert f'config, "{name}", root=root' in blind_source
    assert "_verified_immutable_read" in inspect.getsource(
        runner._training_surface_for_cutoff
    )
    assert "_verified_immutable_read" in inspect.getsource(
        runner.load_fold_truth_after_receipts
    )


def test_blind_receipt_is_committed_before_truth_loader_can_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    source = inspect.getsource(runner.execute_protocol)
    assert source.index("commit_q2_qualification_grid(") < source.index(
        "load_fold_truth_after_receipts("
    )
    assert source.index("load_sealed_q2_grid(") < source.index(
        "load_fold_truth_after_receipts("
    )
    assert source.index("commit_confirmatory_blind(") < source.index(
        "truths = {"
    )

    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "verify_blind_receipt",
        lambda _path: calls.append("verify") or (_ for _ in ()).throw(runner.ContractError("stop")),
    )
    import pyarrow.dataset as arrow_dataset

    monkeypatch.setattr(
        arrow_dataset,
        "dataset",
        lambda *_args, **_kwargs: calls.append("open") or pytest.fail("truth opened early"),
    )
    keys = pd.DataFrame(
        {"station": ["S"], "year": [2025], "layer": [1], "time": ["2025-04-01 00:00:00+09:00"]}
    )
    q2 = runner.RowSurface(keys, np.zeros((1, 1)), np.array(["S"]), np.array(["L"]), np.array(["D"]))
    with pytest.raises(runner.ContractError, match="stop"):
        runner.load_fold_truth_after_receipts(
            {}, q2, [tmp_path / "missing.json"], fold="2025_q2", root=tmp_path
        )
    assert calls == ["verify"]


def test_blind_receipt_grid_semantics_path_and_bytes_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    config = json.loads(json.dumps(runner._canonical_config()))
    config["phase_protocols"]["q2"]["membership_identity"]["membership_rows"] = 2
    monkeypatch.setattr(runner, "_canonical_config", lambda *_args, **_kwargs: config)
    checkpoint_epochs = np.asarray(runner._checkpoint_epochs(config), dtype=np.int16)
    widths = np.repeat(np.asarray([256, 512], dtype=np.int16), len(checkpoint_epochs))
    epochs = np.tile(checkpoint_epochs, 2)
    thresholds = np.asarray(config["decoder"]["q2_high_threshold_grid"], dtype=np.float64)
    capacities = len(widths)
    threshold_count = len(thresholds)
    grid = runner.QualificationGrid(
        widths=widths,
        epochs=epochs,
        thresholds=thresholds,
        row_probability=np.full((capacities, 2), 0.5, dtype=np.float32),
        boundary_probability=np.full((capacities, 2, 2), 0.5, dtype=np.float32),
        proposal=np.zeros((capacities, threshold_count, 2), dtype=np.int8),
        candidate=np.ones((capacities, threshold_count, 2), dtype=np.int8),
        fit_receipts=[{} for _ in range(6)],
    )
    receipt = runner.commit_q2_qualification_grid(
        grid,
        key_sha256="a" * 64,
        config_sha256=runner._sha256(runner.CONFIG_PATH),
        artifact_dir=tmp_path,
    )
    assert (
        runner.verify_blind_receipt(receipt)[
            "same_fold_holdout_truth_columns_opened_before_receipt"
        ]
        == 0
    )
    sealed = runner.load_sealed_q2_grid(receipt)
    assert sealed.candidate.shape == (126, 7, 2)
    assert sealed.candidate.all()

    escaped = json.loads(receipt.read_text(encoding="utf-8"))
    escaped["score_path"] = "../outside.npz"
    receipt.write_text(json.dumps(escaped), encoding="utf-8")
    with pytest.raises(runner.ContractError, match="basename"):
        runner.verify_blind_receipt(receipt)

    valid_dir = tmp_path / "byte_tamper"
    valid_dir.mkdir()
    byte_receipt = runner.commit_q2_qualification_grid(
        grid,
        key_sha256="a" * 64,
        config_sha256=runner._sha256(runner.CONFIG_PATH),
        artifact_dir=valid_dir,
    )
    score_path = valid_dir / json.loads(byte_receipt.read_text(encoding="utf-8"))["score_path"]
    score_path.write_bytes(score_path.read_bytes() + b"changed")
    with pytest.raises(runner.ContractError, match="changed"):
        runner.verify_blind_receipt(byte_receipt)

    shape_dir = tmp_path / "shape_tamper"
    shape_dir.mkdir()
    invalid = runner.QualificationGrid(
        widths=grid.widths,
        epochs=grid.epochs,
        thresholds=grid.thresholds,
        row_probability=grid.row_probability,
        boundary_probability=grid.boundary_probability,
        proposal=grid.proposal,
        candidate=grid.candidate[:, :, :1],
        fit_receipts=grid.fit_receipts,
    )
    shape_receipt = runner.commit_q2_qualification_grid(
        invalid,
        key_sha256="a" * 64,
        config_sha256=runner._sha256(runner.CONFIG_PATH),
        artifact_dir=shape_dir,
    )
    with pytest.raises(runner.ContractError, match="shape or dtype"):
        runner.load_sealed_q2_grid(shape_receipt)


def test_q2_blind_semantic_replay_rejects_candidate_mutation() -> None:
    runner = _load_runner()
    source = str(Path(__file__).resolve().parents[1] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from p1_qc.ms_tcn_asrf_data import SegmentLayout

    config = runner._canonical_config()
    rows = 24
    keys = pd.DataFrame(
        {
            "station": ["S"] * rows,
            "year": [2025] * rows,
            "layer": [1] * rows,
            "time": pd.date_range(
                "2025-04-01", periods=rows, freq="10min", tz="UTC"
            ).astype(str),
        }
    )
    layout = SegmentLayout.from_aligned(
        keys["station"], keys["year"], keys["layer"], keys["time"]
    )
    anchor = np.zeros(rows, dtype=np.int8)
    surface = runner.RowSurface(
        keys,
        np.zeros((rows, 1), dtype=np.float32),
        np.asarray(["S"] * rows),
        np.asarray(["1"] * rows),
        None,
        anchor=anchor,
    )
    encoded = runner.EncodedSurface(
        surface, np.zeros((rows, 1), dtype=np.float32), layout, None
    )
    epochs = np.asarray(runner._checkpoint_epochs(config), dtype=np.int16)
    widths = np.repeat(np.asarray([256, 512], dtype=np.int16), len(epochs))
    epoch_rows = np.tile(epochs, 2)
    thresholds = np.asarray(config["decoder"]["q2_high_threshold_grid"])
    row_probability = np.full((126, rows), 0.95, dtype=np.float32)
    boundary_probability = np.zeros((126, rows, 2), dtype=np.float32)
    boundary_probability[:, 0, 0] = 1.0
    boundary_probability[:, -1, 1] = 1.0
    proposal = np.ones((126, 7, rows), dtype=np.int8)
    candidate = proposal.copy()
    grid = runner.QualificationGrid(
        widths,
        epoch_rows,
        thresholds,
        row_probability,
        boundary_probability,
        proposal,
        candidate,
        [],
    )
    assert runner.validate_sealed_q2_decoder_semantics(
        grid, encoded, config=config
    )["decoder_cells_replayed"] == 882
    mutated = runner.QualificationGrid(
        grid.widths,
        grid.epochs,
        grid.thresholds,
        grid.row_probability,
        grid.boundary_probability,
        grid.proposal,
        grid.candidate.copy(),
        [],
    )
    mutated.candidate[0, 0, 0] = 0
    with pytest.raises(runner.ContractError, match="anchor union replay"):
        runner.validate_sealed_q2_decoder_semantics(mutated, encoded, config=config)


def test_confirmatory_blind_semantic_replay_rejects_union_mutation(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    source = str(Path(__file__).resolve().parents[1] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from p1_qc.ms_tcn_asrf_data import SegmentLayout

    config = json.loads(json.dumps(runner._canonical_config()))
    rows = 24
    config["phase_protocols"]["q3"]["membership_identity"]["membership_rows"] = rows
    keys = pd.DataFrame(
        {
            "station": ["S"] * rows,
            "year": [2025] * rows,
            "layer": [1] * rows,
            "time": pd.date_range(
                "2025-07-01", periods=rows, freq="10min", tz="UTC"
            ).astype(str),
        }
    )
    layout = SegmentLayout.from_aligned(
        keys["station"], keys["year"], keys["layer"], keys["time"]
    )
    anchor = np.zeros(rows, dtype=np.int8)
    surface = runner.RowSurface(
        keys,
        np.zeros((rows, 1), dtype=np.float32),
        np.asarray(["S"] * rows),
        np.asarray(["1"] * rows),
        None,
        anchor=anchor,
    )
    encoded = runner.EncodedSurface(
        surface, np.zeros((rows, 1), dtype=np.float32), layout, None
    )
    boundary_probability = np.zeros((rows, 2), dtype=np.float32)
    boundary_probability[0, 0] = 1.0
    boundary_probability[-1, 1] = 1.0
    blind = runner.PredictionBundle(
        np.full(rows, 0.95, dtype=np.float32),
        boundary_probability,
        np.tile(np.asarray([[0, 1, 0, 0, 0]], dtype=np.float32), (rows, 1)),
    )
    proposal = np.ones(rows, dtype=np.int8)
    candidate = proposal.copy()
    candidate[0] = 0
    selected_recipe = {"threshold": 0.9, "width": 256, "epoch": 5}
    receipt = runner.commit_confirmatory_blind(
        blind,
        proposal,
        candidate,
        phase="q3",
        fold="2025_q3",
        key_sha256="a" * 64,
        config_sha256=runner._sha256(runner.CONFIG_PATH),
        selected_recipe=selected_recipe,
        fit_receipts=[{} for _ in range(3)],
        artifact_dir=tmp_path,
    )
    with pytest.raises(runner.ContractError, match="anchor union replay"):
        runner.load_sealed_confirmatory_candidate(
            receipt,
            holdout=encoded,
            config=config,
            selected_recipe=selected_recipe,
        )


def test_confirmatory_stop_preserves_the_full_registered_lr_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    config = json.loads(json.dumps(runner._canonical_config()))
    config["training"].update(
        {
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "maximum_epochs": 300,
            "warmup_epochs": 10,
        }
    )
    synthetic_window_count = 17
    selected_epoch = 12
    steps_per_epoch, full_total_steps, warmup_steps = runner._schedule_geometry(
        config, window_count=synthetic_window_count
    )
    assert steps_per_epoch == 2
    assert full_total_steps == 600
    assert warmup_steps == 20

    model = torch.nn.Linear(2, 1)
    model.trainable_parameter_count = sum(value.numel() for value in model.parameters())
    monkeypatch.setattr(runner, "_new_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        runner,
        "_selected_windows",
        lambda *_args, **_kwargs: tuple(range(synthetic_window_count)),
    )
    monkeypatch.setattr(runner, "_all_windows", lambda *_args, **_kwargs: (object(),))
    monkeypatch.setattr(runner, "_positive_weight", lambda _labels: 1.0)
    supplied_total_steps: list[int] = []

    def fake_train_epoch(*_args, **kwargs):
        supplied_total_steps.append(int(kwargs["total_steps"]))
        telemetry = {
            "total_loss": 0.25,
            "event_loss": 0.1,
            "temporal_smoothing_loss": 0.02,
            "boundary_loss": 0.05,
            "anomaly_type_loss": 0.08,
            "grad_norm_mean": 0.5,
            "grad_norm_max": 0.6,
            "grad_norm_last": 0.4,
            "gradient_clip_count": 0,
            "optimizer_steps_epoch": steps_per_epoch,
            "observed_windows": synthetic_window_count,
            "nonfinite_count": 0,
        }
        return telemetry, int(kwargs["global_step"]) + steps_per_epoch, 0.001

    monkeypatch.setattr(runner, "_train_epoch", fake_train_epoch)
    blind = runner.PredictionBundle(
        np.asarray([0.5], dtype=np.float32),
        np.asarray([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 5), dtype=np.float32),
    )
    monkeypatch.setattr(runner, "predict_encoded", lambda *_args, **_kwargs: blind)
    surface = SimpleNamespace(labels=np.asarray([0, 1], dtype=np.int8))
    encoded = SimpleNamespace(features=np.zeros((2, 2), dtype=np.float32), surface=surface)
    returned, receipt = runner._smoke_refit_and_predict(
        encoded,
        encoded,
        config=config,
        selected_epoch=selected_epoch,
        device=torch.device("cpu"),
        artifact_dir=None,
    )
    assert returned is blind
    assert receipt["epochs"] == selected_epoch
    assert supplied_total_steps == [full_total_steps] * selected_epoch

    maximum_lr = float(config["training"]["learning_rate"])
    minimum_lr = 3e-6
    for step in (0, selected_epoch * steps_per_epoch - 1):
        registered_lr = runner._lr_at_step(
            step,
            total_steps=full_total_steps,
            warmup_steps=warmup_steps,
            maximum_lr=maximum_lr,
            minimum_lr=minimum_lr,
        )
        stopped_refit_lr = runner._lr_at_step(
            step,
            total_steps=supplied_total_steps[0],
            warmup_steps=warmup_steps,
            maximum_lr=maximum_lr,
            minimum_lr=minimum_lr,
        )
        assert stopped_refit_lr == registered_lr
    compressed_lr = runner._lr_at_step(
        selected_epoch * steps_per_epoch - 1,
        total_steps=selected_epoch * steps_per_epoch,
        warmup_steps=warmup_steps,
        maximum_lr=maximum_lr,
        minimum_lr=minimum_lr,
    )
    full_horizon_lr = runner._lr_at_step(
        selected_epoch * steps_per_epoch - 1,
        total_steps=full_total_steps,
        warmup_steps=warmup_steps,
        maximum_lr=maximum_lr,
        minimum_lr=minimum_lr,
    )
    assert compressed_lr != full_horizon_lr


def test_feature_dependency_projection_is_exact_and_fails_closed() -> None:
    runner = _load_runner()
    config = runner._canonical_config()
    metadata_path = (
        Path(__file__).resolve().parents[1]
        / config["immutable_inputs"]["feature_metadata"]["path"]
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    numeric, projected, receipt = runner._feature_dependency_audit(metadata, config)
    assert len(numeric) == 74
    assert not runner.UNBOUNDED_CACHED_FEATURES.intersection(projected)
    assert receipt["unbounded_features_projected"] == 0
    mutated = json.loads(json.dumps(metadata))
    mutated["feature_columns"].append("unknown_full_series_statistic")
    with pytest.raises(runner.ContractError, match="classification is not exact"):
        runner._feature_dependency_audit(mutated, config)


def test_subset_surface_supports_excluded_cached_depth_regime() -> None:
    runner = _load_runner()
    keys = pd.DataFrame(
        {
            "station": ["S", "S"],
            "year": [2025, 2025],
            "layer": [1, 1],
            "time": ["2025-01-01T00:00:00Z", "2025-01-01T00:10:00Z"],
        }
    )
    surface = runner.RowSurface(
        keys=keys,
        numeric=np.zeros((2, 1), dtype=np.float32),
        station=np.asarray(["S", "S"]),
        layer_category=np.asarray(["1", "1"]),
        depth_regime=None,
        depth=np.asarray([10.0, 11.0], dtype=np.float32),
    )
    subset = runner._subset_surface(surface, [1])
    assert subset.depth_regime is None
    assert subset.depth.tolist() == [11.0]


def test_checkpoint_epochs_are_required_union_every_five() -> None:
    runner = _load_runner()
    epochs = runner._checkpoint_epochs(runner._canonical_config())
    assert len(epochs) == 63
    assert epochs[:5] == (1, 2, 3, 5, 10)
    assert epochs[-1] == 300
    assert set(range(5, 301, 5)).issubset(epochs)


def test_training_receipts_expose_convergence_and_gradient_diagnostics() -> None:
    runner = _load_runner()
    train_source = inspect.getsource(runner._train_epoch)
    for token in (
        "loss_output.event",
        "loss_output.temporal_smoothing",
        "loss_output.boundary",
        "loss_output.anomaly_type",
        '"grad_norm_mean"',
        '"grad_norm_max"',
        '"gradient_clip_count"',
        '"nonfinite_count"',
    ):
        assert token in train_source
    capacity_source = inspect.getsource(runner._fit_capacity_seed_checkpoints)
    confirm_source = inspect.getsource(runner._fit_raw_seed_to_epoch)
    assert "_cuda_peak_memory_receipt" in capacity_source
    assert "_cuda_peak_memory_receipt" in confirm_source
    protocol_source = inspect.getsource(runner.execute_protocol)
    assert "RIGHT_CENSORED_AT_MAX_EPOCH_NOT_PROVEN_CONVERGED" in protocol_source
    assert "GO_RESEARCH_ONLY_NOT_OFFICIAL_PROBE_ELIGIBLE" in protocol_source
    assert "GO_RESEARCH_SUCCESS_OFFICIAL_PROBE_NOT_AUTHORIZED" not in protocol_source


def test_long_type_head_is_used_by_the_fixed_decoder_score() -> None:
    runner = _load_runner()
    row = np.asarray([0.8, 0.8], dtype=np.float32)
    kinds = np.zeros((2, 5), dtype=np.float32)
    kinds[0, 0] = 1.0  # spike-only support
    kinds[1, 1] = 1.0  # noise support
    conditioned = runner._long_type_conditioned_row_probability(
        row, kinds, weight=0.25
    )
    assert conditioned[0] == pytest.approx(0.6)
    assert conditioned[1] == pytest.approx(0.8)


def test_q2_selection_uses_anchor_union_and_smaller_width_tie_break() -> None:
    runner = _load_runner()
    source = str(Path(__file__).resolve().parents[1] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from p1_qc.ms_tcn_asrf_data import SegmentLayout

    keys = pd.DataFrame(
        {
            "station": ["S"] * 4,
            "year": [2025] * 4,
            "layer": [1] * 4,
            "time": pd.date_range("2025-04-01", periods=4, freq="10min", tz="UTC").astype(str),
        }
    )
    layout = SegmentLayout.from_aligned(
        keys["station"], keys["year"], keys["layer"], keys["time"]
    )
    anchor = np.asarray([1, 0, 0, 0], dtype=np.int8)
    surface = runner.RowSurface(
        keys,
        np.zeros((4, 1), dtype=np.float32),
        np.asarray(["S"] * 4),
        np.asarray(["1"] * 4),
        None,
        anchor=anchor,
        depth=np.full(4, 10.0, dtype=np.float32),
    )
    encoded = runner.EncodedSurface(surface, np.zeros((4, 1)), layout, None)
    proposal = np.asarray([[[0, 1, 0, 0]], [[0, 1, 0, 0]]], dtype=np.int8)
    candidate = np.maximum(proposal, anchor[None, None, :]).astype(np.int8)
    grid = runner.QualificationGrid(
        widths=np.asarray([512, 256]),
        epochs=np.asarray([5, 10]),
        thresholds=np.asarray([0.5]),
        row_probability=np.zeros((2, 4), dtype=np.float32),
        boundary_probability=np.zeros((2, 4, 2), dtype=np.float32),
        proposal=proposal,
        candidate=candidate,
        fit_receipts=[],
    )
    truth = keys.copy()
    truth["label"] = [1, 1, 0, 0]
    truth["anomaly_type"] = ["noise", "noise", "", ""]
    result = runner.select_q2_recipe(truth, encoded, grid, config=runner._canonical_config())
    assert result["selected"]["width"] == 256
    assert result["selected"]["candidate"]["f1"] == 1.0
    assert len(result["grid_records"]) == 2
    assert "standalone_proposal" in result["grid_records"][0]
    assert result["convergence_evidence"]["curve_rows"] == 2


def test_long_event_mask_includes_noise_and_fp_gate_uses_row_ratio() -> None:
    runner = _load_runner()
    source = str(Path(__file__).resolve().parents[1] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from p1_qc.ms_tcn_asrf_data import SegmentLayout

    rows = 24
    keys = pd.DataFrame(
        {
            "station": ["S"] * rows,
            "year": [2025] * rows,
            "layer": [1] * rows,
            "time": pd.date_range("2025-04-01", periods=rows, freq="10min", tz="UTC").astype(str),
        }
    )
    layout = SegmentLayout.from_aligned(
        keys["station"], keys["year"], keys["layer"], keys["time"]
    )
    truth = keys.copy()
    truth["label"] = np.r_[np.ones(20, dtype=np.int8), np.zeros(4, dtype=np.int8)]
    truth["anomaly_type"] = ["noise"] * 20 + [""] * 4
    mask = runner._long_event_mask(truth, layout, minimum_rows=19)
    assert int(mask.sum()) == 20


def test_runner_forwards_materialized_valid_masks_and_sanity_uses_real_path() -> None:
    runner = _load_runner()
    train_source = inspect.getsource(runner._train_epoch)
    predict_source = inspect.getsource(runner.predict_encoded)
    sanity_source = inspect.getsource(runner.run_sanity_gate)
    assert "valid_mask=feature_valid_tensor" in train_source
    assert "valid_mask=valid_tensor" in predict_source
    for call in (
        "stitch_center_weighted(",
        "_decoder_row_probability(",
        "decode_long_event_segments(",
        "anchor_preserving_union(",
    ):
        assert call in sanity_source
    assert '"evaluation_weights": "raw"' in sanity_source


def test_scientific_lock_occurs_after_read_only_preparation() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner.execute_protocol)
    lock = source.index("acquire_attempt_namespace(")
    assert source.index("check_only(") < lock
    assert source.index("load_blind_surfaces(") < lock
    assert source.index("_prepare_phase_surfaces(") < lock
    assert source.index("runtime_before_lock = verify_runtime_identity(") < lock
    assert source.index("run_sanity_gate(") > lock
    assert source.index("fit_q2_qualification_grid(") > lock


def test_attempt_namespace_rolls_back_only_its_lock_on_mkdir_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    lock = tmp_path / "attempt.lock.json"
    artifact = tmp_path / "artifact"
    preflight = {
        "config_sha256": "a" * 64,
        "immutable_inputs": {"one": {"sha256": "b" * 64}},
        "external_implementation_attestation": {
            "identities": {"runner": {"sha256": "c" * 64}},
            "launcher_sha256": "d" * 64,
        },
        "runtime_identity_immediately_before_lock": {"result": "PASS"},
    }
    path_type = type(artifact)
    original_mkdir = path_type.mkdir

    def fail_artifact_mkdir(self, *args, **kwargs):
        if self == artifact:
            raise OSError("injected namespace failure")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(path_type, "mkdir", fail_artifact_mkdir)
    with pytest.raises(OSError, match="injected"):
        runner.acquire_attempt_namespace(
            preflight, path=lock, artifact_dir=artifact
        )
    assert not lock.exists()
    assert not artifact.exists()


def test_runtime_identity_rejects_version_and_device_mutation() -> None:
    runner = _load_runner()
    config = runner._canonical_config()
    observed = runner._observe_runtime_identity()
    assert runner._validate_runtime_identity(
        observed, config["runtime_identity"]
    )["result"] == "PASS_EXACT_RUNTIME_IDENTITY"
    wrong_version = dict(config["runtime_identity"])
    wrong_version["numpy"] = "0.0.0"
    with pytest.raises(runner.ContractError, match="runtime identity"):
        runner._validate_runtime_identity(observed, wrong_version)
    wrong_device = dict(config["runtime_identity"])
    wrong_device["cuda_device"] = "different device"
    with pytest.raises(runner.ContractError, match="runtime identity"):
        runner._validate_runtime_identity(observed, wrong_device)


def test_frozen_membership_envelopes_and_series_local_chronology_are_exact() -> None:
    runner = _load_runner()
    config = runner._canonical_config()
    oof_path = runner.ROOT / config["immutable_inputs"]["frozen_truth_and_folds"]["path"]
    memberships = {}
    receipts = {}
    for fold in ("2025_q2", "2025_q3", "2025_q4"):
        source = runner._read_fold_membership_without_truth(oof_path, fold)
        membership, receipt = runner._validate_registered_holdout_membership(
            source, config, fold=fold
        )
        memberships[fold] = membership
        receipts[fold] = receipt
    assert receipts["2025_q2"]["membership_rows"] == 133170
    assert receipts["2025_q3"]["membership_rows"] == 176738
    assert receipts["2025_q4"]["membership_rows"] == 111124
    assert receipts["2025_q3"]["excluded_out_of_bounds_rows"] == 0
    assert receipts["2025_q3"]["membership_max_time_utc"] == (
        "2025-10-01T10:40:00+00:00"
    )
    old_global_boundary = pd.Timestamp("2025-09-30T15:00:00+00:00")
    q3_times = pd.to_datetime(
        memberships["2025_q3"]["time"], utc=True, format="mixed"
    )
    assert int((q3_times >= old_global_boundary).sum()) == 119
    separation = runner._assert_series_local_fold_chronology(memberships)
    assert separation["pairwise_exact_key_overlap_rows"] == 0
    assert separation["series_local_chronology_violations"] == 0
    assert separation["minimum_series_local_gap_minutes"] == pytest.approx(10.0)
    assert separation["pairwise_shared_kst_calendar_days"]["2025_q3|2025_q4"] == 1


def test_confirmatory_anchor_identity_joint_day_bootstrap_and_gates() -> None:
    runner = _load_runner()
    import pyarrow.dataset as dataset

    config = json.loads(json.dumps(runner._canonical_config()))
    config["confirmatory_gate"]["bootstrap_replicates"] = 128
    oof_path = runner.ROOT / config["immutable_inputs"]["frozen_truth_and_folds"]["path"]
    anchor_record = config["immutable_inputs"]["frozen_current_router_anchor"]
    anchor_path = runner.ROOT / anchor_record["path"]
    truths = {}
    anchors = {}
    holdouts = {}
    for phase in ("q3", "q4"):
        fold = config["phase_protocols"][phase]["fold"]
        scanner = dataset.dataset(oof_path, format="parquet").scanner(
            columns=[*runner.KEY_COLUMNS, "label", "anomaly_type", "fold"],
            filter=dataset.field("fold") == fold,
            use_threads=True,
        )
        truth, _receipt = runner._validate_registered_holdout_membership(
            scanner.to_table().to_pandas().reset_index(drop=True),
            config,
            fold=fold,
        )
        anchor_frame, _anchor_receipt = runner._validate_registered_holdout_membership(
            runner._read_current_router_anchor_fold(
                anchor_path, fold, column=anchor_record["column"]
            ),
            config,
            fold=fold,
        )
        assert runner._keys_equal(truth, anchor_frame)
        anchor = anchor_frame[anchor_record["column"]].to_numpy(dtype=np.int8)
        truths[phase] = truth
        anchors[phase] = anchor
        holdouts[phase] = SimpleNamespace(
            surface=SimpleNamespace(keys=truth.loc[:, runner.KEY_COLUMNS], anchor=anchor)
        )
    q3_times = pd.to_datetime(truths["q3"]["time"], utc=True, format="mixed")
    q3_tail = truths["q3"].loc[
        q3_times >= pd.Timestamp("2025-09-30T15:00:00+00:00")
    ]
    assert len(q3_tail) == 119
    assert q3_tail["label"].eq(1).all()
    assert q3_tail["anomaly_type"].astype(str).str.casefold().eq("noise").all()

    identical = runner.evaluate_confirmatory_folds(
        truths, holdouts, anchors, config=config
    )
    assert identical["pooled"]["rows"] == 287862
    assert identical["pooled"]["anchor"]["tp"] == 8961
    assert identical["pooled"]["anchor"]["fp"] == 203
    assert identical["pooled"]["anchor"]["fn"] == 1724
    assert identical["pooled"]["anchor"]["f1"] == pytest.approx(17922 / 19849)
    assert identical["bootstrap"]["pooled_unique_kst_calendar_days"] == 163
    assert identical["bootstrap"]["source_fold_shared_kst_calendar_days"] == 1
    assert identical["bootstrap"]["ci90_lower"] == 0.0
    assert identical["bootstrap"]["ci90_upper"] == 0.0
    assert identical["research_result"] == "FAIL"
    assert identical["high_impact_official_probe_result"] == "FAIL"

    perfect_add_only = {
        phase: np.maximum(
            anchors[phase], truths[phase]["label"].to_numpy(dtype=np.int8)
        ).astype(np.int8)
        for phase in ("q3", "q4")
    }
    perfect = runner.evaluate_confirmatory_folds(
        truths, holdouts, perfect_add_only, config=config
    )
    assert perfect["pooled"]["anchor_positive_removed_rows"] == 0
    assert perfect["stations_improved"] == 3
    assert perfect["research_result"] == "PASS"
    assert perfect["high_impact_official_probe_result"] == "PASS"
