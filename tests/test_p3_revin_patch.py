from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from p3_wave.data import audit_p3_data, load_p3_data
from p3_wave.revin_patch import (
    PatchModelConfig,
    TwoStreamRevINPatchTransformer,
    assign_storm_episodes,
    assign_storm_episodes_from_wave,
    blend_long_leads,
    bounded_cpu_backward_smoke,
    bounded_training_protocol_smoke,
    build_episode_disjoint_folds,
    build_episode_disjoint_folds_from_ids,
    build_inner_episode_split,
    build_synthetic_context,
    event_balanced_weights,
    extract_past_context,
    prepare_streams,
    validate_preregistration,
    validate_raw_context,
    weighted_official_mse,
)
from p3_wave.sequences import build_test_sequences
from scripts.run_p3_revin_patch_v1 import (
    TargetVault,
    _atomic_json,
    _atomic_parquet,
    _seal_blind_prediction_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p3_revin_patch_v1.json"


def test_native_stream_shapes_masks_and_anchor_delta() -> None:
    raw = build_synthetic_context(batch=3)
    streams = prepare_streams(raw)
    assert streams.wave.shape == (3, 145, 10)
    assert streams.atmos.shape == (3, 289, 14)
    assert streams.current_hs.shape == (3,)
    assert streams.hs_scale.shape == (3,)
    assert torch.all(streams.hs_scale >= 0.05)
    assert torch.allclose(streams.wave[:, -1, 0], torch.zeros(3), atol=1e-7)
    # The first five wave channels are normalized values and sin/cos; the next four are masks.
    assert set(torch.unique(streams.wave[:, :, 5:9]).tolist()).issubset({0.0, 1.0})
    # The first seven atmosphere channels are normalized values and sin/cos; six masks follow.
    assert set(torch.unique(streams.atmos[:, :, 7:13]).tolist()).issubset({0.0, 1.0})


def test_future_rows_cannot_change_extracted_48h_context() -> None:
    raw = build_synthetic_context(batch=1)[0].numpy()
    prefix = np.repeat(raw[:1], 50, axis=0)
    full = np.concatenate([prefix, raw, np.repeat(raw[-1:], 40, axis=0)], axis=0)
    anchor = len(prefix) + len(raw) - 1
    first = extract_past_context(full, anchor)
    full[anchor + 1 :] = 9999.0
    second = extract_past_context(full, anchor)
    assert np.array_equal(first, second, equal_nan=True)


def test_model_is_two_stream_47_patch_six_lead_and_backward_finite() -> None:
    torch.manual_seed(20260821)
    model = TwoStreamRevINPatchTransformer()
    raw = build_synthetic_context(batch=4)
    station = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    prediction = model(raw, station)
    assert prediction.shape == (4, 6)
    assert torch.isfinite(prediction).all()
    assert model.config.wave_patch_count == 47
    assert model.config.atmos_patch_count == 47
    target = torch.zeros_like(prediction)
    loss = weighted_official_mse(prediction, target, torch.ones(4))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_structural_wave_rows_are_enforced() -> None:
    raw = build_synthetic_context(batch=1)
    raw[0, 1, 0] = 2.0
    with pytest.raises(ValueError, match="structural 10-minute"):
        validate_raw_context(raw)


def test_blend_protects_3_6_9_bit_exactly() -> None:
    rng = np.random.default_rng(20260821)
    incumbent = rng.normal(2.0, 0.2, size=(10, 6))
    patch = rng.normal(2.1, 0.3, size=(10, 6))
    output = blend_long_leads(incumbent, patch)
    assert np.array_equal(output[:, :3], incumbent[:, :3])
    assert np.allclose(output[:, 3:], 0.8 * incumbent[:, 3:] + 0.2 * patch[:, 3:])
    with pytest.raises(ValueError, match="frozen at 0.2"):
        blend_long_leads(incumbent, patch, patch_weight=0.3)


def _split_anchors() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    anchor_id = 0
    for station in ("G-ORS", "I-ORS", "S-ORS"):
        for episode_start in (
            "2024-01-01",
            "2024-02-01",
            "2024-07-01",
            "2024-07-06",
            "2024-07-12",
        ):
            for offset in range(4):
                records.append(
                    {
                        "anchor_id": anchor_id,
                        "station": station,
                        "anchor_time": pd.Timestamp(episode_start, tz="UTC")
                        + pd.Timedelta(minutes=20 * offset),
                        "current_hs": 2.0,
                    }
                )
                anchor_id += 1
    return assign_storm_episodes(pd.DataFrame(records))


def test_event_weight_and_episode_78h_split_are_disjoint() -> None:
    anchors = _split_anchors()
    first_episode = anchors.loc[anchors["episode_id"].eq(0), "anchor_id"].to_numpy()
    singleton = anchors.loc[anchors["episode_id"].eq(1), "anchor_id"].to_numpy()[:1]
    selected = np.concatenate([first_episode, singleton])
    weight = event_balanced_weights(anchors, selected)
    assert weight.mean() == pytest.approx(1.0)
    assert weight[-1] / weight[0] == pytest.approx(2.0)

    folds = build_episode_disjoint_folds(
        anchors,
        windows=(("synthetic", "2024-07-01", "2024-08-01"),),
        embargo_hours=78,
    )
    fold = folds[0]
    assert len(fold.train_ids) > 0
    assert len(fold.validation_ids) == 9
    lookup = anchors.set_index("anchor_id")
    train = lookup.loc[fold.train_ids]
    validation = lookup.loc[fold.validation_ids]
    train_episode = set(zip(train["station"], train["episode_id"], strict=True))
    validation_episode = set(zip(validation["station"], validation["episode_id"], strict=True))
    assert train_episode.isdisjoint(validation_episode)
    for _, group in validation.groupby("station", observed=True):
        assert group["anchor_time"].sort_values().diff().dropna().ge(pd.Timedelta(hours=78)).all()


def test_frozen_outer_ids_are_preserved_exactly_without_target_columns() -> None:
    anchors = _split_anchors()
    validation = anchors.loc[
        anchors["anchor_time"].isin(
            [pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2024-07-06", tz="UTC")]
        ),
        "anchor_id",
    ].to_numpy(dtype=np.int64)
    folds = build_episode_disjoint_folds_from_ids(
        anchors,
        windows=(("frozen", "2024-07-01", "2024-08-01"),),
        validation_ids_by_fold={"frozen": validation},
    )
    assert np.array_equal(folds[0].validation_ids, np.sort(validation))
    assert np.intersect1d(folds[0].train_ids, validation).size == 0


def test_inner_epoch_split_never_escapes_outer_train() -> None:
    anchors = _split_anchors()
    outer_ids = anchors["anchor_id"].to_numpy(dtype=np.int64)
    # Build a longer synthetic calendar because the preregistered inner window is 45 days.
    extended = pd.concat(
        [
            anchors,
            anchors.assign(
                anchor_id=lambda frame: frame["anchor_id"] + len(anchors),
                anchor_time=lambda frame: frame["anchor_time"] + pd.Timedelta(days=180),
            ),
        ],
        ignore_index=True,
    )
    extended = assign_storm_episodes(
        extended.sort_values(["station", "anchor_time"]).reset_index(drop=True)
    )
    outer_ids = extended["anchor_id"].to_numpy(dtype=np.int64)
    split = build_inner_episode_split(extended, outer_ids)
    assert np.isin(split.train_ids, outer_ids).all()
    assert np.isin(split.validation_ids, outer_ids).all()
    assert np.intersect1d(split.train_ids, split.validation_ids).size == 0


def test_episode_definition_uses_raw_wave_not_eligible_anchor_gaps() -> None:
    time = pd.date_range("2025-01-01", periods=5, freq="20min", tz="UTC")
    wave = pd.DataFrame(
        {
            "station": ["S-ORS"] * 5,
            "time": time,
            "hs": [1.6, 1.7, 1.8, 1.9, 1.0],
        }
    )
    anchors = pd.DataFrame(
        {
            "anchor_id": [0, 1],
            "station": ["S-ORS", "S-ORS"],
            "anchor_time": [time[0], time[2]],
            "current_hs": [1.6, 1.8],
        }
    )
    mapped = assign_storm_episodes_from_wave(anchors, wave)
    assert mapped["episode_id"].nunique() == 1


def test_preregistration_locks_architecture_auxiliary_and_phase_prohibition() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = validate_preregistration(config, verify_frozen_files=False)
    assert result["architecture"] == {
        "wave_patches": 47,
        "atmos_patches": 47,
        "d_model": 64,
        "encoder_layers": 2,
    }
    assert result["outer_validation_labels_opened"] is False

    dense = copy.deepcopy(config)
    dense["loss"]["dense_path_auxiliary_weight"] = 0.1
    with pytest.raises(ValueError, match="dense auxiliary"):
        validate_preregistration(dense, verify_frozen_files=False)

    phase = copy.deepcopy(config)
    phase["validation"]["final_holdout_lattice_phase_hours"] = 39
    with pytest.raises(ValueError, match="phase-shift lattice fields"):
        validate_preregistration(phase, verify_frozen_files=False)

    seed = copy.deepcopy(config)
    seed["training"]["fixed_seeds"][-1] = 7
    with pytest.raises(ValueError, match="training.fixed_seeds"):
        validate_preregistration(seed, verify_frozen_files=False)

    gate = copy.deepcopy(config)
    gate["gate"]["maximum_candidate_rmse"] = 0.8
    with pytest.raises(ValueError, match="gate candidate"):
        validate_preregistration(gate, verify_frozen_files=False)


def test_bounded_cpu_smoke_does_not_create_checkpoint() -> None:
    before = set(ROOT.rglob("p3_revin_patch_v1*.pt"))
    result = bounded_cpu_backward_smoke()
    after = set(ROOT.rglob("p3_revin_patch_v1*.pt"))
    assert result["prediction_shape"] == [4, 6]
    assert result["wave_patch_count"] == 47
    assert result["atmos_patch_count"] == 47
    assert result["dense_72_step_auxiliary"] is False
    assert before == after


def test_bounded_inner_epoch_and_fixed_refit_protocol() -> None:
    before = set(ROOT.rglob("p3_revin_patch_v1*.pt"))
    result = bounded_training_protocol_smoke()
    after = set(ROOT.rglob("p3_revin_patch_v1*.pt"))
    assert result["selected_epoch"] in {1, 2}
    assert result["prediction_shape"] == [6, 6]
    assert result["prediction_finite"] is True
    assert result["outer_labels_opened"] is False
    assert before == after


def test_outer_target_vault_requires_fsynced_nine_file_manifest(tmp_path: Path) -> None:
    target_path = tmp_path / "targets.parquet"
    targets = pd.DataFrame({"anchor_id": np.arange(12, dtype=np.int64)})
    for lead in (3, 6, 9, 12, 18, 24):
        targets[f"target_{lead}"] = 2.0 + lead / 100.0
    targets.to_parquet(target_path, index=False)

    paths: list[Path] = []
    for fold_number in range(3):
        for seed_number, seed in enumerate((20260821, 20260822, 20260823)):
            anchor_id = 3 * fold_number + seed_number
            blocks = []
            for lead in (3, 6, 9, 12, 18, 24):
                blocks.append(
                    {
                        "fold": f"fold_{fold_number}",
                        "seed": seed,
                        "anchor_id": anchor_id,
                        "station": "S-ORS",
                        "episode_id": anchor_id,
                        "lead_h": lead,
                        "current_hs": 2.0,
                        "patch_prediction": 2.1,
                    }
                )
            path = tmp_path / "blind" / f"fold_{fold_number}_seed_{seed}.parquet"
            _atomic_parquet(path, pd.DataFrame(blocks))
            paths.append(path)

    manifest_path = tmp_path / "blind_prediction_manifest.json"
    manifest = _seal_blind_prediction_manifest(paths, manifest_path)
    receipt_path = tmp_path / "outer_label_exposure_receipt.json"
    _atomic_json(
        receipt_path,
        {
            "blind_prediction_manifest_sha256": manifest["manifest_sha256"],
            "outer_validation_labels_opened": False,
            "fsync_completed_before_outer_open": True,
        },
    )
    vault = TargetVault(target_path)
    vault.read_outer_train(
        np.asarray([0, 1], dtype=np.int64),
        forbidden_outer_validation_ids=np.asarray([9, 10], dtype=np.int64),
        fold="fold_0",
    )
    with pytest.raises(PermissionError, match="outer validation label"):
        vault.read_outer_train(
            np.asarray([9], dtype=np.int64),
            forbidden_outer_validation_ids=np.asarray([9, 10], dtype=np.int64),
            fold="fold_0",
        )
    opened = vault.open_outer_once(
        np.asarray([9, 10], dtype=np.int64),
        blind_manifest_path=manifest_path,
        exposure_receipt_path=receipt_path,
    )
    assert opened["anchor_id"].tolist() == [9, 10]
    assert vault.outer_open_count == 1
    with pytest.raises(PermissionError, match="exactly once"):
        vault.open_outer_once(
            np.asarray([9, 10], dtype=np.int64),
            blind_manifest_path=manifest_path,
            exposure_receipt_path=receipt_path,
        )


@pytest.mark.skipif(not os.environ.get("P3_DATA_DIR"), reason="P3_DATA_DIR not configured")
def test_real_p3_case_coverage_is_200_by_289_without_absolute_time() -> None:
    data = load_p3_data(os.environ["P3_DATA_DIR"])
    audit = audit_p3_data(data)
    sequences = build_test_sequences(data)
    raw = torch.from_numpy(sequences.values)
    validate_raw_context(raw)
    assert audit["cases"] == 200
    assert sequences.values.shape == (200, 289, 10)
    assert sequences.station_code.shape == (200,)
    assert "time" not in data.test_context.columns


def test_default_patch_config_is_exactly_preregistered() -> None:
    config = PatchModelConfig()
    config.validate()
    assert config.wave_patch_count == config.atmos_patch_count == 47
