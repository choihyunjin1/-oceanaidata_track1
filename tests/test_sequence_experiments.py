from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from p1_qc.deep_training import robust_fit
from p1_qc.features import FeatureBundle
from p1_qc.sequence_experiments import (
    SequenceExperimentConfig,
    _append_ssl_embeddings,
    _Partition,
    build_inner_fold,
    run_sequence_experiments,
    search_space,
    select_threshold,
)
from p1_qc.splits import Fold


def _fixture() -> tuple[pd.DataFrame, FeatureBundle, Fold]:
    rows = 4 * 24 * 6
    time = pd.date_range("2025-01-01", periods=rows, freq="10min", tz="Asia/Seoul")
    labels = np.zeros(rows, dtype=np.int8)
    for start in (30, 180, 330, 474):
        labels[start : start + 8] = 1
    frame = pd.DataFrame(
        {
            "station": "S-ORS",
            "year": 2025,
            "layer": 1,
            "time": time.astype(str),
            "temp": 15.0 + np.sin(np.arange(rows) / 50),
            "psal": 32.0,
            "depth": 5.0,
            "label": labels,
            "anomaly_type": np.where(labels == 1, "flatline", ""),
        }
    )
    feature_frame = pd.DataFrame(
        {
            "signal": labels.astype(np.float32),
            "row_id": np.arange(rows, dtype=np.float32),
        },
        index=frame.index,
    )
    feature_frame.attrs["feature_mode"] = "offline"
    bundle = FeatureBundle(
        frame=feature_frame,
        feature_columns=("signal", "row_id"),
        categorical_columns=(),
    )
    train_stop = 3 * 24 * 6
    outer = Fold(
        name="smoke_outer",
        train_idx=np.arange(train_stop),
        val_idx=np.arange(train_stop, rows),
        train_end=pd.Timestamp(time[train_stop - 1]).tz_convert("UTC"),
        val_start=pd.Timestamp(time[train_stop]).tz_convert("UTC"),
        val_end=pd.Timestamp(time[-1]).tz_convert("UTC") + pd.Timedelta(minutes=10),
    )
    return frame, bundle, outer


def _fake_trainer(*args, **kwargs):
    train_features = np.asarray(args[0], dtype=np.float32)
    center, scale = robust_fit(train_features)
    return SimpleNamespace(
        model=object(),
        history=[{"epoch": 0.0, "validation_loss": 0.1}],
        best_epoch=0,
        best_validation_loss=0.1,
        center=center,
        scale=scale,
        config={"training": vars(kwargs["config"])},
    )


def _fake_predictor(_result, features, _segments, **_kwargs):
    signal = np.asarray(features)[:, 0]
    return np.where(signal > 0.5, 0.9, 0.1).astype(np.float32), None


def _fake_checkpoint_saver(_result, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"fold-local-test-checkpoint")
    return destination


@pytest.mark.parametrize("architecture", ["tcn", "patch_transformer"])
def test_predefined_search_spaces_have_twelve_unique_settings(architecture: str) -> None:
    settings = search_space(architecture, causal=False)
    assert len(settings) == 12
    assert len({json.dumps(item, sort_keys=True) for item in settings}) == 12
    assert all(item["causal"] is False for item in settings)


def test_inner_fold_is_past_only_purged_and_outer_disjoint() -> None:
    frame, _, outer = _fixture()
    inner = build_inner_fold(frame, outer, validation_days=1, purge_days=0)

    assert np.isin(inner.train_idx, outer.train_idx).all()
    assert np.isin(inner.val_idx, outer.train_idx).all()
    assert not np.intersect1d(inner.train_idx, inner.val_idx).size
    assert not np.intersect1d(np.r_[inner.train_idx, inner.val_idx], outer.val_idx).size
    time = pd.to_datetime(frame["time"], utc=True)
    assert time.iloc[inner.train_idx].max() <= inner.train_end
    assert time.iloc[inner.val_idx].min() >= inner.val_start


def test_threshold_selection_is_finite_and_deterministic() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    probability = np.asarray([0.1, 0.3, 0.6, 0.9])
    assert select_threshold(labels, probability, (0.4, 0.5, 0.7)) == (0.5, 1.0)
    with pytest.raises(ValueError, match="finite"):
        select_threshold(labels, np.asarray([0.1, np.nan, 0.6, 0.9]), (0.5,))


def test_smoke_orchestrator_persists_sanitized_contract_without_outer_leakage(
    tmp_path: Path,
) -> None:
    frame, bundle, outer = _fixture()
    trainer_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def recording_trainer(*args, **kwargs):
        trainer_calls.append((np.asarray(args[0])[:, 1].copy(), np.asarray(args[4])[:, 1].copy()))
        return _fake_trainer(*args, **kwargs)

    experiment = SequenceExperimentConfig(
        architectures=("tcn",),
        inner_validation_days=1,
        purge_days=0,
        window_steps=16,
        stride_steps=8,
        batch_size=4,
        final_seeds=(7,),
        threshold_grid=(0.3, 0.5, 0.7),
        smoke=True,
    )
    result = run_sequence_experiments(
        frame,
        bundle,
        experiment_config=experiment,
        folds=(outer,),
        output_dir=tmp_path,
        device="cpu",
        trainer=recording_trainer,
        predictor=_fake_predictor,
        checkpoint_saver=_fake_checkpoint_saver,
    )

    outer_ids = set(outer.val_idx.tolist())
    assert trainer_calls
    for train_ids, validation_ids in trainer_calls:
        assert not outer_ids.intersection(train_ids.astype(int).tolist())
        assert not outer_ids.intersection(validation_ids.astype(int).tolist())
    assert result.selection["architecture"] == "tcn"
    assert result.comparison[0]["selection_mean_inner_f1"] == pytest.approx(1.0)
    assert result.comparison[0]["diagnostic_outer_micro_f1"] == pytest.approx(1.0)
    assert (tmp_path / "oof_tcn.npz").is_file()
    contract = json.loads((tmp_path / "sequence_experiment.json").read_text(encoding="utf-8"))
    assert contract["contract_version"] == 1
    assert contract["experiment"]["selection_uses_outer_labels"] is False
    assert contract["selection"]["outer_metrics_used_for_selection"] is False
    assert contract["selection"]["configuration_index"] == 0
    assert contract["selection"]["seeds"] == [7]
    assert contract["selection"]["threshold_median"] == pytest.approx(0.5)
    assert len(contract["checkpoints"]) == 1
    assert Path(contract["checkpoints"][0]["path"]).is_file()


def test_optional_ssl_uses_only_fold_local_normal_rows() -> None:
    frame, bundle, outer = _fixture()
    captured: dict[str, np.ndarray] = {}

    def fake_ssl_trainer(features, _segments, row_ids, **kwargs):
        captured["train"] = np.asarray(row_ids).copy()
        captured["validation"] = np.asarray(kwargs["validation_row_ids"]).copy()
        return SimpleNamespace(
            train_row_ids=np.asarray(row_ids).copy(),
            best_epoch=0,
            best_validation_loss=0.1,
        )

    def fake_ssl_extractor(_result, features, _segments, **_kwargs):
        return np.zeros((len(features), 2), dtype=np.float32)

    experiment = SequenceExperimentConfig(
        architectures=("tcn",),
        inner_validation_days=1,
        purge_days=0,
        window_steps=16,
        stride_steps=8,
        batch_size=4,
        final_seeds=(7,),
        threshold_grid=(0.5,),
        use_ssl=True,
        ssl_window_steps=16,
        ssl_stride_steps=8,
        smoke=True,
    )
    result = run_sequence_experiments(
        frame,
        bundle,
        experiment_config=experiment,
        folds=(outer,),
        device="cpu",
        trainer=_fake_trainer,
        predictor=_fake_predictor,
        ssl_trainer=fake_ssl_trainer,
        ssl_extractor=fake_ssl_extractor,
    )

    assert not np.intersect1d(captured["train"], captured["validation"]).size
    assert not np.intersect1d(captured["train"], outer.val_idx).size
    assert (frame.iloc[captured["train"]]["label"] == 0).all()
    assert (frame.iloc[captured["validation"]]["label"] == 0).all()
    assert result.folds[0].ssl is not None
    assert result.folds[0].ssl["embedding_dim"] == 2


def test_ssl_embedding_fallback_never_bridges_singleton_segments() -> None:
    partition = _Partition(
        indices=np.asarray([10, 20]),
        features=np.asarray([[1.0], [2.0]], dtype=np.float32),
        labels=np.zeros(2, dtype=np.int8),
        auxiliary=np.zeros((2, 5), dtype=np.float32),
        segments=np.asarray([3, 4]),
        metadata=pd.DataFrame(
            {
                "station": ["A", "A"],
                "layer": [1, 1],
                "time": ["2025-01-01T00:00:00+09:00", "2025-01-02T00:00:00+09:00"],
            }
        ),
    )
    observed_segments: list[np.ndarray] = []

    def extractor(_result, features, segments, **_kwargs):
        observed_segments.append(np.asarray(segments).copy())
        return np.column_stack([features[:, 0], features[:, 0]])

    result = _append_ssl_embeddings(
        partition,
        ssl_result=object(),
        center=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        window_steps=16,
        stride_steps=8,
        batch_size=2,
        device="cpu",
        extractor=extractor,
    )

    assert observed_segments[0].tolist() == [0, 0, 1, 1]
    assert result.features.shape == (2, 3)
    assert result.features[:, 1:].tolist() == [[1.0, 1.0], [2.0, 2.0]]


def test_normal_mode_rejects_non_fixed_outer_fold_contract() -> None:
    frame, bundle, outer = _fixture()
    experiment = SequenceExperimentConfig(
        architectures=("tcn",),
        inner_validation_days=1,
        purge_days=7,
        final_seeds=(7,),
    )
    with pytest.raises(ValueError, match="fixed outer folds"):
        run_sequence_experiments(
            frame,
            bundle,
            experiment_config=experiment,
            folds=(outer,),
            trainer=_fake_trainer,
            predictor=_fake_predictor,
        )
