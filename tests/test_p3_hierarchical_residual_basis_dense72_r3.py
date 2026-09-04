from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import p3_wave.dense72_targets_r1 as target_module
import p3_wave.hierarchical_residual_basis_dense72_contract_r3 as guard
import p3_wave.hierarchical_residual_basis_dense72_execution_r3 as engine

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path.home() / "Downloads/p3/데이터셋_P3/P3_wave_forecast"


def test_scientific_structure_is_deep_equal_and_predecessor_bytes_are_immutable() -> None:
    config, predecessor = guard.load_canonical_config(ROOT)
    keys = config["scientific_structure"]["deep_equal_keys"]
    surface = {key: predecessor[key] for key in keys}
    assert hashlib.sha256(guard.canonical_json_bytes(surface)).hexdigest() == config[
        "scientific_structure"
    ]["deep_sha256"]
    assert config["scientific_structure"][
        "objective_folds_prefixes_seeds_gates_changed"
    ] is False
    for expected in config["predecessor_r2"]["files"].values():
        path = ROOT / expected["path"]
        assert path.stat().st_size == expected["bytes"]
        assert guard.sha256_file(path) == expected["sha256"]


def test_target_free_fold_ids_match_frozen_historical_surface() -> None:
    config, predecessor = guard.load_canonical_config(ROOT)
    predecessor_paths, _snapshot = guard.predecessor_guard.verify_input_pins(
        ROOT, DATA_DIR, predecessor
    )
    anchors = pd.read_parquet(
        predecessor_paths["compact_cache/train_anchors.parquet"],
        columns=["anchor_id", "station", "anchor_time"],
    )
    validation = pd.read_parquet(predecessor_paths["gen4/validation_keys.parquet"])
    folds, selected, audit = guard.build_target_free_folds(
        anchors, validation, predecessor, config
    )
    assert tuple(fold.name for fold in folds) == guard.FOLD_ORDER
    assert len(selected) == 181
    assert audit["r3_train_wave_hs_float_decodes"] == 0
    assert audit["historical_label_derived"] is True
    assert all(
        audit["folds"][fold.name]["removed_same_episode_train_anchors"] == 0
        for fold in folds
    )


def test_full_preflight_poison_rejects_any_train_wave_hs_float_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_csv = pd.read_csv

    def guarded_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        path = Path(str(args[0] if args else kwargs.get("filepath_or_buffer", "")))
        usecols = kwargs.get("usecols")
        if path.name == "train_wave.csv" and usecols is not None and "hs" in set(usecols):
            raise AssertionError("full-preflight poison: train_wave hs decode attempted")
        return original_read_csv(*args, **kwargs)

    def poisoned_decode(*_args: object, **_kwargs: object) -> dict[int, float]:
        raise AssertionError("full-preflight poison: selective scalar decode attempted")

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr(target_module.Dense72TargetAccessor, "_decode_rows", poisoned_decode)
    monkeypatch.setattr(guard.SelectiveCurrentHsAccessor, "_decode_rows", poisoned_decode)
    config, preflight = guard.prepare_execution_preflight(ROOT, DATA_DIR)
    assert config["target_free_split"]["r3_recomputes_episode_or_split_from_hs"] is False
    assert preflight["summary"]["process_train_wave_hs_float_decodes"] == 0
    assert preflight["target_accessor"].total_scalar_decodes == 0
    assert preflight["target_accessor"].released_groups == ()
    assert preflight["current_hs_accessor"].total_scalar_decodes == 0
    assert preflight["current_hs_accessor"].released_groups == ()


def _write_current_isolation_wave(
    directory: Path,
    anchors: pd.DataFrame,
    *,
    validation_current_offset: float,
) -> Path:
    directory.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    validation_ids = {2, 3}
    for anchor in anchors.itertuples(index=False):
        for step in range(73):
            value = 1.0 + 0.01 * step + 0.1 * int(anchor.anchor_id)
            if step == 0 and int(anchor.anchor_id) in validation_ids:
                value += validation_current_offset
            rows.append(
                {
                    "station": anchor.station,
                    "time": (
                        pd.Timestamp(anchor.anchor_time) + pd.Timedelta(minutes=20 * step)
                    )
                    .tz_convert("Asia/Seoul")
                    .isoformat(),
                    "hs": value,
                    "tp": 7.0,
                    "hmax": 2.0,
                    "wvdir": 180.0,
                }
            )
    path = directory / "train_wave.csv"
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    return path


def _current_isolation_accessors(
    path: Path, anchors: pd.DataFrame
) -> tuple[target_module.Dense72TargetAccessor, guard.SelectiveCurrentHsAccessor]:
    dense_anchors = anchors.copy()
    dense_anchors["current_hs"] = 0.0
    groups = {"fold_a": np.asarray([2]), "fold_b": np.asarray([3])}
    dense = target_module.Dense72TargetAccessor(
        path,
        dense_anchors,
        validation_groups=groups,
        expected_source_sha256=guard.sha256_file(path),
        expected_source_bytes=path.stat().st_size,
        expected_source_rows=4 * 73,
        enforce_canonical_aggregate=False,
    )
    current = guard.SelectiveCurrentHsAccessor(
        dense,
        anchors.loc[:, ["anchor_id", "station", "anchor_time"]],
        validation_groups=groups,
    )
    return dense, current


def test_current_hs_validation_poison_isolated_until_raw_fold_release(
    tmp_path: Path,
) -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    anchors = pd.DataFrame(
        {
            "anchor_id": np.arange(4, dtype=np.int64),
            "station": ["G-ORS", "I-ORS", "S-ORS", "G-ORS"],
            "anchor_time": [
                start + pd.Timedelta(hours=100 * index) for index in range(4)
            ],
        }
    )
    _dense_a, first = _current_isolation_accessors(
        _write_current_isolation_wave(
            tmp_path / "first", anchors, validation_current_offset=0.0
        ),
        anchors,
    )
    _dense_b, second = _current_isolation_accessors(
        _write_current_isolation_wave(
            tmp_path / "second", anchors, validation_current_offset=1_000_000.0
        ),
        anchors,
    )
    train_ids = np.asarray([0, 1], dtype=np.int64)
    active = np.asarray([2], dtype=np.int64)
    first.assert_training_target_current_isolation(train_ids)
    second.assert_training_target_current_isolation(train_ids)
    np.testing.assert_array_equal(
        first.load_training_current_hs(train_ids, active_validation_case_ids=active),
        second.load_training_current_hs(train_ids, active_validation_case_ids=active),
    )
    assert first.validation_group_scalar_decodes("fold_a") == 0
    assert second.validation_group_scalar_decodes("fold_a") == 0
    assert first.validation_group_process_scalar_decodes("fold_a") == 0
    assert second.validation_group_process_scalar_decodes("fold_a") == 0
    with pytest.raises(PermissionError, match="unreleased validation"):
        second.load_training_current_hs(
            np.asarray([2], dtype=np.int64),
            active_validation_case_ids=np.asarray([3], dtype=np.int64),
        )
    assert second.forbidden_scalar_decodes == 1

    commitment_sha = hashlib.sha256(b"raw-fold-a-commitment").hexdigest()
    first.release_validation_group("fold_a", fold_commitment_sha256=commitment_sha)
    second.release_validation_group("fold_a", fold_commitment_sha256=commitment_sha)
    first_validation = first.load_released_validation_current_hs("fold_a")
    second_validation = second.load_released_validation_current_hs("fold_a")
    assert first.validation_group_scalar_decodes("fold_a") == 1
    assert second.validation_group_scalar_decodes("fold_a") == 1
    assert first.validation_group_process_scalar_decodes("fold_a") == 1
    assert second.validation_group_process_scalar_decodes("fold_a") == 1
    assert float(second_validation[0] - first_validation[0]) == pytest.approx(1_000_000.0)


def test_exclusive_writer_retries_partial_writes_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.bin"
    payload = bytes(range(251)) * 17
    calls: list[int] = []

    def partial_write(descriptor: int, remaining: memoryview) -> int:
        calls.append(len(remaining))
        return os.write(descriptor, remaining[: min(13, len(remaining))])

    engine._write_all_exclusive(path, payload, write_fn=partial_write)
    assert len(calls) > 1
    assert path.read_bytes() == payload
    original_sha = guard.sha256_file(path)
    with pytest.raises(FileExistsError):
        engine._write_all_exclusive(path, b"replacement")
    assert guard.sha256_file(path) == original_sha


def test_parquet_is_serialized_before_o_excl_and_duplicate_is_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frame.parquet"
    frame = pd.DataFrame({"anchor_id": [1, 2], "value": [0.5, 1.5]})
    first_sha = engine._write_parquet_exclusive(path, frame)
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    with pytest.raises(FileExistsError):
        engine._write_parquet_exclusive(path, frame.assign(value=[9.0, 9.0]))
    assert guard.sha256_file(path) == first_sha


def test_parquet_o_excl_path_survives_short_os_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "short-write.parquet"
    frame = pd.DataFrame({"anchor_id": np.arange(20), "value": np.linspace(0.0, 1.0, 20)})
    original = engine._write_all_exclusive
    short_write_calls = 0

    def short_write(descriptor: int, remaining: memoryview) -> int:
        nonlocal short_write_calls
        short_write_calls += 1
        return os.write(descriptor, remaining[: min(17, len(remaining))])

    def force_short_writes(target: Path, payload: bytes) -> None:
        original(target, payload, write_fn=short_write)

    monkeypatch.setattr(engine, "_write_all_exclusive", force_short_writes)
    parquet_sha = engine._write_parquet_exclusive(path, frame)
    assert short_write_calls > 1
    assert guard.sha256_file(path) == parquet_sha
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
