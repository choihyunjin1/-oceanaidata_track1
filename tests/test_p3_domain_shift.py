from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.domain_shift import (
    ALLOWED_P3_TRAIN_FILES,
    COMMON_COLUMNS,
    ClassifierEvaluation,
    DomainShiftError,
    SampledRepresentation,
    StationRobustNormalizer,
    _stable_sample_indices,
    assert_allowed_input,
    build_causal_features,
    density_ratio_summary,
    evaluate_domain_classifier,
    gate_decision,
    load_target_train_canonical,
    resolve_p3_train_paths,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p3_kma_domain_shift_gate_v1.json"
SCRIPT_PATH = ROOT / "scripts/run_p3_kma_domain_shift_gate.py"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _common_grid(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="30min", tz="Asia/Seoul")
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "hs": 2.0 + 0.2 * np.sin(phase / 9),
            "tp": 7.0 + 0.1 * np.cos(phase / 11),
            "hmax": 3.2 + 0.3 * np.sin(phase / 9),
            "wvdir": (180 + phase) % 360,
            "wspd": 8.0 + np.sin(phase / 7),
            "gust": 10.0 + np.sin(phase / 7),
            "wdir": (220 + phase * 2) % 360,
            "caph": 1010.0 - phase / 100,
            "airt": 15.0 + np.sin(phase / 20),
            "relh": 70.0 + np.cos(phase / 13),
        },
        index=index,
    )


def test_preregistered_safety_contract_is_fixed() -> None:
    config = _config()
    assert config["seed"] == 20260821
    assert tuple(config["inputs"]["allowed_p3_train_files"]) == ALLOWED_P3_TRAIN_FILES
    assert config["representation"]["history_windows_hours"] == [3, 6, 12, 24, 48]
    assert config["classifier"]["n_splits"] == 5
    assert config["gate"]["full_ablation_max_auc"] == 0.65
    assert config["gate"]["pretrain_only_max_auc"] == 0.8
    assert all(value is False for value in config["prohibitions"].values())


def test_causal_features_do_not_change_before_a_future_mutation() -> None:
    representation = _config()["representation"]
    original = _common_grid()
    baseline = build_causal_features(original, representation)
    mutated = original.copy()
    mutation_row = 180
    mutated.iloc[mutation_row, :] = mutated.iloc[mutation_row, :] + 100.0
    challenger = build_causal_features(mutated, representation)
    np.testing.assert_allclose(
        baseline.iloc[:mutation_row].to_numpy(),
        challenger.iloc[:mutation_row].to_numpy(),
        equal_nan=True,
    )
    assert not np.allclose(
        baseline.iloc[mutation_row].to_numpy(),
        challenger.iloc[mutation_row].to_numpy(),
        equal_nan=True,
    )


def test_deterministic_group_sampling_is_seeded_and_stable() -> None:
    first = _stable_sample_indices(100, 10, seed=20260821, group_key="source|a|2020")
    second = _stable_sample_indices(100, 10, seed=20260821, group_key="source|a|2020")
    other = _stable_sample_indices(100, 10, seed=20260821, group_key="source|a|2021")
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 10
    assert np.all(np.diff(first) > 0)
    assert not np.array_equal(first, other)


@pytest.mark.parametrize(
    ("auc", "tier", "concat_allowed", "pretrain_allowed"),
    [
        (0.65, "full_ablation_allowed", True, True),
        (0.650001, "pretrain_only_challenger", False, True),
        (0.8, "pretrain_only_challenger", False, True),
        (0.800001, "no_go_source_concat_or_full_finetune", False, False),
    ],
)
def test_gate_threshold_boundaries(
    auc: float, tier: str, concat_allowed: bool, pretrain_allowed: bool
) -> None:
    decision = gate_decision(auc, _config()["gate"])
    assert decision["tier"] == tier
    assert decision["source_concat_or_full_finetune_allowed"] is concat_allowed
    assert decision["pretrain_only_challenger_allowed"] is pretrain_allowed


def test_station_normalizer_is_fit_only_from_passed_training_rows() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, np.nan, 14.0]})
    stations = np.array(["source|a"] * 3, dtype=object)
    holdout = pd.DataFrame({"a": [1_000_000.0], "b": [-1_000_000.0]})
    holdout_station = np.array(["target|unseen"], dtype=object)
    normalizer = StationRobustNormalizer(iqr_floor=1e-6, clip=20.0).fit(train, stations)
    before = normalizer.transform(train, stations)
    transformed_holdout = normalizer.transform(holdout, holdout_station)
    after = normalizer.transform(train, stations)
    np.testing.assert_array_equal(before, after)
    assert np.isfinite(transformed_holdout).all()
    assert np.abs(transformed_holdout).max() <= 20.0


def test_grouped_classifier_has_no_station_year_overlap() -> None:
    rng = np.random.default_rng(20260821)
    feature_parts = []
    labels = []
    stations = []
    groups = []
    for domain in (0, 1):
        for group_index in range(6):
            rows = 20
            feature_parts.append(
                pd.DataFrame(
                    {
                        "f1": rng.normal(domain * 0.8, 1.0, rows),
                        "f2": rng.normal(domain * -0.5, 1.0, rows),
                    }
                )
            )
            labels.extend([domain] * rows)
            stations.extend([f"d{domain}|s{group_index % 3}"] * rows)
            groups.extend([f"d{domain}|s{group_index % 3}|y{group_index}"] * rows)
    sampled = SampledRepresentation(
        features=pd.concat(feature_parts, ignore_index=True),
        domain=np.asarray(labels, dtype=np.uint8),
        station_keys=np.asarray(stations, dtype=object),
        group_keys=np.asarray(groups, dtype=object),
        summary={},
    )
    classifier = dict(_config()["classifier"])
    classifier["max_iter"] = 5
    classifier["min_samples_leaf"] = 5
    evaluation = evaluate_domain_classifier(
        sampled,
        classifier,
        _config()["representation"],
        seed=20260821,
    )
    assert isinstance(evaluation, ClassifierEvaluation)
    assert np.isfinite(evaluation.auc)
    assert len(evaluation.fold_auc) == 5
    assert all(fold["group_overlap_count"] == 0 for fold in evaluation.fold_summaries)
    assert np.isfinite(evaluation.probabilities).all()


def test_density_ratio_ess_is_bounded() -> None:
    labels = np.array([0, 0, 0, 1, 1], dtype=np.uint8)
    probabilities = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    summary = density_ratio_summary(labels, probabilities, _config()["gate"])
    assert 0 < summary["effective_sample_size"] <= 3
    assert 0 < summary["effective_sample_fraction"] <= 1


@pytest.mark.parametrize("name", sorted(_config()["inputs"]["forbidden_p3_files"]))
def test_all_forbidden_p3_inputs_fail_closed(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.touch()
    with pytest.raises(DomainShiftError, match="forbidden"):
        assert_allowed_input(path, role="p3_train")


def test_p3_path_resolver_and_loader_read_only_two_train_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wave = pd.DataFrame(
        {
            "station": ["G-ORS"],
            "time": ["2024-01-01T00:00:00+09:00"],
            "hs": [1.5],
            "tp": [7.0],
            "hmax": [2.5],
            "wvdir": [180.0],
        }
    )
    atmos = pd.DataFrame(
        {
            "station": ["G-ORS"],
            "time": ["2024-01-01T00:00:00+09:00"],
            "wspd": [8.0],
            "gust": [10.0],
            "wdir": [200.0],
            "airt": [12.0],
            "relh": [70.0],
            "caph": [1010.0],
        }
    )
    wave.to_csv(tmp_path / "train_wave.csv", index=False)
    atmos.to_csv(tmp_path / "train_atmos.csv", index=False)
    for forbidden in _config()["inputs"]["forbidden_p3_files"]:
        (tmp_path / forbidden).touch()

    paths = resolve_p3_train_paths(tmp_path)
    observed: list[str] = []
    real_read_csv = pd.read_csv

    def read_spy(path: Path, *args, **kwargs):
        observed.append(Path(path).name)
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_spy)
    canonical = load_target_train_canonical(paths)
    assert observed == ["train_wave.csv", "train_atmos.csv"]
    assert tuple(canonical.columns) == ("station", "time", *COMMON_COLUMNS)
    assert not set(observed).intersection(_config()["inputs"]["forbidden_p3_files"])


def test_status_gauge_contains_only_aggregate_progress_fields(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("p3_domain_shift_runner", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "status.json"
    module.Gauge(path).update("unit_test", 0.5)
    status = json.loads(path.read_text(encoding="utf-8"))
    assert set(status) == {"status", "phase", "progress", "elapsed_seconds", "eta_seconds"}
    serialized = path.read_text(encoding="utf-8").casefold()
    assert "secret" not in serialized
    assert "path" not in serialized
    assert "row" not in serialized
