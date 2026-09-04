from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

import p2_restore.gbm_tuning as tuning
from p2_restore.features import FeatureTable
from p2_restore.gbm_tuning import (
    TUNING_FAMILIES,
    consensus_iterations,
    fit_tuned_model,
    nested_masks,
    sample_parameters,
)


def _small_table(rows: int = 360) -> FeatureTable:
    layer = np.resize(np.array([2, 3, 4]), rows)
    x = np.linspace(-2.0, 2.0, rows)
    baseline = 19.0 + 0.1 * layer + 0.2 * x
    residual = 0.05 * layer + 0.12 * np.sin(2 * x)
    frame = pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": layer,
            "time": pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC"),
            "baseline": baseline,
            "x": x,
            "x2": x**2,
            "residual": residual,
            "target": baseline + residual,
        }
    )
    return FeatureTable(frame, ("baseline", "x", "x2"))


def _calendar_table() -> FeatureTable:
    dates = pd.date_range("2024-06-01", "2025-12-20", freq="D", tz="Asia/Seoul")
    frame = pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": np.resize([2, 3, 4], len(dates)),
            "time": dates,
            "baseline": 20.0,
            "x": np.arange(len(dates), dtype=float),
            "residual": 0.0,
            "target": 20.0,
        }
    )
    return FeatureTable(frame, ("baseline", "x"))


def test_top_three_and_preregistration_are_exact() -> None:
    contract = json.loads(
        Path("configs/experiments/p2_top3_parallel_tuning_v1.json").read_text(encoding="utf-8")
    )
    assert tuple(contract["families"]) == TUNING_FAMILIES
    assert contract["parallelism"] == {
        "workers": 3,
        "threads_per_worker": 2,
        "optuna_jobs_per_worker": 1,
        "reason": "Three independent family searches share an 8-logical-CPU host without nested oversubscription.",
    }
    assert contract["search"]["trials_per_family"] == 36
    assert contract["convergence"]["catboost_max_iterations"] == 3000
    assert contract["upload_allowed"] is False


def test_nested_masks_do_not_read_or_change_with_targets() -> None:
    table = _calendar_table()
    original = nested_masks(table)
    flipped = FeatureTable(
        table.frame.assign(
            target=table.frame["target"].to_numpy()[::-1] + 99,
            residual=table.frame["residual"].to_numpy()[::-1] - 99,
        ),
        table.feature_columns,
    )
    changed = nested_masks(flipped)
    assert original.keys() == changed.keys()
    for block in original:
        for name in original[block]:
            assert np.array_equal(original[block][name], changed[block][name])
        union = (
            original[block]["inner_fit"]
            | original[block]["inner_validation"]
            | original[block]["outer_validation"]
        )
        assert union.all()


@pytest.mark.parametrize("family", TUNING_FAMILIES)
def test_search_space_is_finite_and_contains_round_policy(family: str) -> None:
    if family.startswith("catboost"):
        values = {
            "bootstrap_type": "MVS",
            "learning_rate": 0.04,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "random_strength": 0.2,
            "rsm": 0.8,
            "leaf_estimation_iterations": 4,
            "subsample": 0.85,
        }
    else:
        values = {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 7,
            "min_child_samples": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "drop_rate": 0.1,
            "skip_drop": 0.5,
            "max_drop": 50,
        }
    parameters = sample_parameters(family, optuna.trial.FixedTrial(values))
    assert parameters
    assert all(
        np.isfinite(float(value)) for value in parameters.values() if not isinstance(value, str)
    )
    if family == "lgbm_dart":
        assert parameters["n_estimators"] == 800


@pytest.mark.parametrize("family", TUNING_FAMILIES)
def test_each_tuned_family_fits_and_predicts_finite(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _small_table()
    training = np.arange(len(table.frame)) % 5 != 0
    validation = ~training
    if family.startswith("catboost"):
        monkeypatch.setattr(tuning, "CATBOOST_MAX_ITERATIONS", 30)
        monkeypatch.setattr(tuning, "CATBOOST_EARLY_STOPPING", 5)
        parameters = {
            "learning_rate": 0.08,
            "depth": 5,
            "l2_leaf_reg": 2.0,
            "random_strength": 0.01,
            "rsm": 0.9,
            "leaf_estimation_iterations": 2,
            "bootstrap_type": "MVS",
            "subsample": 0.9,
        }
        fitted = fit_tuned_model(
            table,
            family,
            parameters,
            training,
            validation_rows=validation,
            seed=11,
            threads=1,
        )
    else:
        parameters = {
            "n_estimators": 8,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 10,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "drop_rate": 0.1,
            "skip_drop": 0.5,
            "max_drop": 20,
        }
        fitted = fit_tuned_model(
            table,
            family,
            parameters,
            training,
            seed=11,
            threads=1,
        )
    prediction = fitted.model.predict(table)
    assert prediction.shape == (len(table.frame),)
    assert np.isfinite(prediction).all()


def test_iteration_consensus_is_layerwise_or_pooled_median() -> None:
    layerwise = consensus_iterations(
        "catboost_layerwise",
        [
            {"2": 100, "3": 200, "4": 300},
            {"2": 300, "3": 100, "4": 200},
            {"2": 200, "3": 300, "4": 100},
        ],
    )
    assert layerwise == {"2": 200, "3": 200, "4": 200}
    assert consensus_iterations("catboost_pooled", [100, 300, 200]) == 200
