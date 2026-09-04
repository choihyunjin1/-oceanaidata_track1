from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from p2_restore.gaussian_copula_conditional_mean import (
    GaussianCopulaConditionalMean,
    SeasonalCopulaConditionalMean,
    empirical_quantile,
    empirical_to_normal,
    kendall_latent_correlation,
    normal_scores,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p2_gaussian_copula_conditional_mean_20260830_v1.py"
SPEC = importlib.util.spec_from_file_location("p2_gaussian_copula_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_normal_scores_are_finite_monotone_and_tie_preserving() -> None:
    values = np.asarray([3.0, 1.0, 2.0, 2.0, 4.0])
    scores = normal_scores(values)
    order = np.argsort(values, kind="stable")
    assert np.isfinite(scores).all()
    assert np.all(np.diff(scores[order]) >= 0.0)
    assert scores[2] == scores[3]


def test_empirical_transform_and_inverse_are_monotone() -> None:
    support = np.linspace(-2.0, 3.0, 101)
    query = np.linspace(-4.0, 5.0, 301)
    scores = empirical_to_normal(support, query)
    restored = empirical_quantile(support, np.linspace(0.0, 1.0, 301))
    assert np.isfinite(scores).all()
    assert np.all(np.diff(scores) >= 0.0)
    assert np.all(np.diff(restored) >= 0.0)


def test_kendall_latent_correlation_is_symmetric_with_unit_diagonal() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 4))
    x[:, 1] = 0.7 * x[:, 0] + 0.3 * x[:, 1]
    correlation = kendall_latent_correlation(x)
    assert np.allclose(correlation, correlation.T)
    assert np.allclose(np.diag(correlation), 1.0)
    assert correlation[0, 1] > 0.5


def test_copula_conditional_mean_learns_nonlinear_monotone_dependence() -> None:
    rng = np.random.default_rng(11)
    latent = rng.normal(size=800)
    x = np.column_stack([np.exp(0.3 * latent), rng.normal(size=800)])
    y = np.column_stack(
        [
            np.sinh(0.5 * latent) + rng.normal(scale=0.08, size=800),
            np.tanh(latent) + rng.normal(scale=0.08, size=800),
            0.2 * latent + rng.normal(scale=0.08, size=800),
        ]
    )
    model = GaussianCopulaConditionalMean.fit(
        x[:600],
        y[:600],
        shrinkage=0.3,
        quadrature_order=15,
        eigen_floor=1e-8,
        maximum_condition_number=1e6,
    )
    prediction = model.predict(x[600:])
    zero_rmse = np.sqrt(np.mean(np.square(y[600:])))
    model_rmse = np.sqrt(np.mean(np.square(prediction - y[600:])))
    assert model_rmse < zero_rmse
    assert model.minimum_eigenvalue >= -1e-10
    assert model.condition_number <= 1e6


def test_seasonal_model_uses_global_fallback_for_unseen_season() -> None:
    rng = np.random.default_rng(19)
    x = rng.normal(size=(500, 2))
    y = np.column_stack([x[:, 0], x[:, 0] * 0.5, -x[:, 0]]) + rng.normal(
        scale=0.1, size=(500, 3)
    )
    seasons = np.repeat("JJA", len(x))
    model = SeasonalCopulaConditionalMean.fit(
        x,
        y,
        seasons,
        shrinkage=0.3,
        quadrature_order=9,
        eigen_floor=1e-8,
        maximum_condition_number=1e6,
        minimum_season_profiles=200,
    )
    prediction = model.predict(x[:10], np.repeat("SON", 10))
    assert prediction.shape == (10, 3)
    assert np.isfinite(prediction).all()


def test_config_is_one_shot_small_grid_and_forbids_official_access() -> None:
    config = json.loads(RUNNER.CONFIG.read_text(encoding="utf-8"))
    assert config["experiment_id"] == RUNNER.EXPERIMENT_ID
    assert config["copula"]["shrinkage_candidates"] == [0.1, 0.3, 0.5]
    assert config["resource_contract"]["maximum_conceptual_copula_fits"] == 30
    assert config["execution_policy"]["maximum_executions"] == 1
    assert not any(
        [
            config["execution_policy"]["official_hidden_gap_values_read_allowed"],
            config["execution_policy"]["official_test_sample_submission_read_allowed"],
            config["execution_policy"]["submission_csv_generation_allowed"],
            config["execution_policy"]["official_upload_authorized"],
            config["execution_policy"]["result_based_retry"],
            config["execution_policy"]["wide_hpo_allowed"],
        ]
    )


def test_sealed_outer_folds_and_fit_budget_match_preregistration() -> None:
    config = json.loads(RUNNER.CONFIG.read_text(encoding="utf-8"))
    assert list(config["folds"]) == ["2024_sep_oct", "2025_jul_aug", "2025_nov_dec"]
    expected = len(config["folds"]) * (
        len(config["copula"]["shrinkage_candidates"])
        * config["copula"]["inner_time_groups"]
        + 1
    )
    assert expected == config["resource_contract"]["maximum_conceptual_copula_fits"]
