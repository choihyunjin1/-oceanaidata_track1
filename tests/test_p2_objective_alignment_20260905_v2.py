"""Synthetic-only P2 factorial loss, weighting and leakage contracts."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("objective_alignment", ROOT / "scripts/run_p2_objective_alignment_20260905_v2.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def sample_frame():
    records = []
    for time in pd.date_range("2024-09-01", periods=12, freq="10min", tz="UTC"):
        for layer in range(1, 9):
            records.append({"station": "S-ORS", "time": time, "layer": layer, "nominal_depth": layer * 5.0, "depth": layer * 5.0 + 0.7, "temp": 20.0 - layer * 0.3, "psal": 30.0 + layer * 0.1})
    return runner.base.public_frame(pd.DataFrame(records))


def test_source_target_poison_invariance():
    frame, _ = sample_frame()
    poisoned = frame.copy()
    poisoned["target"] = -999
    for first, second in zip(runner.base.arrays(frame), runner.base.arrays(poisoned), strict=True):
        np.testing.assert_array_equal(first, second)


def test_C_data_exact_and_both_weight_measures_preserve_original_mass():
    frame, truth = sample_frame()
    cfg = runner.load_config()
    cfg["blockmask"]["coverage"] = 1.0
    fresh, _ = runner.make_training_arrays(frame, truth, cfg)
    old, _ = runner.base.training_arrays(frame, truth, "v23_blockmask", cfg)
    for first, second in zip(fresh[:5], old, strict=True):
        np.testing.assert_array_equal(first, second)
    n = len(frame)
    np.testing.assert_allclose(fresh[6][:n] + fresh[6][n:], 1)
    unmasked, _ = runner.base.training_arrays(frame, truth, "v23", cfg)
    np.testing.assert_allclose(fresh[4][:n] + fresh[4][n:], unmasked[4])


def test_absolute_C_MSE_formula_and_penalty_invariance():
    tokens = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2).requires_grad_(True)
    estimate = tokens[:, :, 0].sum(dim=1) * 0.2
    target = torch.tensor([0.1, 0.4])
    scale = torch.tensor([0.5, 4.0])
    domain, equal = torch.tensor([3.0, 1.0]), torch.ones(2)
    mask = torch.ones(2, 2)
    terms = {arm: runner.objective_terms(estimate, target, scale, domain, equal, tokens, mask, arm) for arm in ("C", "M", "R", "MR")}
    for arm in terms:
        torch.testing.assert_close(terms[arm][1], terms["C"][1], rtol=0, atol=0)
    absolute_square = (scale * (estimate - target)).square()
    torch.testing.assert_close(terms["M"][0], (domain * absolute_square).sum() / domain.sum())
    torch.testing.assert_close(terms["MR"][0], absolute_square.mean())
    (terms["MR"][0] + 0.01 * terms["MR"][1]).backward()
    assert torch.isfinite(tokens.grad).all()


def test_stress_label_independent_and_recomputed_before_arrays():
    frame, _ = sample_frame()
    cfg = runner.load_config()
    cfg["stress"]["start"] = "2024-09-01T00:00:00+00:00"
    cfg["stress"]["stop"] = "2024-09-02T00:00:00+00:00"
    altered, support, selected = runner.stress_frame(frame, cfg)
    assert selected.all() and support.all()
    assert altered.temp_5.isna().all() and altered.psal_5.isna().all()
    assert not runner.base.arrays(altered)[1][:, 1].any()
    np.testing.assert_array_equal(altered.baseline, runner.base.nominal_baseline(altered))


def test_primary_is_intact_autumn_and_budget_frozen():
    cfg = runner.load_config()
    assert cfg["primary_fold"] == "2024_sep_oct"
    assert cfg["maximum_new_historical_fits"] == 3 * 3 + 2 * 3
    assert cfg["stress"]["selection_use"] is False
    assert cfg["official_access_rows"] == cfg["csv_written"] == cfg["upload"] == 0


def test_pooled_metric_not_average_folds():
    truth, pred = np.zeros(4), np.asarray([0, 0, 0, 4])
    result = runner.metrics_by_scope(truth, pred, np.asarray(["a", "a", "a", "b"]), np.asarray([True, True, True, False]))
    assert result["pooled"]["rmse"] == pytest.approx(2)
    assert result["primary"]["rmse"] == 0


def test_single_source_reader_and_no_official_writer():
    text = (ROOT / "scripts/run_p2_objective_alignment_20260905_v2.py").read_text(encoding="utf-8")
    assert text.count("pd.read_csv(") == 1
    for forbidden in ("test_index.csv", "sample_submission.csv", "baseline_interp.csv", "to_csv(", "read_parquet("):
        assert forbidden not in text
