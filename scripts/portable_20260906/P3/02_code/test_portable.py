"""Synthetic-only portable contracts; no organizer inputs or historical fits."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import audit  # noqa: E402
import run as r  # noqa: E402
from p3_clean import backbone  # noqa: E402
from p3_clean.loss_router import RouterConfig  # noqa: E402


def synthetic_cases(n=18):
    grid = pd.DataFrame({"step_minute": np.arange(-2880, 1, 10)})
    for i, key in enumerate([*r.BASE_COLUMNS, *r.DIRECTION_COLUMNS]):
        grid[key] = 2 + i * 0.1 + np.sin(np.arange(289) / 15) * 0.2
    features = r.summarize_context(grid)
    cases = pd.DataFrame([features] * n)
    cases["case_id"] = [f"S{i}" for i in range(n)]
    cases["station"] = ["A", "B", "C"] * (n // 3)
    cases["hs_current"] += np.arange(n) / 100
    return cases, r.compact_feature_columns(list(features))


def test_fixed_feature_count_and_local_imports():
    _, columns = synthetic_cases()
    assert len(columns) == 591
    assert not any(name.startswith("p3_wave") for name in sys.modules)
    for name, module in list(sys.modules.items()):
        if name.startswith("p3_clean") and getattr(module, "__file__", None):
            assert CODE in Path(module.__file__).resolve().parents


def test_native_models_and_complete_policy_replay(tmp_path):
    cases, columns = synthetic_cases()
    matrix, current, _ = r.rows_for_cases(cases, columns)
    matrix = backbone._cat_frame(matrix)
    recipe = json.loads((CODE / "recipe.json").read_text())
    recipe["model"]["single"]["iterations"] = 3
    recipe["model"]["multi"].update(iterations=3, task_type="CPU", thread_count=2)
    recipe["model"]["multi"].pop("devices")
    single = backbone._single_model(recipe, 11)
    single.fit(matrix, np.sin(np.arange(len(matrix))) * 0.1, cat_features=[0, 1])
    multi = backbone._multi_model(recipe, 11)
    mx = cases[["station", *columns]].copy()
    multi.fit(mx, np.sin(np.arange(len(cases) * 6).reshape(-1, 6)) * 0.1, cat_features=[0])
    x = r.build_inference_router_features(
        cases.loc[:, r.OBSERVED_FEATURES],
        cases.station.to_numpy(),
        cases.hs_current.to_numpy(),
        np.repeat(current.reshape(-1, 6, 1), 3, axis=2),
    )
    x.insert(1, "lead_h", "12")
    router = r.ComponentLossRouter(RouterConfig(10, 2, 0.5, "smooth_medium"))
    router.fit(x, np.square(np.arange(len(cases) * 3).reshape(-1, 3) / 100) + 0.01)
    expected_keys, expected = r.predict_cases(cases, columns, single, multi, router)
    single.save_model(tmp_path / "single.cbm")
    multi.save_model(tmp_path / "multi.cbm")
    r.joblib.dump(router, tmp_path / "router.joblib")
    actual_keys, actual = r.predict_cases(
        cases,
        columns,
        r.CatBoostRegressor().load_model(tmp_path / "single.cbm"),
        r.CatBoostRegressor().load_model(tmp_path / "multi.cbm"),
        r.joblib.load(tmp_path / "router.joblib"),
    )
    assert np.array_equal(expected, actual)
    assert expected_keys.equals(actual_keys)
    assert np.isfinite(actual).all()
    assert len(actual) == len(cases) * 6


def test_shrink_and_router_are_not_equal_only():
    leads = np.array(r.LEADS)
    value = r.apply_long_lead_persistence_shrink(np.repeat(3.0, 6), np.repeat(2.0, 6), leads)
    assert np.array_equal(value[:3], [3, 3, 3])
    assert np.array_equal(value[3:], np.repeat(0.8 * 3.0 + 0.2 * 2.0, 3))
    recipe = json.loads((CODE / "recipe.json").read_text())
    assert recipe["router"]["strength"] == 0.5
    assert recipe["router"]["alpha"] == 10


def test_rmse_is_pooled_sse():
    sse, rmse = audit.arithmetic([0, 0, 0], [1, 2, 3])
    assert sse == 14
    assert rmse == pytest.approx(np.sqrt(14 / 3))
    with pytest.raises(ValueError):
        audit.arithmetic([0], [])


@pytest.mark.parametrize(
    "name,writing",
    [("test_index.csv", False), ("sample_submission.csv", False), ("train_wave.csv", True)],
)
def test_stage_source_guard_in_child(tmp_path, name, writing):
    source = tmp_path / "source"
    source.mkdir()
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import run as r; p=Path(sys.argv[2]); r.guard(p, False); "
        + (
            "import os; os.open(p/sys.argv[3], os.O_WRONLY|os.O_CREAT)"
            if writing
            else "(p/sys.argv[3]).open('rb')"
        )
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(CODE), str(source), name],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode != 0
    assert "PermissionError" in result.stderr


def test_answer_bytes_and_duplicate_case_rejection():
    cases, columns = synthetic_cases()
    cases.iloc[1, cases.columns.get_loc("case_id")] = cases.iloc[0].case_id
    cases.iloc[1, cases.columns.get_loc("station")] = cases.iloc[0].station
    with pytest.raises(ValueError, match="duplicate"):
        r.rows_for_cases(cases, columns)
    frame = pd.DataFrame({"case_id": ["A"], "station": ["S"], "lead_h": [3], "hs_pred": [1.2]})
    assert r.csv_bytes(frame).startswith(b"case_id,station,lead_h,hs_pred")


def test_no_discovery_or_broad_data_loader():
    spec = importlib.util.find_spec("p3_clean.data")
    text = Path(spec.origin).read_text()
    assert "def load_p3_data" not in text
    assert "def resolve_p3_data_dir" not in text
    assert "pd.read_csv" not in text
