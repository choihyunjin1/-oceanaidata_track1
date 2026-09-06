import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "hmax_ablation",
    Path(__file__).resolve().parents[1] / "scripts/run_p3_hmax_removed_forward_20260906_v1.py",
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def columns():
    return json.loads(
        (
            mod.ROOT
            / "artifacts/p3_clean_regeneration_20260905_v4/04_validation/feature_columns.json"
        ).read_text()
    )["columns"]


def test_exact64features_removed():
    original = columns()
    kept = mod.selected_columns(original)
    assert len(kept) == 527 and len(original) - len(kept) == 64
    assert all(not name.startswith("hmax_") for name in kept)
    assert "hs_current" in kept and "tp_current" in kept


@pytest.mark.parametrize("hmax", [0.0, 777.0, np.nan])
def test_raw_hmax_perturbation_has_no_retained_effect(hmax):
    rng = np.random.default_rng(31)
    context = pd.DataFrame(
        {name: 2 + rng.random(289) for name in (*mod.BASE_COLUMNS, *mod.DIRECTION_COLUMNS)}
    )
    context.loc[::2, ["hs", "tp", "hmax", "wvdir"]] = np.nan
    before = mod.summarize_context(context)
    context["hmax"] = hmax
    after = mod.summarize_context(context)
    kept = mod.selected_columns(columns())
    np.testing.assert_array_equal([before[name] for name in kept], [after[name] for name in kept])


def test_router_really_drops_hmax_instead_of_only_base_models():
    feature = pd.DataFrame(
        {"anchor_id": [0, 1], **{name: [2.0, 3.0] for name in mod.OBSERVED_FEATURES}}
    )
    frame = pd.DataFrame(
        {
            "fold": "Q2_2024",
            "anchor_id": np.repeat([0, 1], 6),
            "station": "A",
            "anchor_time": pd.to_datetime(np.repeat(["2024-04-05T00:00Z", "2024-04-10T00:00Z"], 6)),
            "episode_id": np.repeat([1, 2], 6),
            "lead_h": np.tile(mod.LEADS, 2),
            "current_hs": 2.0,
            "target_hs": 2.5,
            "single_prediction": 2.4,
            "multi_prediction": 2.3,
            "persistence": 2.0,
        }
    )
    x1, meta1, comp1, loss1 = mod.router_material(frame, feature)
    feature.hmax_current = [999.0, -123.0]
    x2, meta2, comp2, loss2 = mod.router_material(frame, feature)
    assert "hmax_current" not in x1.columns and x1.equals(x2) and meta1.equals(meta2)
    np.testing.assert_array_equal(comp1, comp2)
    np.testing.assert_array_equal(loss1, loss2)


def test_same_recipe_devices_and_categorical_lead():
    recipe = json.loads(
        (
            mod.ROOT / "configs/experiments/p3_corrected_repeated_forward_catboost_v2.json"
        ).read_text()
    )
    a, b = mod.parameters(recipe, "single", 31), mod.parameters(recipe, "multi", 31)
    assert a["task_type"] == "CPU" and b["task_type"] == "GPU" and b["devices"] == "0"
    assert a["thread_count"] == b["thread_count"] == 2
    assert a["iterations"] == 700 and b["iterations"] == 1200
    cfg = json.loads(mod.CONFIG.read_text())
    assert cfg["removal"]["numeric_lead"] is False and cfg["budget"]["backbone_fits"] == 10
    assert cfg["budget"]["router_fits"] == 4 and cfg["budget"]["wall_seconds"] == 3600


def test_preexisting_path_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "OUT", tmp_path)
    with pytest.raises(FileExistsError):
        mod.preflight({}, tmp_path)
