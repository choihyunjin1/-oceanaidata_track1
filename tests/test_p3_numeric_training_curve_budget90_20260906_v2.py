import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_p3_numeric_training_curve_budget90_20260906_v2 as m


@pytest.fixture
def cfg():
    return copy.deepcopy(m.core.read(m.CONFIG))


@pytest.fixture
def anchors():
    times = pd.to_datetime(
        [
            "2024-01-10",
            "2024-02-01",
            "2024-02-20",
            "2024-02-26",
            "2024-02-27",
            "2024-03-01",
            "2024-03-15",
            "2024-03-24",
            "2024-04-02",
            "2024-07-02",
            "2024-10-02",
            "2025-01-02",
            "2025-04-02",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "anchor_id": np.arange(len(times)),
            "station": "SYNTH",
            "anchor_time": times,
            "episode_id": [f"ep_{i}" for i in range(len(times))],
            "current_hs": 1.5 + np.arange(len(times)) / 100,
        }
    )
    for lead in m.e.LEADS:
        frame[f"target_{lead}"] = frame.current_hs + lead / 100 + np.arange(len(times)) / 1000
    return frame


def test_inner_inside_earliest_training_and_78h_footprint(cfg, anchors):
    contract = m.e.cv.load_contract()
    split = m.inner_masks(anchors, contract, cfg["held_inner"])
    assert np.flatnonzero(split["train"]).tolist() == [0, 1, 2, 3]
    assert np.flatnonzero(split["validation"]).tolist() == [5, 6, 7]
    assert anchors.loc[split["train"], "anchor_time"].max() + pd.Timedelta(hours=24) < anchors.loc[
        split["validation"], "anchor_time"
    ].min() - pd.Timedelta(hours=48)


def test_inner_excludes_same_episode_and_parent_heldout(cfg, anchors):
    anchors.loc[0, "episode_id"] = anchors.loc[5, "episode_id"]
    anchors.loc[6, "episode_id"] = anchors.loc[8, "episode_id"]
    split = m.inner_masks(anchors, m.e.cv.load_contract(), cfg["held_inner"])
    assert not split["train"][0]
    assert not split["validation"][6]


def test_latest_parent_allowed_inner_target_still_precedes_outer(cfg, anchors):
    cfg["held_inner"]["end"] = "2024-04-01T00:00:00+00:00"
    anchors.loc[7, "anchor_time"] = pd.Timestamp("2024-03-28T17:00:00Z")
    # Still parent-train eligible, but +24h must precede Apr1 -48h.
    split = m.inner_masks(anchors, m.e.cv.load_contract(), cfg["held_inner"])
    assert split["validation"][7]


def test_empty_inner_rejected(cfg, anchors):
    cfg["held_inner"]["start"] = "2024-03-10T00:00:00+00:00"
    cfg["held_inner"]["end"] = "2024-03-11T00:00:00+00:00"
    with pytest.raises(ValueError, match="support"):
        m.inner_masks(anchors, m.e.cv.load_contract(), cfg["held_inner"])


def test_frozen_schedule_values_and_boosting_time(cfg):
    m.validate_config(cfg)
    previous = m.core.read(m.e.CONFIG)
    original = m.core.read(m.ROOT / previous["recipe"])
    for arm in cfg["recipes"]:
        recipe = m.modified_recipe(previous, arm)
        for kind in ("single", "multi"):
            assert recipe["model"][kind]["iterations"] * recipe["model"][kind][
                "learning_rate"
            ] == pytest.approx(
                original["model"][kind]["iterations"] * original["model"][kind]["learning_rate"]
            )
            changed = {
                key
                for key in recipe["model"][kind]
                if recipe["model"][kind][key] != original["model"][kind][key]
            }
            assert changed <= {"iterations", "learning_rate"}


@pytest.mark.parametrize(
    "field,value", [("single_iterations", 701), ("multi_learning_rate", 0.123), ("id", "hmax")]
)
def test_unknown_recipe_rejected(cfg, field, value):
    cfg["recipes"][1][field] = value
    with pytest.raises(ValueError):
        m.validate_config(cfg)


def test_inner_argmin_and_baseline_tie():
    rows = [{"recipe": r, "sse_m2": 10.0} for r in ("baseline", "compact", "gentle")]
    assert m.select_inner(rows) == 0
    rows[2]["sse_m2"] = 9.0
    assert m.select_inner(rows) == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_selection_rejected(bad):
    rows = [{"recipe": r, "sse_m2": bad} for r in ("baseline", "compact", "gentle")]
    with pytest.raises(ValueError):
        m.select_inner(rows)


def test_short_equal_long_fixed_shrink():
    frame = pd.DataFrame(
        {
            "single_prediction": np.arange(6, dtype=float),
            "multi_prediction": np.arange(6, dtype=float) + 2,
            "persistence": 2.0,
            "lead_h": m.e.LEADS,
        }
    )
    expected = np.arange(6, dtype=float) + 1
    expected[3:] = 0.8 * expected[3:] + 0.2 * 2
    np.testing.assert_array_equal(m.inner_prediction(frame), expected)


def test_helper_out_restored_on_error(tmp_path, monkeypatch):
    original = m.e.OUT
    monkeypatch.setattr(m, "OUT", tmp_path)
    with pytest.raises(RuntimeError):
        with m.owned_output():
            assert m.e.OUT == tmp_path
            raise RuntimeError("synthetic")
    assert m.e.OUT == original


def test_real_cpu_synthetic_fit_save_and_case_major_replay(cfg, anchors, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "OUT", tmp_path)
    features = anchors[["anchor_id", "station"]].copy()
    features["hs_current"] = anchors.current_hs
    features["hmax_current"] = 2 * features.hs_current
    columns = ["hs_current", "hmax_current"]
    split = m.inner_masks(anchors, m.e.cv.load_contract(), cfg["held_inner"])
    previous = m.core.read(m.e.CONFIG)
    recipe = m.modified_recipe(previous, cfg["recipes"][0])
    real_params = m.e.parameters

    def params(*args, **kwargs):
        result = real_params(*args, **kwargs)
        result.pop("devices", None)
        result.update(task_type="CPU", iterations=3, depth=2)
        return result

    monkeypatch.setattr(m.e, "parameters", params)
    for kind in ("numeric_single", "multi"):
        prediction, receipt = m.fit_component(
            cfg["recipes"][0],
            {"id": "INNER"},
            0,
            split,
            features,
            anchors,
            columns,
            recipe,
            previous,
            time.perf_counter() + 20,
            kind,
        )
        ids = anchors.loc[split["validation"], "anchor_id"].to_numpy()
        reproduced = m.predict_saved(receipt, features, anchors, columns, ids)
        np.testing.assert_array_equal(prediction, reproduced)
        assert prediction.shape == (3, 6)
        assert m.native_matches(
            tmp_path / receipt["model_path"], kind, receipt["parameters"], columns
        )
        assert receipt["cat_features"] == ["station"]
        assert receipt["validation_prediction_sha256"] == m.prediction_sha(reproduced)


def test_existing_output_preserved(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "OUT", tmp_path)
    with pytest.raises(FileExistsError):
        m.preflight(cfg, {})


def test_resource_amendment_only_scientific_fields_unchanged(cfg):
    original = m.core.read(
        m.ROOT / "configs/experiments/p3_numeric_training_curve_20260906_v1.json"
    )
    for field in ("recipes", "held_inner", "outer", "fixed", "baseline"):
        assert cfg[field] == original[field]
    assert cfg["budget"]["wall_seconds"] == 5400
    assert cfg["budget"]["max_new_fits"] + cfg["reuse"]["model_count"] == 20
    assert cfg["budget"]["max_new_backbones"] == 14


@pytest.mark.parametrize("bad", ["parameters", "trees", "categories", "feature_order"])
def test_native_mismatch_rejected_without_predict(cfg, monkeypatch, bad):
    expected = {"iterations": 700, "learning_rate": 0.035, "task_type": "CPU"}

    class FakeModel:
        tree_count_ = 699 if bad == "trees" else 700
        feature_names_ = ["station", "lead_h", "current_hs_for_residual", "hmax_current"]
        if bad == "feature_order":
            feature_names_ = list(reversed(feature_names_))

        def load_model(self, path):
            return self

        def get_params(self):
            value = expected.copy()
            if bad == "parameters":
                value["learning_rate"] = 0.123
            return value

        def get_cat_feature_indices(self):
            return [0, 1] if bad == "categories" else [0]

        def predict(self, *args):
            raise AssertionError("metadata acceptance must never predict")

    monkeypatch.setattr(m.core, "CatBoostRegressor", FakeModel)
    with pytest.raises(ValueError):
        m.native_matches("synthetic.cbm", "numeric_single", expected, ["hmax_current"])


def test_copy_only_two_owned_models_preserves_source(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "OUT", tmp_path / "new")
    (tmp_path / "old").mkdir()
    receipts = {}
    for kind in ("numeric_single", "multi"):
        old = tmp_path / "old" / (kind + ".cbm")
        old.write_bytes(b"synthetic model metadata only " + kind.encode())
        receipts[kind] = {
            "source_path": old.relative_to(tmp_path).as_posix(),
            "model_path": "models/baseline_INNER/" + kind + ".cbm",
            "model_sha256": m.core.sha(old),
        }
    accepted = {"status": "PASS", "receipts": receipts}
    m.copy_accepted_models(accepted)
    m.verify_accepted_copies(accepted)
    for record in receipts.values():
        assert m.core.sha(tmp_path / record["source_path"]) == record["model_sha256"]
    with pytest.raises(ValueError, match="not empty"):
        m.copy_accepted_models(accepted)


def test_reuse_copy_escape_and_mutation_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "OUT", tmp_path / "own")
    accepted = {
        "status": "PASS",
        "receipts": {
            kind: {"model_path": "../other.cbm", "model_sha256": "not-used"}
            for kind in ("numeric_single", "multi")
        },
    }
    with pytest.raises(ValueError, match="escapes"):
        m.verify_accepted_copies(accepted)


def test_execution_does_not_refit_reused_baseline():
    import ast
    import inspect

    source = inspect.getsource(m.execute)
    tree = ast.parse(source)
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "arm['id'] == 'baseline'"
        and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "predict_saved"
            for child in node.body
            for n in ast.walk(child)
        )
    )
    names = {
        node.func.id
        for child in branch.body
        for node in ast.walk(child)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "predict_saved" in names and "fit_component" not in names
    assert "fit_component" in ast.unparse(branch.orelse)
