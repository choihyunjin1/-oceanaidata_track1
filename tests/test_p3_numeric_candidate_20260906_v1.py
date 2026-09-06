import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_p3_numeric_candidate_20260906_v1 as m


@pytest.fixture
def config():
    return copy.deepcopy(m.core.read(m.CONFIG))


@pytest.mark.parametrize(
    "field,value",
    [
        ("numeric_lead", False),
        ("hmax_removed", True),
        ("feature_count", 527),
        ("long_persistence", 0.3),
    ],
)
def test_policy_changes_rejected(config, field, value):
    config["policy"][field] = value
    with pytest.raises(ValueError, match="policy"):
        m.verify_contract(config)


def test_extra_fits_rejected(config):
    config["budget"]["new_backbone_fits"] = 12
    with pytest.raises(ValueError, match="budget"):
        m.verify_contract(config)


def test_identity_rejected(config):
    config["historical_experiment"] = "hmax"
    with pytest.raises(ValueError, match="identity"):
        m.verify_contract(config)


def test_pin_tampering_rejected(config):
    config["pins"] = {"synthetic.py": "expected"}
    original = m.core.sha
    m.core.sha = lambda path: "changed"
    try:
        with pytest.raises(ValueError, match="pin differs"):
            m.verify_contract(config)
    finally:
        m.core.sha = original


def test_existing_out_no_fit(config, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "OUT", tmp_path)
    with pytest.raises(FileExistsError, match="restart"):
        m.preflight(config, {})


def test_missing_qa_no_official_io(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "OUT", tmp_path)
    with pytest.raises(FileNotFoundError):
        m.require_numeric_qa()


def test_failed_qa_no_official_io(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "OUT", tmp_path)
    m.core.save(
        tmp_path / "04_logs/numeric-independent-qa.json",
        {
            "status": "FAIL",
            "failed_checks": ["synthetic"],
            "checks": {"synthetic": False},
        },
    )
    with pytest.raises(ValueError, match="QA required"):
        m.require_numeric_qa()


def test_exact_numeric_native_cat_compatibility_and_saved_replay(tmp_path):
    frame = pd.DataFrame(
        {
            "station": np.repeat(["SYNTH_A", "SYNTH_B", "SYNTH_C"], 6),
            "lead_h": np.tile(m.experiment.LEADS, 3),
            "current_hs_for_residual": np.repeat([1.5, 2.0, 3.0], 6),
            "hmax_current": np.repeat([3.0, 4.0, 6.0], 6),
        }
    )
    x = m.experiment.matrix(frame, True)
    assert x.lead_h.dtype == np.float64
    target = np.sin(np.arange(len(x)))
    model = CatBoostRegressor(
        iterations=3,
        depth=2,
        thread_count=2,
        allow_writing_files=False,
        verbose=False,
        random_seed=1,
    )
    model.fit(x, target, cat_features=["station"])
    assert model.get_cat_feature_indices() == [0]
    path = tmp_path / "synthetic.cbm"
    model.save_model(path)
    loaded = CatBoostRegressor().load_model(path)
    np.testing.assert_array_equal(model.predict(x), loaded.predict(x))


def test_case_major_numeric_lead_and_hmax_preserved():
    e = m.experiment
    context = pd.DataFrame(
        {name: 2.0 + np.arange(289) / 1000 for name in (*e.BASE_COLUMNS, *e.DIRECTION_COLUMNS)}
    )
    features = e.summarize_context(context)
    cases = pd.DataFrame(
        [dict(case_id=case, station="SYNTH", **features) for case in ("z", "a", "n")]
    )
    columns = e.compact_feature_columns(list(features))

    class Single:
        def predict(self, x, **kwargs):
            assert x.lead_h.dtype == np.float64
            assert "hmax_current" in x
            return x.lead_h.to_numpy() / 100

    class Multi:
        def predict(self, x, **kwargs):
            assert "hmax_current" in x
            return np.zeros((len(x), 6))

    class Router:
        def predict_weights(self, x):
            assert "hmax_current" in x
            return np.tile([0.5, 0.5, 0], (len(x), 1))

    keys, pred = m.core.predict_cases(e, {}, cases, columns, (Single(), Multi(), Router()))
    assert keys.case_id.tolist() == np.repeat(["z", "a", "n"], 6).tolist()
    assert keys.lead_h.tolist() == list(e.LEADS) * 3
    expected = np.repeat(cases.hs_current.to_numpy(), 6) + np.tile(e.LEADS, 3) / 200
    long_mask = keys.lead_h.isin([12, 18, 24]).to_numpy()
    expected[long_mask] = (
        0.8 * expected[long_mask] + 0.2 * np.repeat(cases.hs_current.to_numpy(), 6)[long_mask]
    )
    # Independent algebra groups float operations differently; serialized model
    # replay above and the production answer replay still require exact equality.
    np.testing.assert_allclose(pred, expected, rtol=0, atol=1e-14)


def test_model_only_predict_default_cpu():
    import inspect

    assert inspect.signature(CatBoostRegressor.predict).parameters["task_type"].default == "CPU"


def test_closure_excludes_environment_and_data():
    closure = m.source_closure()
    assert "scripts/run_p3_numeric_candidate_20260906_v1.py" in closure
    assert all(name.endswith(".py") and name.startswith(("scripts/", "src/")) for name in closure)
    assert not any(".venv" in name for name in closure)


def test_149_item_actual_pass_field_schema():
    qa = {
        "status": "PASS",
        "checks_count": 149,
        "failed_checks": [],
        "checks": [{"check": f"synthetic_{i}", "pass": True} for i in range(149)],
    }
    assert m.historical_checks_pass(qa)
    qa["checks"][100]["pass"] = False
    assert not m.historical_checks_pass(qa)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        [True] * 149,
        [{"ok": True}] * 149,
        [{"pass": "true"}] * 149,
        [{"pass": 1}] * 149,
        [{"pass": True}] * 148,
    ],
)
def test_invalid_or_non_boolean_qa_fail_closed(invalid):
    assert not m.historical_checks_pass(
        {"status": "PASS", "checks_count": 149, "failed_checks": [], "checks": invalid}
    )


def test_code_qa_tied_to_exact_source(config, monkeypatch):
    monkeypatch.setattr(
        m.core,
        "read",
        lambda _: {
            "status": "PASS",
            "ruff": "PASS",
            "focused_pytest_count": 19,
            "driver_sha256": "old",
            "config_sha256": "old",
            "tests_sha256": "old",
        },
    )
    monkeypatch.setattr(m.core, "sha", lambda _: "changed")
    with pytest.raises(ValueError, match="code QA"):
        m.verify_code_qa()
