import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p3_forward_candidate_materialize_20260906_v1 as mod


class StubSingle:
    def predict(self, x, **kwargs):
        self.x = x.copy()
        return np.zeros(len(x))


class StubMulti:
    def predict(self, x, **kwargs):
        self.x = x.copy()
        return np.zeros((len(x), 6))


class StubRouter:
    def predict_weights(self, x):
        self.x = x.copy()
        return np.tile([0.5, 0.5, 0.0], (len(x), 1))


@pytest.mark.parametrize("experiment", mod.EXPERIMENTS)
def test_full_predict_preserves_single_change_and_six_keys(experiment):
    e = importlib.import_module("run_" + experiment)
    cfg = mod.read(e.CONFIG)
    context = pd.DataFrame(
        {name: 2 + np.arange(289) / 1000 for name in (*e.BASE_COLUMNS, *e.DIRECTION_COLUMNS)}
    )
    features = e.summarize_context(context)
    cases = pd.DataFrame([dict(case_id="SYNTHETIC", station="A", **features)])
    columns = mod.policy_columns(e, cfg, e.compact_feature_columns(list(features)))
    a, b, router = StubSingle(), StubMulti(), StubRouter()
    keys, prediction = mod.predict_cases(e, cfg, cases, columns, (a, b, router))
    assert keys.lead_h.to_list() == list(e.LEADS) and len(prediction) == 6
    assert list(keys.columns) == mod.KEYS
    expected = np.repeat(cases.hs_current.to_numpy(), 6)
    expected[3:] = (1.0 - 0.2) * expected[3:] + 0.2 * expected[3:]
    np.testing.assert_array_equal(prediction, expected)
    if "removal" in cfg:
        assert not any(
            name.startswith("hmax_") for model in (a, b, router) for name in model.x.columns
        )
        assert pd.api.types.is_string_dtype(a.x.lead_h)
    else:
        assert a.x.lead_h.dtype == np.float64 and "hmax_current" in router.x.columns


def test_fullfit_rejects_existing_output_without_retraining(tmp_path):
    with pytest.raises(FileExistsError):
        mod.fit(None, {}, tmp_path)


def test_no_mean_improvement_stops_before_models(monkeypatch, tmp_path):
    e = SimpleNamespace(REPORT=tmp_path)
    result = {"comparison": {"candidate_retained": False, "delta_candidate_minus_control": 0.01}}
    monkeypatch.setattr(mod, "read", lambda path: result if path.name == "result.json" else {})
    monkeypatch.setattr(mod, "sha", lambda path: "synthetic")
    with pytest.raises(ValueError, match="no mean-improved"):
        mod.verify_inputs(e, {})


def test_synthetic_answer_keys_public_hashes_lf_and_new_pid_replay(monkeypatch, tmp_path):
    out, source, report = (tmp_path / name for name in ("out", "source", "prior_report"))
    for path in (out / "04_logs", out / "05_answer", source, report):
        path.mkdir(parents=True)
    mod.save(out / "04_logs/training-result.json", {"synthetic": True})
    mod.save(report / "result.json", {"synthetic": True})
    mod.save(
        out / "04_logs/training-independent-qa.json",
        {"status": "PASS", "training_result_sha256": mod.sha(out / "04_logs/training-result.json")},
    )
    (source / "test_context.parquet").write_bytes(b"synthetic bytes, not an official file")
    (source / "test_index.csv").write_bytes(b"synthetic bytes, not an official file")
    case_ids = [f"SYNTHETIC_{number}" for number in range(200)]
    context = pd.DataFrame(
        {
            "case_id": np.repeat(case_ids, 289),
            "station": "SYNTHETIC",
            "step_minute": np.tile(np.arange(-2880, 1, 10), 200),
        }
    )
    index = pd.DataFrame(
        {
            "case_id": np.repeat(case_ids, 6),
            "station": "SYNTHETIC",
            "lead_h": np.tile([3, 6, 9, 12, 18, 24], 200),
        }
    )
    e = SimpleNamespace(
        REPORT=report,
        BASE_COLUMNS=(),
        DIRECTION_COLUMNS=(),
        summarize_context=lambda group: {"hs_current": 1.5},
    )
    monkeypatch.setattr(mod, "load", lambda *args: ({"pid": 100}, None, []))
    monkeypatch.setattr(mod.pd, "read_parquet", lambda *args, **kwargs: context.copy())
    monkeypatch.setattr(mod.pd, "read_csv", lambda *args, **kwargs: index.copy())
    monkeypatch.setattr(mod, "predict_cases", lambda *args: (index.copy(), np.ones(len(index))))
    monkeypatch.setattr(mod.os, "getpid", lambda: 200)
    mod.inference(e, {}, out, source, verify_answer=False)
    payload = (out / "05_answer/submission.csv").read_bytes()
    assert b"\r\n" not in payload and payload.count(b"\n") == 1201
    receipt = mod.read(out / "04_logs/answer-qa.json")
    assert receipt["rows"] == 1200 and receipt["key_order_exact"]
    assert receipt["public_input_sha256"] == {
        name: mod.sha(source / name) for name in ("test_context.parquet", "test_index.csv")
    }
    monkeypatch.setattr(mod.os, "getpid", lambda: 300)
    mod.inference(e, {}, out, source, verify_answer=True)
    replay = mod.read(out / "04_logs/answer-replay-qa.json")
    assert replay["pid"] != receipt["pid"] and replay["sha256"] == receipt["sha256"]
