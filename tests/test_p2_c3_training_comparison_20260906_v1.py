"""Synthetic-only preregistered C3 comparison contract checks."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/p2_c3_training_comparison_20260906_v1/run.py"
SPEC = importlib.util.spec_from_file_location("p2_study_test", PATH)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
SPEC_QA = importlib.util.spec_from_file_location("p2_study_qa_test", PATH.with_name("qa.py"))
qa = importlib.util.module_from_spec(SPEC_QA)
SPEC_QA.loader.exec_module(qa)


def observations():
    return pd.DataFrame([{"station": "S-ORS", "year": 2024, "time": stamp, "layer": layer,
        "depth": float(layer * 5), "nominal_depth": float(layer * 5),
        "temp": float(20 - layer + i * .1), "psal": float(30 + layer * .1)}
        for i, stamp in enumerate(pd.date_range("2024-10-01", periods=24, freq="D", tz="Asia/Seoul"))
        for layer in range(1, 9)])


def test_exact_recipes_budget_and_v5():
    config = r.cfg()
    assert config["maximum_fits"] == 3 * 8 + 2 * 2 == 28
    assert len(r.contract()["folds"]) == 8
    assert r.contract()["primary"]["fold"] == "B3"
    assert config["duplicate_exclusion"]["absolute_C_MSE"]
    assert [x["epochs"] for x in config["recipes"]] == [60, 120, 60]
    assert [x["weight_decay"] for x in config["recipes"]] == [.0001, .0001, .001]


def test_target_temp_salinity_never_features():
    raw = observations()
    a, labels_a = r.prepare_frame(raw)
    changed = raw.copy()
    changed.loc[changed.layer.isin((2, 3, 4)), ["temp", "psal"]] += 1234
    b, labels_b = r.prepare_frame(changed)
    assert not np.array_equal(labels_a, labels_b)
    for x, y in zip(r.core.arrays(a), r.core.arrays(b), strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(r.key_values(a), r.key_values(b))


def test_public_joint_mask_recomputes_features_only_in_interval():
    frame, _ = r.prepare_frame(observations())
    altered, take, supported = r.make_outage(frame, {"end": "2024-11-01T00:00:00+09:00"})
    assert take.any() and (~take).any() and supported.all()
    assert altered.loc[take, ["temp_5", "psal_5"]].isna().all().all()
    for a, b in zip(r.core.arrays(frame), r.core.arrays(altered), strict=True):
        np.testing.assert_array_equal(a[~take], b[~take])
    assert not np.array_equal(r.core.arrays(frame)[0][take], r.core.arrays(altered)[0][take])
    assert np.all(altered.public_temp_count[take] == frame.public_temp_count[take] - 1)


def test_empty_outage_is_not_artificially_moved():
    frame, _ = r.prepare_frame(observations())
    altered, take, supported = r.make_outage(frame, {"end": "2025-01-01T00:00:00+09:00"})
    assert not take.any() and supported.all()
    for a, b in zip(r.core.arrays(frame), r.core.arrays(altered), strict=True):
        np.testing.assert_array_equal(a, b)
    assert r.metric(np.array([]), np.array([]))["rmse_C"] is None


def test_two_sided_purge_boundaries():
    times = pd.to_datetime(["2024-08-24T23:59Z", "2024-08-25T00:00Z", "2024-09-01T00:00Z",
                            "2024-11-07T23:59Z", "2024-11-08T00:00Z"], utc=True)
    training, valid = r.split_masks(pd.DataFrame({"time": times}), {"start": "2024-09-01T00:00Z", "end": "2024-11-01T00:00Z"})
    np.testing.assert_array_equal(training, [True, False, False, False, True])
    np.testing.assert_array_equal(valid, [False, False, True, False, False])


def test_nominal_depth_fallback_keeps_rows():
    raw = observations()
    raw.loc[raw.layer.isin((2, 3, 4)), "depth"] = np.nan
    frame, _ = r.prepare_frame(raw)
    assert len(frame) == 72 and frame.target_actual_depth.isna().all()
    assert all(np.isfinite(a).all() for a in r.core.arrays(frame))


def test_augmentation_preserves_original_total_mass_and_recipe_inputs():
    frame, truth = r.prepare_frame(observations())
    expected = None
    for recipe in r.cfg()["recipes"]:
        arrays, receipt = r.core.training_arrays(frame, truth, "v23_blockmask", r.recipe_config(r.cfg(), recipe))
        assert np.isclose(arrays[-1].sum(), len(frame))
        assert receipt["original_rows"] == len(frame) and receipt["augmented_rows"] > 0
        hashes = [r.array_sha(a) for a in arrays]
        assert expected is None or hashes == expected
        expected = hashes


def test_selection_only_declared_challengers_tie_order():
    scores = {"natural": {arm: {"B3_primary": {"rmse_C": 1}, "all8_pooled": {"rmse_C": 1}} for arm in r.RECIPE_IDS}}
    assert r.choose_challenger(scores) == "L120"
    scores["natural"]["D60"]["B3_primary"]["rmse_C"] = .9
    assert r.choose_challenger(scores) == "D60"


def test_denominator_not_fold_mean():
    expected = np.sqrt((1 + 81) / 2)
    assert qa.metric(np.array([0., 0]), np.array([1., 9]))["rmse_C"] == expected
    assert qa.metric(np.array([]), np.array([]))["rmse_C"] is None
    with pytest.raises(ValueError):
        qa.metric(np.array([0.]), np.array([np.nan]))


def test_paired_bootstrap_and_insufficient_support():
    a = qa.bootstrap(np.zeros(4), np.ones(4) * 2, np.ones(4), np.array([1, 1, 2, 2]))
    assert a["ci90_C"] == [-1., -1.] and a["empirical_improvement_fraction"] == 1
    b = qa.bootstrap(np.zeros(4), np.ones(4), np.ones(4), np.ones(4))
    assert b["status"] == "NOT_ESTIMABLE"


def test_keys_reject_duplicate_or_missing():
    frame, _ = r.prepare_frame(observations())
    with pytest.raises(ValueError):
        r.key_values(pd.concat([frame, frame.iloc[:1]]))
    frame.loc[0, "time"] = None
    with pytest.raises(ValueError):
        r.key_values(frame)


def test_native_training_save_sha_next_seed_and_forbidden_old_read(tmp_path):
    # Real subprocess under the installed audit hook: previous guards failed here.
    old = tmp_path / "old.pt"
    old.write_bytes(b"synthetic-old")
    out = tmp_path / "new_artifact"
    out.mkdir()
    code = """
import importlib.util, pathlib, numpy as np
spec=importlib.util.spec_from_file_location("s",PATH)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
out=pathlib.Path(OUT);old=pathlib.Path(OLD)
s.install_guard(pathlib.Path(SOURCE),out,True)
data=(np.ones((8,5,8),np.float32),np.ones((8,5),np.float32),np.ones((8,11),np.float32),np.zeros(8,np.float32),np.ones(8,np.float32))
cfg={"cpu_threads":2,"device":"cpu","learning_rate":.001,"weight_decay":.0001,"epochs":1,"batch_size":8,"gradient_coefficient":.01}
for seed in (1,2):
 model,receipt=s.core.fit_model(data,"v23_blockmask",seed,cfg,lambda *a:None)
 target=out/(str(seed)+".pt")
 with target.open("xb") as stream:s.torch.save(model.state_dict(),stream)
 assert len(s.sha(target))==64
 for filename in (target,old):
  try:s.torch.load(filename,weights_only=True);raise AssertionError("load allowed")
  except PermissionError:pass
 try:old.read_bytes();raise AssertionError("old read allowed")
 except PermissionError:pass
print("PASS synthetic 2 fits; historical 0")
"""
    prefix = "\n".join(f"{k}={json.dumps(str(v))}" for k, v in {
        "PATH": PATH, "OUT": out, "OLD": old, "SOURCE": tmp_path / "observations.csv"}.items())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-I", "-B", "-c", prefix + "\n" + code], capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS synthetic" in result.stdout


def test_pilot_preperformance_cap():
    assert r.pilot_estimate(24.078, r.cfg()) < 5400
    assert r.pilot_estimate(100, r.cfg()) > 5400


def test_official_and_foreign_artifacts_denied(tmp_path):
    source = tmp_path / "source/observations.csv"
    output = tmp_path / "new"
    assert r.path_allowed(source, False, source, output)
    assert not r.path_allowed(source, True, source, output)
    assert not r.path_allowed(tmp_path / "test_index.csv", False, source, output)
    assert not r.path_allowed(tmp_path / "old/model.pt", False, source, output)
    assert r.path_allowed(output / "03_model/new.pt", False, source, output)


def test_canonical_predict_has_no_projector():
    code = PATH.read_text()
    section = code.split("def predict(model, frame):", 1)[1].split("\ndef ", 1)[0]
    assert "baseline.to_numpy" in section and "compute_profile_scale" in section
    assert "pava" not in section.lower() and "envelope" not in section.lower()
