"""Synthetic mask/split/resource/native-model-path tests; no competition data."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_crossfit_copula_forward_20260906_v1 as m  # noqa: E402


def observations():
    records = []
    for stamp in pd.date_range("2024-10-14T23:50:00+09:00", periods=5, freq="10min"):
        for layer in range(1, 7):
            records.append(
                {
                    "station": "S-ORS",
                    "year": 2024,
                    "time": stamp,
                    "layer": layer,
                    "depth": float(layer * 3),
                    "nominal_depth": float(layer * 3),
                    "temp": float(21 - layer),
                    "psal": float(31 + layer / 10),
                }
            )
    return pd.DataFrame(records)


def test_target_joint_mask_invariance():
    obs = observations()
    first, truth = m.adapter(obs)
    changed = obs.copy()
    changed.loc[changed.layer.isin([2, 3, 4]), ["temp", "psal"]] = [1000.0, -1000.0]
    second, newtruth = m.adapter(changed)
    pd.testing.assert_frame_equal(first, second)
    assert not np.array_equal(truth, newtruth)
    for left, right in zip(m.base.arrays(first), m.base.arrays(second), strict=True):
        assert np.array_equal(left, right)


def test_outage_before_derived_feature_recomputation():
    obs = observations()
    frame, _ = m.adapter(obs)
    spec = {"end": "2024-11-01T00:00:00+09:00"}
    altered, rows = m.outage_frame(frame, spec)
    assert rows.any() and (~rows).any()
    assert altered.loc[rows, ["temp_5", "psal_5"]].isna().all().all()
    source_mask = obs.layer.eq(5) & obs.time.ge(pd.Timestamp("2024-10-15T00:00:00+09:00"))
    obs.loc[source_mask, ["temp", "psal"]] = np.nan
    rebuilt, _ = m.adapter(obs)
    pd.testing.assert_frame_equal(altered, rebuilt)
    before, after = m.base.arrays(frame), m.base.arrays(altered)
    assert all(np.array_equal(a[~rows], b[~rows]) for a, b in zip(before, after, strict=True))
    assert not np.array_equal(before[0][rows], after[0][rows])
    assert not np.array_equal(
        m.profile.physical_features(frame, frame.baseline.to_numpy()),
        m.profile.physical_features(altered, altered.baseline.to_numpy()),
        equal_nan=True,
    )


def test_actual_missing_nominal_support_keeps_rows():
    obs = observations()
    obs.loc[obs.layer.isin([1, 5, 6]), "depth"] = np.nan
    frame, _ = m.adapter(obs)
    tokens, mask, context = m.base.arrays(frame)
    assert len(tokens) == 15 and np.all(mask.sum(axis=1) == 3)
    assert np.isfinite(tokens).all() and np.isfinite(context).all()
    physics = m.profile.physical_features(frame, frame.baseline.to_numpy())
    assert np.isfinite(physics[:, :5]).all()
    assert np.allclose(physics[:, 3], 0)


def test_inner_double_purge_excludes_outer_labels():
    frame = pd.DataFrame(
        {"time": pd.date_range("2024-05-01", "2025-12-31", freq="D", tz="Asia/Seoul").astype(str)}
    )
    _, contract, _ = m.settings()
    for outer in contract["P2"]["folds"]:
        ot, ov = m.masks(frame.time, outer)
        choices = m.choose_inner(frame, outer)
        assert len(choices) == 2
        all_valid = np.zeros(len(frame), bool)
        for inner in choices:
            it, iv = m.masks(frame.time, inner)
            assert np.all(ot[iv]) and not np.any(ov & (ot & it))
            assert not np.any(iv & all_valid)
            all_valid |= iv
            times = pd.to_datetime(frame.time, utc=True)
            left, end = m.base.utc(inner["start"]), m.base.utc(inner["end"])
            assert (
                (times[ot & it] < left - pd.Timedelta(days=7))
                | (times[ot & it] >= end + pd.Timedelta(days=7))
            ).all()


def test_empty_outage_not_zero_metric():
    assert m.metric(np.array([]), np.array([])) == {
        "status": "NOT_ESTIMABLE_NO_ROWS",
        "n": 0,
        "sse": None,
        "rmse": None,
    }


def test_budget_and_recipe_are_explicit():
    cfg, contract, recipe = m.settings()
    assert recipe["batch_size"] == 4096 and recipe["gradient_coefficient"] == 0.01
    assert cfg["max_seconds"] == 5400 and cfg["epochs"] == 60
    assert len(contract["P2"]["folds"]) == 8 and contract["P2"]["primary"]["fold"] == "B3"
    assert cfg["maximum_backbone_fits"] + cfg["maximum_copula_fits"] == 88
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_native_writer_hash_reload_guard_subprocess(tmp_path):
    code = """
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_p2_crossfit_copula_forward_20260906_v1 as m
root=Path(sys.argv[2]); m.OUT=root/'fresh'; m.OUT.mkdir()
source=root/'source'; source.mkdir(); (source/'observations.csv').write_text('synthetic')
old=root/'old.pt'; m.torch.save({'v':m.torch.zeros(1)},old)
os.environ['P2_DATA_DIR']=str(source)
m.install_guard()
for seed in [1,2,3]:
    path=m.OUT/f'{seed}.pt'
    m.torch.save({'v':m.torch.tensor([seed])},path)
    assert len(m.base.file_hash(path))==64
    assert m.torch.load(path,weights_only=True)['v'].item()==seed
for path,mode in [(old,'rb'),(source/'observations.csv','w'),(root/'sample_submission.csv','rb')]:
    try:
        open(path,mode)
    except PermissionError:
        pass
    else:
        raise AssertionError('guard did not reject')
print(json.dumps({'status':'PASS','native_saves':3,'negative_reads':3}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "scripts"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout)["status"] == "PASS"


def test_cpu_training_own_synthetic_model_reload(tmp_path):
    frame, truth = m.adapter(observations())
    _, _, recipe = m.settings()
    recipe = {**recipe, "epochs": 1, "batch_size": 16}
    arrays, receipt = m.base.training_arrays(frame, truth, "v23_blockmask", recipe)
    assert np.isclose(receipt["training_weight_sum"], len(frame), rtol=1e-5)
    model, fitted = m.fit_cpu(arrays, 20260901, recipe, lambda *_: None, float("inf"))
    assert fitted["device"] == "cpu" and fitted["epochs"] == 1
    path = tmp_path / "synthetic.pt"
    m.torch.save(model.state_dict(), path)
    loaded = m.base.make_model("v23", 11)
    loaded.load_state_dict(m.torch.load(path, weights_only=True))
    assert np.array_equal(m.cmean([model], frame), m.cmean([loaded.eval()], frame))
