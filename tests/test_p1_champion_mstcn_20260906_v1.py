"""No organizer rows or candidate fit/GPU; includes one tiny CPU synthetic step."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/p1_champion_reconstruction_20260906_v1/mstcn.py"
SPEC = importlib.util.spec_from_file_location("champion_mstcn", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


@pytest.fixture(scope="module")
def source():
    return M.load_source(ROOT)


def synthetic_frame():
    rows = []
    for station in ("G-ORS", "I-ORS", "S-ORS"):
        for layer in range(1, 9):
            for index in range(32):
                rows.append({
                    "station": station, "year": 2025, "layer": layer,
                    "time": pd.Timestamp("2025-01-01", tz="Asia/Seoul") + pd.Timedelta(minutes=10 * index),
                    "temp": 10 + np.sin(index / 4) + layer / 10,
                    "psal": 30 + np.cos(index / 4),
                    "depth": np.nan if index == 0 else layer * 5 + index / 10,
                    "label": int(12 <= index < 16),
                    "anomaly_type": "offset" if 12 <= index < 16 else "",
                })
    return pd.DataFrame(rows)


def test_exact_source_pins():
    assert M.verify_sources(ROOT) == M.PINS


@pytest.mark.parametrize("field,expected", [
    ("seeds", [20260827, 20260839, 20260863]), ("epochs", 150),
    ("schedule_horizon_epochs", 300), ("width", 512), ("batch_size", 64),
    ("full_fit_budget", 3), ("threshold", 0.8), ("end_to_end_cap_seconds", 21600),
])
def test_fixed_contract(field, expected):
    assert M.contract()[field] == expected


def test_no_approval_has_no_output(tmp_path):
    with pytest.raises(PermissionError):
        M.train(ROOT, tmp_path / "train.csv", tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_new_output_no_reuse(tmp_path):
    with pytest.raises(ValueError, match="no restart"):
        M.train(ROOT, tmp_path / "train.csv", tmp_path,
                training_approved=True, gpu_approved=True)


def test_schedule_horizon_not_compressed(source):
    old, config = source
    capacity = old._config_for_capacity(config, width=512, seed=M.SEEDS[0])
    assert old._schedule_geometry(capacity, window_count=1707) == (27, 8100, 270)


def test_fresh_feature_target_isolation_and_order(source):
    old, config = source
    frame = synthetic_frame()
    left, names, audit = M.fresh_features(frame, ROOT, old, config)
    changed = frame.copy()
    changed["label"] = 1 - changed["label"]
    changed["anomaly_type"] = "forbidden synthetic target must never enter features"
    right, other_names, _audit = M.fresh_features(changed, ROOT, old, config)
    np.testing.assert_array_equal(left.numeric, right.numeric)
    assert names == other_names and len(names) == 74
    assert audit["unbounded_features_projected"] == 0
    pd.testing.assert_frame_equal(left.keys, frame.loc[:, M.KEYS])
    assert left.labels is None and left.anchor is None


def test_encoder_roundtrip_165_and_year_not_category(source):
    old, config = source
    frame = synthetic_frame()
    surface, names, _ = M.fresh_features(frame, ROOT, old, config)
    encoder, (encoded,) = old._fit_encoder_and_transform(
        surface, [], fit_ids=np.arange(len(frame)), forbidden_ids=np.array([], dtype=np.int64),
        numeric_names=names)
    payload = json.loads(json.dumps(old._encoder_receipt(encoder)))
    loaded = M.encoder_from_receipt(payload)
    replay = M.encode_with_saved(surface, loaded, old)
    assert replay.features.shape == (len(frame), 165)
    np.testing.assert_array_equal(encoded.features, replay.features)
    before = json.dumps(old._encoder_receipt(loaded), sort_keys=True)
    surface.keys["year"] = 2030
    another = M.encode_with_saved(surface, loaded, old)
    np.testing.assert_array_equal(replay.features, another.features)
    assert json.dumps(old._encoder_receipt(loaded), sort_keys=True) == before


def test_window_gap_padding_and_target_masks(source):
    old, config = source
    frame = synthetic_frame()
    frame = frame.drop(index=17).reset_index(drop=True)
    surface, names, _ = M.fresh_features(frame, ROOT, old, config)
    surface.labels = frame.label.to_numpy(dtype=np.int8)
    surface.anomaly_type = frame.anomaly_type.to_numpy()
    _encoder, (encoded,) = old._fit_encoder_and_transform(
        surface, [], fit_ids=np.arange(len(frame)), forbidden_ids=np.array([], dtype=np.int64),
        numeric_names=names)
    windows = old._selected_windows(encoded, config)
    values, valid = old._load_scientific()[4].materialize_windows(encoded.features, windows)
    target_valid = old._materialize_target_batch(encoded.targets, windows)[3]
    np.testing.assert_array_equal(valid.astype(bool), target_valid)
    assert np.all(values[~valid.astype(bool)] == 0)
    assert len(encoded.layout.segments) == 25


def test_hash_shape_and_order_bound():
    values = np.arange(12, dtype=np.float32)
    assert M.array_sha(values) != M.array_sha(values.reshape(3, 4))
    assert M.array_sha(values) != M.array_sha(values[::-1])


def test_new_model_explicit_and_own_hash_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="03_model"):
        M.predict_frame(pd.DataFrame(), model_dir=tmp_path / "old_weights")
    result = {"contract": M.contract(), "completed_fits": 3,
              "files_sha256": {"03_model/fake.pt": "0" * 64}}
    M.save_json(tmp_path / "training-result.json", result)
    (tmp_path / "03_model").mkdir()
    (tmp_path / "03_model/fake.pt").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="SHA mismatch"):
        M.verify_owned(tmp_path)


def test_existing_receipts_not_overwritten(tmp_path):
    path = tmp_path / "receipt.json"
    M.save_json(path, {"first": True})
    with pytest.raises(FileExistsError):
        M.save_json(path, {"second": True})
    assert M.read_json(path) == {"first": True}


def test_source_calls_do_not_route_old_io():
    import ast

    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    prohibited = {"_load_surfaces", "_load_completed_seed", "_fit_seed", "_preflight", "read_parquet"}
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls.intersection(prohibited)
    assert "_train_epoch" in calls and "predict_encoded" in calls


def test_real_synthetic_subprocess_source_snapshot_io_guard_and_adamw(tmp_path):
    program = r'''
import importlib.util, json, pathlib, shutil, sys
root, own = map(pathlib.Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('mstcn_guard_test', root / 'scripts/p1_champion_reconstruction_20260906_v1/mstcn.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
own.mkdir(); code = own / '02_code'
for relative in m.PINS:
    destination = code / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / relative, destination)
m.configure_owned_runtime(own)
source, cfg = m.load_source(code)
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
m.install_io_guard(own)
frame = pd.DataFrame({'station':['S']*16,'year':[2025]*16,'layer':[1]*16,
  'time':pd.date_range('2025-01-01', periods=16, freq='10min', tz='Asia/Seoul'),
  'temp':np.sin(np.arange(16)),'psal':[30.]*16,'depth':[5.]*16})
surface, names, audit = m.fresh_features(frame, code, source, cfg)
assert len(names) == 74 and audit['unbounded_features_projected'] == 0
# This CPU width-8 synthetic model is only a native I/O/optimizer smoke, not a candidate fit.
cfg['architecture']['width'] = 8
model = source._new_model(5, cfg, torch.device('cpu'))
opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
x=torch.zeros((2,16,5)); valid=torch.ones((2,16),dtype=torch.bool)
y=model(x,valid_mask=valid); loss=y.final_logits.square().mean();loss.backward();opt.step()
state=own/'synthetic.pt'
with state.open('xb') as handle: torch.save(model.state_dict(), handle)
digest=m.sha(state)
loaded=source._new_model(5,cfg,torch.device('cpu'))
loaded.load_state_dict(torch.load(state,map_location='cpu',weights_only=True))
model.eval();loaded.eval()
with torch.no_grad():
    assert torch.equal(model(x,valid_mask=valid).final_logits,loaded(x,valid_mask=valid).final_logits)
try:
    (root/'artifacts/forbidden_old_model.pt').open('rb')
except PermissionError:
    pass
else: raise AssertionError('old artifact boundary did not fail closed')
print(json.dumps({'status':'PASS','synthetic_optimizer_steps':1,'source_copied':True,'sha_chars':len(digest)}))
'''
    result = subprocess.run([sys.executable, "-I", "-c", program, str(ROOT), str(tmp_path / "own")],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])["status"] == "PASS"
