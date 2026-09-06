"""Bounded synthetic checks for new full3 adapter; never official input."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts/p2_c3_training_comparison_full_20260906_v1"
SPEC = importlib.util.spec_from_file_location("p2_full_builder_test", HERE / "build.py")
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


def test_recipe_and_frozen_lineage():
    cfg = json.loads(b.CONFIG.read_text())
    assert cfg["epochs"] == 120 and cfg["weight_decay"] == .0001
    assert cfg["seeds"] == [20260901, 20260902, 20260903]
    assert cfg["maximum_new_fits"] == 3 and cfg["wallcap_seconds"] == 1800
    assert cfg["device"] == "cuda" and cfg["cpu_threads"] == 2
    assert b.sha(ROOT / "scripts/portable_20260906/P2/02_code/core.py") == cfg["core_sha256"]


def test_adaptations_are_serialization_and_names_only():
    original = (ROOT / "scripts/portable_20260906/P2/02_code/run.py").read_text(encoding="utf-8")
    adapted = b.adapted_base(original)
    assert 'lineterminator="\\n"' in adapted
    restored = adapted.replace("submission_p2_L120_3seed.csv", "submission_p2_clean_C3.csv")
    restored = restored.replace("replay_p2_L120_3seed.csv", "replay_p2_clean_C3.csv")
    restored = restored.replace(', lineterminator="\\n"', "")
    assert restored == original
    compile(adapted, "base.py", "exec")
    with pytest.raises(ValueError):
        b.adapted_base("wrong source")


def test_stage_gate_and_no_new_projection():
    code = (HERE / "run.py").read_text()
    assert '"RUN_TRAINING", "TRAIN_REPLAY", "RUN_INFERENCE", "REPLAY", "FINAL_QA"' in code
    assert 'qa["model_manifest_sha256"]' in code
    assert "full_training_normalized_prediction_sha256" in code
    assert 'data[0][:166268]' in code
    assert "bin17" not in code and "PAVA" not in code


def test_copied_base_real_guard_native_writer_and_csv_schema(tmp_path):
    package = tmp_path / "package"
    code_dir = package / "02_code"
    code_dir.mkdir(parents=True)
    for d in ("03_model", "04_logs", "05_answer"):
        (package / d).mkdir()
    core = ROOT / "scripts/portable_20260906/P2/02_code/core.py"
    (code_dir / "core.py").write_bytes(core.read_bytes())
    source_base = ROOT / "scripts/portable_20260906/P2/02_code/run.py"
    (code_dir / "base.py").write_text(b.adapted_base(source_base.read_text(encoding="utf-8")), encoding="utf-8")
    old = tmp_path / "prior.pt"
    old.write_bytes(b"synthetic-old")
    program = """
import pathlib,sys,numpy as np,pandas as pd,torch
sys.path.insert(0,CODE)
import base
sample=pd.DataFrame({"station":["S-ORS"]*3,"layer":[2,3,4],"time":["2025-09-01T00:00:00+09:00"]*3})
out=sample.copy();out["temp"]=[1.,2.,3.]
assert all(base.validate_output(out,sample,sample,3).values())
bad=out.iloc[::-1].copy()
try:base.validate_output(bad,sample,sample,3);raise AssertionError("order accepted")
except ValueError:pass
source=pathlib.Path(SOURCE);old=pathlib.Path(OLD)
base.install_guard("RUN_TRAINING",source)
data=(np.ones((4,5,8),np.float32),np.ones((4,5),np.float32),np.ones((4,11),np.float32),np.zeros(4,np.float32),np.ones(4,np.float32))
cfg={"cpu_threads":2,"device":"cpu","learning_rate":.001,"weight_decay":.0001,"epochs":1,"batch_size":4,"gradient_coefficient":.01}
for seed in (1,2):
 model,_=base.core.fit_model(data,"v23_blockmask",seed,cfg,lambda *a:None)
 p=base.MODELS/(str(seed)+".pt");torch.save(model.state_dict(),p)
 assert len(base.sha(p))==64
 try:torch.load(p,weights_only=True);raise AssertionError("training load allowed")
 except PermissionError:pass
 try:old.read_bytes();raise AssertionError("old model read allowed")
 except PermissionError:pass
 try:pd.read_csv(source.parent/"test_index.csv",usecols=base.KEYS);raise AssertionError("official during training")
 except PermissionError:pass
print("PASS synthetic-only; historical/full0")
"""
    prefix = "\n".join(f"{k}={json.dumps(str(v))}" for k, v in {"CODE": code_dir, "OLD": old, "SOURCE": tmp_path / "observations.csv"}.items())
    result = subprocess.run([sys.executable, "-I", "-B", "-c", prefix + "\n" + program],
                            text=True, capture_output=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_candidate_named_separately_and_full_replay_scope():
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    assert "submission_p2_L120_3seed.csv" in readme and "not the current official best" in readme
    assert "166268" in readme and "26061" in readme
