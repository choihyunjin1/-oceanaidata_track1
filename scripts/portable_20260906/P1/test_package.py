"""Synthetic parity and boundary checks; no real data or trained weights."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("p1_package_builder", HERE / "build_package.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    path = tmp_path_factory.mktemp("portable_parent") / "package"
    builder.build(path)
    return path


def probe(prefix, cwd):
    code = prefix + '''
import hashlib,json,numpy as np,pandas as pd
t=pd.date_range("2025-01-01",periods=240,freq="10min",tz="Asia/Seoul")
parts=[]
for layer in (1,2):
 parts.append(pd.DataFrame({"station":"S-ORS","year":2025,"layer":layer,"time":t.astype(str),"temp":10+layer+np.sin(np.arange(240)/10),"psal":30+np.cos(np.arange(240)/17),"depth":layer*5.,"label":(np.arange(240)%41<5).astype(int),"anomaly_type":""}))
f=pd.concat(parts,ignore_index=True)
cfg=json.loads(config.read_text(encoding="utf-8"))
stats=screen.stats_fit(f)
b=screen.feature_pair(f,stats,cfg)[0]
encoder=screen.TabularEncoder().fit(b,np.arange(len(f)))
x=encoder.transform(b)
h=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
rules=screen.rule_masks(f,stats)
p={"original":np.linspace(0,1,len(f)),"balanced":np.linspace(1,0,len(f))}
selected,cals,metrics=runner.select_policy(f,p,rules,cfg)
print(json.dumps({"columns":list(b.frame),"features":h(x),"weights":h(screen._event_day_weight(f,f.label.to_numpy())),"rules":[h(r) for r in rules],"selection":selected,"cals":cals,"metrics":metrics}))
'''
    return json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=cwd, text=True).splitlines()[-1])


def test_synthetic_numeric_parity(package):
    original = probe("from pathlib import Path\nimport sys\nsys.path.insert(0,'scripts')\nimport verify_p1_clean_regeneration_20260905_v5 as runner\nscreen=runner.screen\nconfig=Path('configs/experiments/p1_score_repair_20260905_v1.json')\n", ROOT)
    portable = probe("from pathlib import Path\nimport run as runner\nscreen=runner.screen\nconfig=Path('configs/feature.json')\n", package / "02_code")
    assert portable == original


def test_manifest_and_empty_guard(package):
    manifest = json.loads((package / "02_code/source-manifest.json").read_text())
    for name, item in manifest.items():
        assert hashlib.sha256((package / "02_code" / name).read_bytes()).hexdigest() == item["sha256"]
    with pytest.raises(FileExistsError):
        builder.build(package)


def test_original_repo_and_phase_guard(package, tmp_path):
    env = os.environ.copy()
    env.update(P1_DATA_DIR=str(tmp_path), P1_DENY_REPO=str(ROOT))
    code = "import run; run.install_boundary_guard('train'); open('" + str(ROOT / 'AGENTS.md').replace('\\', '/') + "').read()"
    proc = subprocess.run([sys.executable, "-c", code], cwd=package / "02_code", env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "original repository access denied" in proc.stderr


@pytest.mark.parametrize("action", ["official", "network"])
def test_training_disallows_official_and_network(package, tmp_path, action):
    env = os.environ.copy()
    env["P1_DATA_DIR"] = str(tmp_path)
    operation = ("open('" + str(tmp_path / "test.csv").replace("\\", "/") + "').read()"
                 if action == "official" else "import socket; socket.socket().connect(('127.0.0.1', 1))")
    proc = subprocess.run([sys.executable, "-c", "import run; run.install_boundary_guard('train'); " + operation],
                          cwd=package / "02_code", env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "PermissionError" in proc.stderr
