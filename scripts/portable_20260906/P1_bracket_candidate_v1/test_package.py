"""No-fit, synthetic parity/boundary tests for the standalone bracket candidate."""
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
spec = importlib.util.spec_from_file_location("bracket_package_builder", HERE / "build_package.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    path = tmp_path_factory.mktemp("bracket_candidate_parent") / "package"
    builder.build(path)
    return path


def execute(package, code, env=None):
    result = subprocess.run([sys.executable, "-c", code], cwd=package / "02_code",
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


TOY = '''
import numpy as np,pandas as pd
def toy(n=1200,begin="2025-01-01"):
 t=np.arange(n)
 return pd.DataFrame({"station":"S-ORS","year":pd.Timestamp(begin).year,"layer":1,
 "time":pd.date_range(begin,periods=n,freq="10min",tz="Asia/Seoul").astype(str),
 "temp":10+np.sin(t/20),"psal":32+.1*np.sin(t/13),"depth":5.,
 "label":(t%41<5).astype(np.int8),"anomaly_type":""})
'''


def test_source_manifest_and_empty_destination(package):
    manifest = json.loads((package / "02_code/source-manifest.json").read_text())
    for name, item in manifest.items():
        assert hashlib.sha256((package / "02_code" / name).read_bytes()).hexdigest() == item["sha256"]
    assert not list((package / "03_model").iterdir())
    assert not list((package / "05_answer").iterdir())
    with pytest.raises(FileExistsError):
        builder.build(package)


def test_exact_research_feature_selection_and_weight_parity(package):
    tail = TOY + '''
import hashlib,json
f=toy()
stats=screen.stats_fit(f)
h=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
out={}
for name in ("original","balanced"):
 b=make(f,stats,name)
 encoder=screen.TabularEncoder().fit(b,np.arange(len(f)))
 out[name]={"columns":list(b.frame),"matrix_sha":h(encoder.transform(b)),"maps":encoder.category_maps}
rules=screen.rule_masks(f,stats)
out["rules"]=[h(r) for r in rules]
out["weights"]=h(screen._event_day_weight(f,f.label.to_numpy()))
p={"original":np.linspace(0,1,len(f)),"balanced":np.linspace(1,0,len(f))}
out["policy"]=runner.select_policy(f,p,rules,frozen)
print(json.dumps(out))
'''
    prefix = '''from pathlib import Path
import json,sys
sys.path.insert(0,"scripts")
import run_p1_bracket_forward_20260906_v1 as research
import verify_p1_clean_regeneration_20260905_v5 as runner
screen=runner.screen
cfg=json.loads(Path("configs/experiments/p1_bracket_forward_20260906_v1.json").read_text())
frozen=json.loads(Path("configs/experiments/p1_score_repair_20260905_v1.json").read_text())
make=lambda f,s,name:research.bundle(f,s,frozen,cfg,bracket=(name=="balanced"))
'''
    original = subprocess.check_output([sys.executable, "-c", prefix + tail], cwd=ROOT, text=True, timeout=90)
    portable = execute(package, '''import run as runner
screen=runner.screen
cfg,frozen,_=runner.source_contract()
make=lambda f,s,name:runner.arm_bundle(f,s,frozen,cfg,name)
''' + tail)
    assert json.loads(portable.splitlines()[-1]) == json.loads(original.splitlines()[-1])


def test_bracket_radius_gap_and_target_independence(package):
    execute(package, "import run as r\ncfg,fr,_=r.source_contract()\n" + TOY + '''
f=toy(); raw=f[r.screen.RAW]; center=600; radius=37*6
a=r.bracket.bracket_features(raw,cfg)
changed=raw.copy(); idx=np.arange(len(raw))
changed.loc[(idx<center-radius)|(idx>center+radius),["temp","psal","depth"]]=9999
b=r.bracket.bracket_features(changed,cfg)
np.testing.assert_array_equal(a.iloc[center],b.iloc[center])
assert a.shape==(1200,27)
assert a.loc[center,"bracket_72h_coverage"]==1
later=toy(300,"2025-02-01")[r.screen.RAW]
joined=pd.concat([raw,later],ignore_index=True)
j=r.bracket.bracket_features(joined,cfg)
pd.testing.assert_frame_equal(a,j.iloc[:1200])
try:r.bracket.bracket_features(f,cfg)
except ValueError:pass
else:raise AssertionError("target accepted")
''')


def test_final_inner_boundaries_partition_sentinels_and_encoder(package):
    execute(package, "import run as r\ncfg,fr,_=r.source_contract()\n" + TOY + '''
frame=pd.concat([toy(400,"2025-01-01"),toy(400,"2025-06-25"),toy(400,"2025-07-12"),toy(400,"2025-11-01")],ignore_index=True)
train,evaluation=r.inner_split(frame,fr,cfg)
changed=frame.copy()
changed.loc[(changed.time.str[:10]=="2025-06-25")|(changed.time.str[:10]=="2025-11-01"),["temp","psal","depth"]]=99999
# Perturb every excluded partition, including its keys/labels, without entering the permitted dates.
outside=~pd.to_datetime(frame.time,utc=True).isin(pd.to_datetime(pd.concat([train,evaluation]).time,utc=True))
changed.loc[outside,["temp","psal","depth"]]=99999
changed.loc[outside,"station"]="SENTINEL"
changed.loc[outside,"label"]=1-changed.loc[outside,"label"]
tr2,ev2=r.inner_split(changed,fr,cfg)
pd.testing.assert_frame_equal(train,tr2);pd.testing.assert_frame_equal(evaluation,ev2)
stats=r.screen.stats_fit(train)
for name in ("original","balanced"):
 b=r.arm_bundle(train,stats,fr,cfg,name);e=r.arm_bundle(evaluation,stats,fr,cfg,name)
 enc=r.screen.TabularEncoder().fit(b,np.arange(len(train)))
 np.testing.assert_array_equal(enc.transform(e),enc.transform(r.arm_bundle(ev2,stats,fr,cfg,name)))
 # Allowed evaluation observations/labels cannot influence training stats/features/encoder.
 changed_eval=evaluation.copy();changed_eval[["temp","psal","depth"]]=12345;changed_eval.label=1-changed_eval.label
 before=enc.transform(b)
 r.arm_bundle(changed_eval,stats,fr,cfg,name)
 np.testing.assert_array_equal(before,enc.transform(r.arm_bundle(train,stats,fr,cfg,name)))
rules=r.screen.rule_masks(evaluation,stats);rules2=r.screen.rule_masks(ev2,stats)
for a,b in zip(rules,rules2):np.testing.assert_array_equal(a,b)
p=np.linspace(0,1,len(evaluation))
np.testing.assert_array_equal(r.screen.decode(evaluation,p,rules,fr,.2),r.screen.decode(ev2,p,rules2,fr,.2))
assert pd.to_datetime(train.time,utc=True).max()<pd.Timestamp(cfg["training_stop"])
assert pd.to_datetime(evaluation.time,utc=True).min()==pd.Timestamp(cfg["inner_start"])
''')


def test_whole_terminal_positive_run_exclusion_no_future_labels(package):
    execute(package, "import run as r\n" + TOY + '''
frame=toy(12,"2025-06-20 22:30")
frame["label"]=[0]*6+[1]*6
stop=pd.Timestamp("2025-06-21",tz="Asia/Seoul")
a=r.screen.train_slice(frame,stop)
assert len(a)==6 and not a.label.any()
changed=frame.copy();changed.loc[pd.to_datetime(frame.time,utc=True)>=stop,"label"]=0
pd.testing.assert_frame_equal(a,r.screen.train_slice(changed,stop))
''')


def test_missing_depth_and_unknown_categories_are_not_imputed(package):
    execute(package, "import run as r\ncfg,fr,_=r.source_contract()\n" + TOY + '''
train=toy();stats=r.screen.stats_fit(train)
ev=train.copy();ev["year"]=2026;ev["depth"]=np.nan
for name in ("original","balanced"):
 b=r.arm_bundle(train,stats,fr,cfg,name);e=r.arm_bundle(ev,stats,fr,cfg,name)
 assert e.frame.nominal_depth_m.isna().all()
 assert e.frame.depth_regime.str.contains("unknown").all()
 enc=r.screen.TabularEncoder().fit(b,np.arange(len(train)));x=enc.transform(e)
 assert np.isnan(x[:,list(e.frame).index("nominal_depth_m")]).all()
 assert (x[:,list(e.frame).index("depth_regime")]==-1).all()
''')


def test_policy_decoder_and_independent_f1(package):
    execute(package, "import run as r\ncfg,fr,_=r.source_contract()\n" + TOY + '''
f=toy(240);s=r.screen.stats_fit(f);rules=r.screen.rule_masks(f,s)
p={"original":np.linspace(0,1,len(f)),"balanced":np.linspace(1,0,len(f))}
selected,cals,metrics=r.select_policy(f,p,rules,fr)
options=r.policy_bits(f,p,rules,fr,cals)
for name in r.POLICIES:
 bits,_,_=r.canonical.decoder.control_components(f,p,rules,fr,name,cals,1e-6)
 np.testing.assert_array_equal(bits,options[name])
 assert abs(r.independent_metric(f.label,bits)["f1"]-metrics[name]["f1"])<1e-14
assert selected==max(r.POLICIES,key=lambda n:metrics[n]["f1"])
''')


def test_no_official_read_before_independent_training_gate(package):
    execute(package, '''import run as r
r.checked_training=lambda:({"pid":1}, {})
def forbidden(*a,**k):raise AssertionError("official read happened before QA gate")
r.pd.read_csv=forbidden
try:r.infer()
except FileNotFoundError:pass
else:raise AssertionError("missing QA receipt accepted")
''')


@pytest.mark.parametrize("phase,operation", [
    ("train", "official"), ("model-qa", "official"),
    ("infer", "train"), ("train", "network"), ("model-qa", "repo")])
def test_phase_and_network_boundary(package, tmp_path, phase, operation):
    env = os.environ.copy()
    env.update(P1_DATA_DIR=str(tmp_path), P1_DENY_REPO=str(ROOT))
    paths = {"official": tmp_path / "test.csv", "train": tmp_path / "train.csv", "repo": ROOT / "AGENTS.md"}
    call = ("import socket;socket.socket().connect(('127.0.0.1',1))" if operation == "network"
            else "open(" + repr(str(paths[operation])) + ").read()")
    code = "import run as r;r.install_boundary_guard(" + repr(phase) + ")\ntry:\n " + call
    code += "\nexcept PermissionError:pass\nelse:raise AssertionError('boundary bypass')"
    execute(package, code, env)


def test_schema_empty_models_and_probability_guards(package, tmp_path):
    execute(package, "import run as r\n" + TOY + '''
from pathlib import Path
f=toy(12);keys=f[r.screen.KEYS].iloc[::-1].reset_index(drop=True)
bits=np.arange(12)%2;answer=r.align_answer(f,bits,keys)
assert answer.label.tolist()==bits[::-1].tolist()
for bad in (np.full(12,np.nan),np.full(12,2.),np.zeros(11)):
 try:r.checked_probability(bad,12)
 except ValueError:pass
 else:raise AssertionError("invalid probability accepted")
try:r.align_answer(pd.concat([f,f]),np.r_[bits,bits],keys)
except ValueError:pass
else:raise AssertionError("duplicate keys accepted")
''' + "\np=Path(" + repr(str(tmp_path / "03_model")) + ")\nr.ensure_empty_models(p)\n(p/'keep').touch()\ntry:r.ensure_empty_models(p)\nexcept FileExistsError:pass\nelse:raise AssertionError('overwrite accepted')\nassert (p/'keep').exists()")


def test_fixed_resource_and_model_parameter_contract(package):
    execute(package, '''import run as r,json
cfg,fr,_=r.source_contract()
base=r.screen.load_config(r.ROOT/fr["base_config"],env={})
lgb=json.loads((r.ROOT/fr["lightgbm_recipe"]).read_text())
assert cfg["max_fits"]==r.MAX_FITS==4
assert cfg["threads"]==r.THREADS==4 and not cfg["gpu"]
assert cfg["features"]=={"original":80,"balanced":107}
assert cfg["trees"]==base.raw["models"]["xgboost"]["n_estimators"]==lgb["lightgbm_parameters"]["n_estimators"]==700
assert cfg["fit_order"]==["inner_balanced","inner_original","full_original","full_balanced"]
assert cfg["workflow_wall_cap_seconds"]==r.WALL_SECONDS==3600
''')


@pytest.mark.parametrize("isolated", [False, True])
def test_entrypoint_independent_of_working_directory(package, tmp_path, isolated):
    command = [sys.executable] + (["-I"] if isolated else [])
    command += [str(package / "02_code/run.py"), "--help"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "model-qa" in result.stdout and "infer" in result.stdout
