"""No real source/official data: parity, mask, inner split and scratch boundary."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "scripts/p2_crossfit_copula_materialization_20260906_v3/02_code"


def run_probe(code, *args):
    return subprocess.run([sys.executable, "-c", code, str(ROOT), str(CODE), *map(str, args)],
                          text=True, capture_output=True, check=True, timeout=45)


def test_same_cpu_recipe_features_fit_and_copula():
    source = """
import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'scripts'))
import run_p2_crossfit_copula_forward_20260906_v1 as old
sys.path.insert(0,sys.argv[2]);import run as package
records=[]
for stamp in pd.date_range('2024-05-01',periods=20,freq='10min',tz='Asia/Seoul'):
    for layer in range(1,7):
        records.append(dict(station='S-ORS',year=2024,time=stamp,layer=layer,depth=layer*3.,nominal_depth=layer*3.,temp=20-layer*.6,psal=31+layer*.1))
obs=pd.DataFrame(records);frame,truth=old.adapter(obs)
_,_,cfg=old.settings();cfg.update(device='cpu',epochs=1,batch_size=16,cpu_threads=2)
a,ar=old.base.training_arrays(frame,truth,'v23_blockmask',cfg)
b,br=package.core.training_arrays(frame,truth,'v23_blockmask',cfg)
assert ar==br and all(np.array_equal(x,y) for x,y in zip(a,b))
old.torch.set_num_threads(2)
with old.threadpool_limits(2):
    first,_=old.fit_cpu(a,20260901,cfg,lambda *_:None,float('inf'))
    second,_=package.core.fit_model(b,'v23_blockmask',20260901,cfg,lambda *_:None)
    assert all(old.torch.equal(first.state_dict()[k],v) for k,v in second.state_dict().items())
    c=old.cmean([first],frame);assert np.array_equal(c,package.prediction([second],frame))
    x=old.profile.physical_features(frame,c);assert np.array_equal(x,package.copula.physical_features(frame,c),equal_nan=True)
    model=old.profile.fit_copula(x,truth-c)
    assert np.array_equal(old.profile.predict_copula(model,x),package.copula.predict_copula(model,x))
    assert all(package.copula.covariance_checks(model,x,truth-c).values())
print(json.dumps({'status':'PASS'}))
"""
    assert json.loads(run_probe(source).stdout.splitlines()[-1])["status"] == "PASS"


def test_calendar_inner_split_source_only():
    source = """
import sys,json,pandas as pd,numpy as np
sys.path.insert(0,sys.argv[2]);import run as p
frame=pd.DataFrame({'time':pd.date_range('2024-05-01','2025-12-31',freq='D',tz='Asia/Seoul')})
plan=p.inner_plan(frame);assert len(plan)==2
seen=np.zeros(len(frame),bool)
for item in plan:
    t,v=p.split(frame,item);assert not np.any(t&v) and not np.any(v&seen)
    seen|=v
    assert item['train_rows']==int(t.sum()) and item['validation_rows']==int(v.sum())
print(json.dumps({'status':'PASS'}))
"""
    assert json.loads(run_probe(source).stdout)["status"] == "PASS"


def test_native_scratch_hash_three_seeds_and_denials(tmp_path):
    source = """
import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[2]);import run as p
root=Path(sys.argv[3]);p.ROOT=root/'package';p.ROOT.mkdir();p.MODEL=p.ROOT/'03_model';p.MODEL.mkdir();p.ANSWER=p.ROOT/'05_answer';p.ANSWER.mkdir()
source=root/'source';source.mkdir();source=source/'observations.csv';source.write_text('synthetic')
old=root/'old.pt';p.torch.save({'x':p.torch.ones(1)},old)
p.guard('RUN_TRAINING',source)
for seed in (1,2,3):
    path=p.MODEL/f'{seed}.pt';p.torch.save({'v':p.torch.tensor([seed])},path);assert len(p.sha(path))==64
for path,mode in [(old,'rb'),(source,'w'),(source.parent/'test_index.csv','r')]:
    try:open(path,mode)
    except PermissionError:pass
    else:raise AssertionError('unexpected access')
try:p.torch.load(path)
except PermissionError:pass
else:raise AssertionError('training model load')
print(json.dumps({'status':'PASS'}))
"""
    assert json.loads(run_probe(source, tmp_path).stdout)["status"] == "PASS"


def test_target_joint_mask_in_package_adapter(tmp_path):
    source = """
import sys,json,pandas as pd,numpy as np
from pathlib import Path
sys.path.insert(0,sys.argv[2]);import run as p
path=Path(sys.argv[3])/'observations.csv';rows=[]
for t in pd.date_range('2024-05-01',periods=4,freq='10min',tz='Asia/Seoul'):
    for layer in range(1,7):
        rows.append(dict(station='S-ORS',year=2024,time=t,layer=layer,depth=layer*3.,nominal_depth=layer*3.,temp=20-layer,psal=31+layer*.1))
obs=pd.DataFrame(rows);obs.to_csv(path,index=False)
first,y=p.public_population(path,{'source_sha256':p.sha(path)},False)
obs.loc[obs.layer.isin([2,3,4]),['temp','psal']]=[999.,-999.];obs.to_csv(path,index=False)
second,z=p.public_population(path,{'source_sha256':p.sha(path)},False)
pd.testing.assert_frame_equal(first,second);assert not np.array_equal(y,z)
print(json.dumps({'status':'PASS'}))
"""
    assert json.loads(run_probe(source, tmp_path).stdout)["status"] == "PASS"


def test_empty_copy_does_not_copy_weights_or_answers(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("empty_copy", CODE / "create_empty_copy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, target = tmp_path / "completed", tmp_path / "new"
    for directory in ("02_code", "03_model", "05_answer"):
        (source / directory).mkdir(parents=True)
    for name in ("config.json", "README.md", "requirements.txt", "02_code/run.py", "03_model/model.pt", "05_answer/answer.csv"):
        (source / name).write_text("synthetic")
    module.create(source, target)
    assert not list((target / "03_model").iterdir()) and not list((target / "05_answer").iterdir())
    assert (source / "03_model/model.pt").read_text() == "synthetic"


def test_archive_excludes_data_locks_and_calibration(tmp_path):
    import importlib.util
    from zipfile import ZipFile
    spec = importlib.util.spec_from_file_location("package_archive", CODE.parent / "archive_package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "completed"
    for directory in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (source / directory).mkdir(parents=True)
    for name in ("README.md", "requirements.txt", "config.json", "02_code/run.py", "03_model/a.pt",
                 "03_model/c.npz", "05_answer/a.csv", "06_docs/BUILD_MANIFEST.json", "06_docs/VERIFIED_RUN.md",
                 "01_data/observations.csv", "ATTEMPT_LOCK.json", "04_logs/calibration.npz"):
        (source / name).write_text("synthetic")
    for name in ("training-qa.json", "inference-qa.json", "replay-qa.json"):
        (source / "04_logs" / name).write_text(json.dumps({"status": "PASS"}))
    archive, extracted = tmp_path / "verified.zip", tmp_path / "extracted"
    module.archive(source, archive, extracted)
    with ZipFile(archive) as bundle:
        names = bundle.namelist()
    assert "03_model/a.pt" in names and "03_model/c.npz" in names and "05_answer/a.csv" in names
    assert "ATTEMPT_LOCK.json" not in names and "01_data/observations.csv" not in names
    assert "04_logs/calibration.npz" not in names
    assert "06_docs/VERIFIED_RUN.md" in names
    assert (extracted / "03_model/a.pt").read_text() == "synthetic"
