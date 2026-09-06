"""Pure synthetic/portable checks; no models are fitted and no real data is read."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p3_cold_builder", HERE / "build_package.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    destination = tmp_path_factory.mktemp("p3_cold_source") / "P3"
    b.build(destination)
    return destination


def isolated(package, code):
    bootstrap = "import sys;sys.path.insert(0," + repr(str(package / "02_code")) + ");\n"
    result = subprocess.run([sys.executable, "-I", "-c", bootstrap + code],
                            cwd=package.parent, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr


def test_code_only_exact_source_closure(package):
    c = json.loads((package / "02_code/frozen.json").read_text())
    for path, digest in c["source_snapshot"].items():
        assert b.sha(package / "02_code" / path) == b.sha(b.REPO / path) == digest
    assert not any(any((package / folder).iterdir()) for folder in ("03_model", "04_logs", "05_answer"))
    assert not any(p.suffix in {".csv", ".parquet", ".npz", ".npy", ".cbm", ".joblib"}
                   for p in package.rglob("*"))
    assert c["budget"]["backbone_fits"] == 5 * 2 + 2
    assert c["budget"]["router_fits"] == 5 - 1 + 1
    assert c["budget"]["total_seconds"] == 21600


def test_existing_destination_never_overwritten(package):
    with pytest.raises(FileExistsError):
        b.build(package)


@pytest.mark.parametrize("stage", ["prepare", "train", "infer", "verify-answer"])
def test_unapproved_stage_fails_before_data(package, stage):
    result = subprocess.run([sys.executable, "-I", str(package / "02_code/run.py"), stage],
                            cwd=package.parent, capture_output=True, text=True, timeout=45)
    assert result.returncode == 2 and "authorization required" in result.stderr
    assert not (package / "PREPARE_LOCK.json").exists()


def test_isolated_import_manifest_variants_and_cli(package):
    isolated(package, '''import run as r
c=r.contract()
for variant in ('numeric','hmax'):
 e=r.module(variant)
 assert str(e.__file__).startswith(str(r.CODE))
 anchors=r.pd.DataFrame({'anchor_time':r.pd.to_datetime(['2024-01-02','2024-04-02','2024-07-02','2024-10-02','2025-01-02','2025-04-02'],utc=True),'station':['A']*6,'episode_id':list(range(6))})
 assert len(e.fold_masks(anchors,r.read(r.CODE/'configs/evaluation/ocean_forward_v5.json')))==5
 assert e.parameters(c['recipe'],'single',1)['thread_count']==2
 assert e.parameters(c['recipe'],'multi',1)['task_type']=='GPU'
assert c['recipe']['shrink']['persistence_weight']==.2
''')
    result = subprocess.run([sys.executable, "-I", str(package / "02_code/run.py"), "--help"],
                            cwd=package.parent, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0


@pytest.mark.parametrize("variant", ["numeric", "hmax"])
def test_synthetic_prediction_policy_and_feature_parity(package, variant):
    isolated(package, '''import run as r
import numpy as np
import pandas as pd
variant=''' + repr(variant) + '''
e=r.module(variant); c=r.contract()
context=pd.DataFrame({name:2+np.arange(289)/1000 for name in (*e.BASE_COLUMNS,*e.DIRECTION_COLUMNS)})
values=e.summarize_context(context)
cases=pd.DataFrame([dict(case_id='SYNTHETIC',station='A',**values)])
columns=r.chosen_columns(e,c,variant,e.compact_feature_columns(list(values)))
class Single:
 def predict(self,x,**kw):self.x=x;return np.zeros(len(x))
class Multi:
 def predict(self,x,**kw):self.x=x;return np.zeros((len(x),6))
class Router:
 def predict_weights(self,x):self.x=x;return np.tile([.5,.5,0.],(len(x),1))
single,multi,router=Single(),Multi(),Router()
keys,pred=r.materializer.predict_cases(e,r.policy_config(variant),cases,columns,(single,multi,router))
assert keys.lead_h.tolist()==[3,6,9,12,18,24]
np.testing.assert_allclose(pred,np.repeat(cases.hs_current.to_numpy(),6),rtol=0,atol=1e-14)
assert (single.x.lead_h.dtype==np.float64)==(variant=='numeric')
assert len(columns)==(591 if variant=='numeric' else 527)
if variant=='hmax':
 assert not any(name.startswith('hmax_') for obj in [single,multi,router] for name in obj.x.columns)
''')


def test_synthetic_context_and_prior_fold_exclusion(package):
    isolated(package, '''import run as r
import numpy as np
import pandas as pd
from types import SimpleNamespace
e=r.module('numeric'); t=pd.Timestamp('2024-07-01',tz='UTC')
grid=pd.DataFrame({'station':'A','time':pd.date_range(t-pd.Timedelta(hours=50),t+pd.Timedelta(hours=2),freq='10min')})
for col in (*e.BASE_COLUMNS,*e.DIRECTION_COLUMNS):grid[col]=np.arange(len(grid),dtype=float)/100+2
anchor=SimpleNamespace(station='A',anchor_time=t)
before=e.summarize_context(e.raw_context(grid,anchor))
outside=(grid.time<t-pd.Timedelta(hours=48))|(grid.time>t)
grid.loc[outside,list(e.BASE_COLUMNS)]=-9999
after=e.summarize_context(e.raw_context(grid,anchor))
np.testing.assert_allclose(list(before.values()),list(after.values()),equal_nan=True,rtol=0,atol=0)
c=r.read(r.CODE/'configs/evaluation/ocean_forward_v5.json')
anchors=pd.DataFrame({'anchor_id':[0,1,2,3],'station':['A']*4,'anchor_time':pd.to_datetime(['2024-06-01','2024-06-02','2024-06-29','2024-07-01'],utc=True),'episode_id':[1,2,3,2]})
split=e.cv.p3_split(anchors,'Q3_2024',c)
assert list(split['train'])==[True,False,False,False]
assert list(e.safe_meta_mask(anchors,anchors,'Q3_2024',c))==[True,False,False,False]
''')


def test_no_old_pipeline_or_performance_selection_entrypoints():
    code = (HERE / "run.py").read_text()
    for forbidden in ("e.execute(", "e.preflight(", "e.load_cache(", "verify_inputs(",
                      "materializer.fit(", "candidate_retained", "official_score"):
        assert forbidden not in code
    assert 'build_training_features(data, dense_spacing_minutes=20' in code
    assert 'lambda: os._exit(124)' in code
    assert 'started_unix' in code and 'TRAIN_LOCK.json' in code
    assert 'len(router_x) != 103602' in code
    assert "sum(name.endswith('.cbm') for name in r['models']) != 12" in code
    assert "sum(name.endswith('.joblib') for name in r['models']) != 5" in code


def test_synthetic_train_rejects_nonempty_models_before_fit(package):
    isolated(package, '''import run as r
from tempfile import TemporaryDirectory
from pathlib import Path
with TemporaryDirectory() as d:
 r.MODELS=Path(d)
 (r.MODELS/'unrelated.txt').write_text('synthetic marker')
 r.prepared=lambda c:(None,None,None,None)
 try:r.train({},'numeric')
 except FileExistsError:pass
 else:raise AssertionError('nonempty model directory accepted')
''')


def test_fixed_postprocessing_recipe_matches_copied_helpers(package):
    isolated(package, '''import run as r
import copy
recipe=r.contract()['recipe']
actual=r.postprocessing_contract(recipe)
assert actual['router']['alpha']==10 and actual['shrink']['persistence_weight']==.2
for section,key,new in [('router','alpha',11),('router','strength',.8),('shrink','persistence_weight',.3),('model','single_weight',.8)]:
 bad=copy.deepcopy(recipe);bad[section][key]=new
 try:r.postprocessing_contract(bad)
 except ValueError:pass
 else:raise AssertionError('postprocessing mismatch accepted')
''')


def test_single_replay_restores_case_major_from_lead_major(package):
    isolated(package, '''import run as r
import numpy as np
import pandas as pd
meta=pd.DataFrame([dict(anchor_id=a,station='A',lead_h=h,current_hs=float(a)) for h in (3,6,9,12,18,24) for a in (2,1,3)])
residual=meta.lead_h.to_numpy()/100+meta.anchor_id.to_numpy()/10
expected=np.array([[a+(h/100+a/10) for h in (3,6,9,12,18,24)] for a in (1,2,3)])
np.testing.assert_array_equal(r.single_case_major(meta,residual),expected)
assert not np.array_equal((meta.current_hs.to_numpy()+residual).reshape(-1,6),expected)
''')


def test_source_and_shrink_provenance_guards(package):
    isolated(package, '''import run as r
from tempfile import TemporaryDirectory
from pathlib import Path
c=r.contract()
assert r.sha(r.DOCS/'shrink-provenance.json')==c['shrink_provenance_sha256']
with TemporaryDirectory() as d:
 p=Path(d);(p/'train_wave.txt').write_bytes(b'synthetic')
 c={'source_files':{'train_wave.txt':r.sha(p/'train_wave.txt')}}
 assert r.source_hashes(c,p)==c['source_files']
 (p/'train_wave.txt').write_bytes(b'changed')
 try:r.source_hashes(c,p)
 except ValueError:pass
 else:raise AssertionError('source mutation accepted')
''')


def test_python_audit_guard_rejects_old_artifacts_and_premature_official(package, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "test_context.parquet").write_bytes(b"synthetic")
    old = tmp_path / "old.cbm"
    old.write_bytes(b"synthetic")
    isolated(package, '''import run as r
from pathlib import Path
source=Path(''' + repr(str(source)) + ''')
r.access_guard(source,False)
for p in [source/'test_context.parquet',Path(''' + repr(str(old)) + ''')]:
 try:p.read_bytes()
 except PermissionError:pass
 else:raise AssertionError('unapproved read accepted')
''')
