"""Wait for owned training, independent OOF QA, completion, fresh cold and ZIP replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import nbformat
import numpy as np
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
ID="p2_l120_s10_proj_20260907_v1"
OUT,REPORT=ROOT/"artifacts"/ID,ROOT/"reports"/ID
DEST=Path("C:/Users/cedis/Documents/OceanFinalDay_20260907/P2_L120_s10_proj_v1")
PARENT=Path("C:/Users/cedis/Documents/OceanFinalRelease_20260907/P2/SAVED_MODELS")
sys.path.insert(0,str(ROOT/"scripts"))
from build_final_release_20260907_v1 import archive, copy_pins, read, save, sha  # noqa: E402


def ah(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def notebook(path,modes):
    body="import os,sys,subprocess\nassert os.environ.get('P2_DATA_DIR')\n"
    for mode in modes:
        body+=f"subprocess.run([sys.executable,'-B','02_code/runner.py',{mode!r}],check=True)\n"
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# P2 L120 ten-seed plus projection\nFrozen recipe. Distributed data only. No runtime kill switch. See CURRENT_README.md."),nbformat.v4.new_code_cell(body)])
    nbformat.validate(nb)
    nbformat.write(nb,path)


def execute_notebook(package,name):
    start=time.monotonic()
    nb=nbformat.read(package/name,as_version=4)
    km=KernelManager(kernel_name="python3")
    km.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    NotebookClient(nb,km=km,timeout=None,resources={"metadata":{"path":str(package)}}).execute(cleanup_kc=True)
    path=package/"04_logs"/(name+".executed.ipynb")
    nbformat.write(nb,path)
    save(package/"04_logs"/(name+".receipt.json"),{"status":"PASS","seconds":time.monotonic()-start,"source_notebook_sha256":sha(package/name),"pid":os.getpid()})


def source_package():
    src=DEST/"SOURCE_ONLY"
    for p in ("01_data","02_code","03_model","04_logs","05_answer","06_docs"):
        (src/p).mkdir(parents=True)
    for name in ("base.py","core.py"):
        shutil.copy2(PARENT/"02_code"/name,src/"02_code"/name)
    shutil.copy2(ROOT/"scripts/p2_l120_s10_proj_20260907_v1_portable.py",src/"02_code/runner.py")
    shutil.copy2(ROOT/"scripts/p2_final_day_projection_20260907_v1.py",src/"02_code/projection.py")
    c=read(PARENT/"config.json")
    c.update(package_id=ID,seeds=list(range(20260901,20260911)),maximum_new_fits=10,wallcap_seconds=None,prediction="equal mean10; original base12g serialization; exact clip then endpoint-direction PAVA")
    save(src/"config.json",c)
    shutil.copy2(REPORT/"independent-qa.json",src/"06_docs/internal-qa.json")
    shutil.copy2(ROOT/"configs/experiments"/(ID+".json"),src/"06_docs/preregistration.json")
    for requirement in PARENT.glob("requirements*"):
        shutil.copy2(requirement,src/requirement.name)
    notebook(src/"TRAIN.ipynb",["train","trainqa"])
    notebook(src/"PREDICT.ipynb",["infer","replay"])
    notebook(src/"SAVED_PREDICT.ipynb",["trainqa","infer","replay"])
    (src/"CURRENT_README.md").write_text(
        "# P2 L120 s10 projection\n\nSet P2_DATA_DIR to the immutable organizer P2_profile_restore folder. Use the pinned Python/CUDA environment (Python 3.12, torch CUDA); no network or pretrained/external sources.\n\n"
        "SOURCE_ONLY: fresh extraction with empty03_model, run TRAIN.ipynb then PREDICT.ipynb. Ten full-data L120/120epoch DeepSets fits, seeds20260901..20260910, equal mean. CPU2 preserves prior numerical recipe while GPU0 is the dedicated accelerator. No wall timeout.\n\n"
        "SAVED_MODELS: fresh extraction, run SAVED_PREDICT.ipynb only. It verifies all10 models against full-training prediction hashes before inference, zero new fits.\n\n"
        "01_data source location (not bundled);02_code numerical code;03_model learned weights;04_logs local receipts;05_answer generated CSV;06_docs retrospective internal QA. Upload answer 05_answer/submission_p2_L120_s10_proj.csv if user selects it. SOURCE_ONLY/SAVED_MODELS are reproduction attachments, not scoring CSV. No automatic upload or final designation.\n\n"
        "Projection: complete requested layers2/3/4 only, finite T1 and first finite T5->T6->T7->T8; clip before endpoint-direction exact PAVA. Missing endpoints/incomplete groups unchanged. Target temp+psal masked before features. All fitted values originate in model files; leaderboard-derived coefficients0.\n\n"
        "Internal eight-fold/7day purge/B3 primary evaluation is previously exposed retrospective evidence, not an expected official score. Mean improvement is separate from worst-block risk. Official score unknown. General-model6h rule scope unverified; report actual measured runtime, not regulatory certification.\n",
        encoding="utf-8")
    pins={p.relative_to(src).as_posix():sha(p) for p in src.rglob("*") if p.is_file()}
    save(src/"SOURCE_MANIFEST.json",{"files":pins})
    pins["SOURCE_MANIFEST.json"]=sha(src/"SOURCE_MANIFEST.json")
    return src,pins


def empty_copy(source,target,pins):
    for p in ("01_data","02_code","03_model","04_logs","05_answer","06_docs"):
        (target/p).mkdir(parents=True)
    copy_pins(source,target,pins)


def build_completion(src,pins):
    target=DEST/"completion"
    empty_copy(src,target,pins)
    fits=[]
    for f in read(PARENT/"03_model/MODEL_MANIFEST.json")["fits"]:
        assert sha(PARENT/"03_model"/f["file"])==f["sha256"]
        shutil.copy2(PARENT/"03_model"/f["file"],target/"03_model"/f["file"])
        fits.append(f|{"normalized_sha256":f["full_training_normalized_prediction_sha256"],"reused":True})
    for f in read(OUT/"terminal_result.json")["fits"]:
        if f["fold"]!="FULL":
            continue
        name=f"model_seed{f['seed']}.pt"
        assert sha(OUT/f["model_file"])==f["model_sha256"]
        shutil.copy2(OUT/f["model_file"],target/"03_model"/name)
        with np.load(OUT/f["predictions_file"],allow_pickle=False) as z:
            absolute=ah(z["natural"])
        fits.append(f|{"file":name,"sha256":f["model_sha256"],"absolute_prediction_sha256":absolute,"reused":False})
    save(target/"03_model/MODEL_MANIFEST.json",{"status":"TRAINED","run_kind":"VERIFIED_3_PLUS_NEW_7_COMPLETION","pid":os.getpid(),"fits":fits,"new_fits":7,"reused_fits":3,"source_sha256":read(src/"config.json")["source_sha256"]})
    execute_notebook(target,"SAVED_PREDICT.ipynb")
    return target


def finish():
    save(REPORT/"FINISH_LOCK.json",{"pid":os.getpid(),"start":time.time(),"no_wall_timeout":True})
    while not (OUT/"terminal_result.json").exists():
        time.sleep(10)
    terminal=read(OUT/"terminal_result.json")
    assert terminal["status"]=="TRAINED_PENDING_INDEPENDENT_QA",terminal.get("error","training failed")
    subprocess.run([sys.executable,"-B",str(ROOT/"scripts/p2_l120_s10_proj_20260907_v1_qa.py")],check=True)
    DEST.mkdir(parents=True,exist_ok=False)
    src,pins=source_package()
    completion=build_completion(src,pins)
    first=read(completion/"04_logs/ANSWER_QA.json")
    save(REPORT/"candidate-ready.json",{"status":"INTERNAL_QA_AND_COMPLETION_REPLAY_PASS_COLD_PENDING","answer":str(completion/"05_answer/submission_p2_L120_s10_proj.csv"),"rows":26061,"sha256":first["answer_sha256"],"uploads":0})
    cold=DEST/"fresh_cold"
    empty_copy(src,cold,pins)
    start=time.monotonic()
    execute_notebook(cold,"TRAIN.ipynb")
    execute_notebook(cold,"PREDICT.ipynb")
    coldseconds=time.monotonic()-start
    second=read(cold/"04_logs/ANSWER_QA.json")
    assert first["answer_sha256"]==second["answer_sha256"],"fresh cold answer mismatch; preserve both"
    saved=DEST/"SAVED_MODELS"
    empty_copy(src,saved,pins)
    for p in (cold/"03_model").iterdir():
        shutil.copy2(p,saved/"03_model"/p.name)
    archives=[archive(src,DEST/"P2_L120_s10_proj_SOURCE_ONLY.zip"),archive(saved,DEST/"P2_L120_s10_proj_SAVED_MODELS.zip")]
    extracted=DEST/"extracted_saved_replay"
    with zipfile.ZipFile(DEST/archives[1]["file"]) as z:
        z.extractall(extracted)
    t=time.monotonic()
    execute_notebook(extracted,"SAVED_PREDICT.ipynb")
    replayseconds=time.monotonic()-t
    final=read(extracted/"04_logs/ANSWER_QA.json")
    assert final["answer_sha256"]==first["answer_sha256"]
    (DEST/"ANSWER").mkdir()
    shutil.copy2(cold/"05_answer/submission_p2_L120_s10_proj.csv",DEST/"ANSWER/submission_p2_L120_s10_proj.csv")
    result={"status":"LOCAL_CANDIDATE_COLD_AND_EXTRACTED_REPLAY_PASS","answer":str(DEST/"ANSWER/submission_p2_L120_s10_proj.csv"),"rows":26061,"answer_sha256":first["answer_sha256"],"answer_bytes":first["answer_bytes"],"new_internal_fits":56,"new_full_completion_fits":7,"reused_internal_fits":24,"reused_full_fits":3,"new_full_cold_fits":10,"total_new_fits":73,"cold_wall_seconds":coldseconds,"extracted_saved_replay_seconds":replayseconds,"archives":archives,"official_score":None,"six_hour_general_rule_scope":"UNVERIFIED","internal_qa_sha256":sha(REPORT/"independent-qa.json"),"uploads":0,"hidden_truth_rows":0,"fresh_offline_OS_tested":False}
    save(DEST/"BUILD_QA.json",result)
    save(REPORT/"result.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="archives"}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.parse_args()
    try:
        finish()
    except BaseException as e:
        failure=REPORT/"finish-failure.json"
        if not failure.exists():
            save(failure,{"status":"TERMINAL_TECHNICAL_FAILURE","error":str(e),"automatic_restart":False,"pid":os.getpid()})
        raise
