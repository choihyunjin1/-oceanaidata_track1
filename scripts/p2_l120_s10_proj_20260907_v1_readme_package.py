"""Add required README filename in a new package only; zero fits, exact ZIP replay."""
from __future__ import annotations

import importlib.util
import json
import shutil
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports/p2_l120_s10_proj_20260907_v1"
spec=importlib.util.spec_from_file_location("p2_finish_helper",Path(__file__).with_name("p2_l120_s10_proj_20260907_v1_finish.py"))
f=importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
SOURCE=f.DEST
DEST=SOURCE.with_name("P2_L120_s10_proj_v2")


def run():
    old=f.read(SOURCE/"BUILD_QA.json")
    assert old["status"]=="LOCAL_CANDIDATE_COLD_AND_EXTRACTED_REPLAY_PASS"
    DEST.mkdir(parents=True,exist_ok=False)
    archives=[]
    for role in ("SOURCE_ONLY","SAVED_MODELS"):
        src,target=SOURCE/role,DEST/role
        original=next(a for a in old["archives"] if role in a["file"])
        assert f.sha(SOURCE/original["file"])==original["sha256"]
        f.empty_copy(src,target,original["member_sha256"])
        shutil.copy2(target/"CURRENT_README.md",target/"README.md")
        manifest=f.read(target/"SOURCE_MANIFEST.json")
        for p,h in manifest["files"].items():
            assert f.sha(target/p)==h
        manifest["files"]["README.md"]=f.sha(target/"README.md")
        # New destination only, never overwrite or relabel v1 evidence.
        (target/"SOURCE_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        for path in (target/"02_code").glob("*.py"):
            assert f.sha(path)==f.sha(src/"02_code"/path.name)
        assert f.sha(target/"config.json")==f.sha(src/"config.json")
        archives.append(f.archive(target,DEST/f"P2_L120_s10_proj_{role}.zip"))
    extracted=DEST/"extracted_saved_replay"
    with zipfile.ZipFile(DEST/archives[1]["file"]) as z:
        z.extractall(extracted)
    start=time.monotonic()
    f.execute_notebook(extracted,"SAVED_PREDICT.ipynb")
    seconds=time.monotonic()-start
    replay=f.read(extracted/"04_logs/REPLAY_QA.json")
    assert replay["answer_sha256"]==old["answer_sha256"]
    (DEST/"ANSWER").mkdir()
    answer=DEST/"ANSWER/submission_p2_L120_s10_proj.csv"
    shutil.copy2(Path(old["answer"]),answer)
    assert f.sha(answer)==old["answer_sha256"]
    result=old|{"answer":str(answer),"archives":archives,"status":"README_AMENDMENT_EXTRACTED_REPLAY_PASS","parent_build_sha256":f.sha(SOURCE/"BUILD_QA.json"),"amendment_new_fits":0,"model_code_config_changed":False,"doc_amendment_extracted_replay_seconds":seconds,"readme_present_in_both_zips":True}
    f.save(DEST/"BUILD_QA.json",result)
    f.save(REPORT/"packaging-amendment.json",result)
    lines=["# P2 L120 10seed + 투영 — 제출용 안내", "", "최종 지정/업로드는 사용자가 선택합니다. 공식 점수는 미확인입니다.", "",
        "## 답안 채점", "", f"파일: `{answer}`", f"행 수:26061 / bytes:{old['answer_bytes']} / SHA256:{old['answer_sha256']}",
        "제목: P2_L120_s10_proj_20260907", "한줄요약: 배포 공개층만으로 처음부터 학습한 L120 DeepSets 10seed 동등평균에 동일 시각 공개 endpoint clip/PAVA를 적용한 후보.", "",
        "## 모델 재현 첨부", "", "SOURCE_ONLY는 빈 모델 폴더부터 TRAIN.ipynb→PREDICT.ipynb를 실행합니다. SAVED_MODELS는 새 압축 해제 폴더에서 SAVED_PREDICT.ipynb를 실행합니다. 각 ZIP의 README.md를 먼저 읽으세요. 배포 데이터는 동봉하지 않고 P2_DATA_DIR로 지정합니다.", ""]
    for a in archives:
        lines.extend([f"- `{DEST/a['file']}`",f"  bytes:{a['bytes']} / SHA256:{a['sha256']}"])
    lines.extend(["",f"우리 환경 실측 전체10fit+학습QA+추론/재생 {old['cold_wall_seconds']:.3f}초. README 보완 ZIP 새 추출 재생 {seconds:.3f}초. 일반 모델에 대한 공식6시간 규정 적용 범위는 미확인입니다.",
        "", "내부 pooled RMSE1.236732066→1.216631946℃로 개선, B3주평가0.463529468→0.471278625℃로 악화. 사전등록 B3 승격 기준은 미통과이며 평균 개선 가능성 비교 후보로 보존합니다. 공개 점수로 값을 조정하지 않았습니다.",
        "", "제출 모달 첨부 수/크기 제한, 정확한 마감 시각, 최종 잠금 여부는 사용자가 확인해야 합니다. 이 작업은 업로드·삭제·최종제출 클릭을 수행하지 않았습니다."])
    (DEST/"FORM.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    shutil.copy2(DEST/"FORM.md",DEST/"README.md")
    print(json.dumps({k:v for k,v in result.items() if k!="archives"}))


if __name__=="__main__":
    run()
