# P1 champion portable — 새 11-fit cold / 저장 모델 재생

현재: **저장 모델 ZIP→새 외부 폴더→실제 PREDICT 노트북→두 새 프로세스 답안 재생은 b2f17 exact PASS(84.813초)**. 별도 cold11fit 학습은 모두 완료해 배포7모델 SHA를 원본과 정확히 복원했으나, MS 재생 QA의 실행 소스 경로 오류로 TRAIN 노트북이 기술 실패했다. **Cold 전체 자동실행/공식 답안/PREDICT PASS는 아니다.** 기존 b2f17 모델·답안·31-fit 연구 seal과 만료 시계는 변경하지 않았다.

저장 모델 검증에서 169,011행·양성6,843·tree양성6,640·MS양성2,853·추가203·원 tree양성 제거0이었고, 생성 PID39112와 재생 PID19240의 SHA가 모두 원본과 정확히 일치했다. 학습0, sample label/hidden/과거답안 값 읽기0, upload0. 원본 노트북은 불변이고 실행본만 별도 보존했다.

## 고정 범위

사용자/Root 승인에 따른 새 배포 재생성 계약은 Q4 earlier-inner O1/B3 4 fits → 동일 train-only selector → 전체 released train O1/B3 4 fits → original MS-TCN full3, **합계 11 fits**다. 기존 연구 CLI의 24 historical+4full+3MS를 재시작하지 않고 배포에 필요한 Q4 inner 함수만 호출한다. 소스·features80·year-depth train-only·168h cap·가중치·seeds·700trees·decoder는 원형을 그대로 재사용한다. 실제 tree arm은 고정 union이며 generic cell selector는 계산하되 추론에 사용하지 않는다.

MS는 기존165features/width512/e150/300epoch horizon/3seeds/0.8,0.4,snap12,min19를 그대로 사용한다. Tree CPU4, MS CPU2+독점 CUDA0 bf16. 새 전체 clock에 prepare·11fit·모델 QA·공식 추론·다른 PID 답안 replay를 모두 포함하며 상한6시간. 기존 MS full3 실측6215.835초, source→replay6432.701초를 근거로 전체약2시간 예상이나 실제로 입증되기 전 보장하지 않는다. 소스/환경/자원 변경에 따른 byte 재현성과 자체 saved replay는 분리한다.

## 사전 검증

- [전용 구현·README](../../scripts/portable_20260906/P1_champion_portable_v1/README.md), [합성20 PASS](pytest.xml), Ruff PASS. Q4-only와 원3fold splitter의 동일 mask/support,21day/whole-run 배제,feature label sentinel,실제1+3seed 회계,3seed MS 원형 평균→type conditioning→decoder toy 동등성을 검증했다.
- saved 원본 SHA 불일치는 FAIL receipt로 남긴 뒤 중단한다. cold 새 답안의 자체 byte replay와 과거 b2f17 exact 여부는 별도 필드로 기록한다. 결과에 맞춘 threshold 조정·자동 재시도는 없다.
- 두 ZIP의 빈 03_model/04_logs/05_answer directory entry를 보존했다. 실제 새 OS temp에 추출하고 별도 PID에서 import: cold tree14108, cold MS28684(`cuda_initialized=false`), saved tree17596. source pin113개 및 관련 모듈16개가 새 package 안을 가리킴을 검사했다. 이 검사는 모델 학습/공식 추론이 아니다.
- root 제공 TRAIN/PREDICT 노트북도 archive manifest에 포함했다. `P1_DATA_DIR` 또는 명시적 `--data`만 받으며 개인정보 절대경로를 notebook에 넣지 않는다.
- 위 **사전 검증 단계**는 fit0/official0였다. 이후 saved replay는 승인된 test 관측과 sample키만 사용했다. hidden0/upload0/Git0는 유지한다. Python 감사 이벤트로 네트워크와 파일경계를 제한하지만 실제 OS 차단망·새 OS/다른 GPU·offline wheelhouse 설치는 미검증이다.

## 보존·실행 경로

| 종류 | ZIP | SHA256 |
|---|---|---|
| 빈 모델 재학습 source | [P1_CHAMPION_COLD_SOURCE.zip](../../artifacts/p1_champion_portable_20260906_v1/P1_CHAMPION_COLD_SOURCE.zip),786029bytes | `d026592f087110a3791eaad0289376d33005b0605f8395b870459e7eabf2c8d2` |
| 기존 모델 장기 재생 | [P1_CHAMPION_SAVED.zip](../../artifacts/p1_champion_portable_20260906_v1/P1_CHAMPION_SAVED.zip),592601203bytes | `610701bc2612edfdff842fc985c5c74f916f9134f3896e1c2021e29705fcbb31` |

source manifest `1905befb5cab08dce7fed6130066a6183bf8e8dbb8e583a55e2216bb4ca8ffa6`; saved manifest `083835f98d4b6d777d827ad6f20a93c1165b00887d6892f95cb299f4ac06a0e7`.

검증용 외부 추출 root는 `C:/Users/cedis/AppData/Local/Temp/p1_champion_portable_73eac704ce264505882a06ace27e1168/`이고 `cold/`, `saved/`가 분리돼 있다. Saved 완료본은 `C:/Users/cedis/Documents/OceanFinalCandidates_20260906/P1_champion_saved_verified_v1/`에 복사·SHA 확인하여 보존했다. Root [분할 재조립 QA](../parallel_completion_training_20260906_v2/p1-upload-parts-qa.json)는 saved ZIP의15part→원byte/SHA 복원 PASS이며 새로운 cold 완료 증거는 아니다.

Saved receipt SHA: `06_docs/answer.json` = `8551adfa655014147c379091b056d090e99b365ec0cf81571ba530737bd20dce`; `answer-replay-qa.json` = `894ea4ad489ef114ee597c5b305432413df21f36e12510740dd0e9aa199c7867`; 실제 notebook receipt = `9fcab6b6aeb021410c8d33a5f46ae334687c13dc3535c3b3c81c260d74647eb5`.

Cold는 saved PASS 후 16:21:47.872KST에 빈03_model에서 실제 TRAIN→PREDICT 노트북을 **한 번** 시작했다(기한22:21:47.872KST). tree8fit·fresh PID24868의15check QA PASS; MS3seed도 e150/각4050steps/nonfinite0으로 완료했다(MS worker37180,6179.242초). full tree4 및 MS3의 배포 모델 SHA는 원본과 전부 동일하다. Q4 selector JSON 내용도 동일하고 파일 SHA 차이는 원본CRLF120개 대 새LF 직렬화만이며, CRLF 정규화 후 byte가 동일하다. 사용량 제한 중에도 기존 프로세스가 계속 실행됐으며 재시작하지 않았다.

## Cold v1 기술 실패 — 자동 재시작 없음

학습 이후 `ms-qa` worker24688은 원형 `mstcn.replay()`를 호출했다. 이 원형은 자기 MS 출력 root만 허용하는 file audit hook을 설치하고 마지막 `verify_owned()`에서 실행 모듈 자체를 다시 hash한다. 어댑터가 **바깥 `02_code/scripts/.../mstcn.py`를 import하여 실행한 것**이 잘못이었다. 동일 SHA의 모듈이라도 그 위치는 MS 소유 root 밖이므로 `PermissionError: File outside source-only boundary: mstcn.py`가 발생했다. 올바른 실행 소스는 새 학습이 생성한 `03_model/mstcn/02_code/mstcn.py`다. 모델/확률 실패로 단정하지 않으며, 기록되지 않은 probe QA를 PASS로 추정하지 않는다.

- 결과: `train-terminal.json` TERMINAL_TECHNICAL_FAILURE, MS `terminal.json` TERMINAL_REPLAY_FAILURE; cold `fresh-process-replay.json`, `training-qa.json`, 공식 CSV는 미생성. 공식/hidden/CSV/upload0.
- 학습·source·model·lock·failed terminal·source ZIP을 그대로 보존했다. 실패 실행 전체도 `C:/Users/cedis/Documents/OceanFinalCandidates_20260906/P1_champion_cold_attempt_v1/`에 복사하고 manifest/MS result/두 failed terminal SHA 일치를 확인했다. 모든 관련 Python이 종료됐고 GPU를 root에 해제 통지했다. 새 QA ID에서 학습0으로 owned snapshot 실행을 검증하는 기술 복구를 제안했으나, 기존 sealed attempt를 재시작하거나 경계를 넓혀 우회하지 않는다.
- SHA256: MS training result `1ee0b4737531ef525b232cc0e9174f4729dc8ed0b464010957c374389c528fe5`; MS failure `1679999e20155d298d9ca427ef7dc26cfb342bcec0eae5050bd9093ec57364e6`; TRAIN failure `0064b3262fb98e7fac30328ac3265fe1037d47f10950e8d66d5c91419de18d41`; executed TRAIN receipt `05ceb3f66205ed90ab61b2b9ccf574cddd2c4b1fa50e297b151ecaf9554161bb`.

원본 답안 값은 입력으로 읽지 않으며 목표 SHA `b2f17f5cda8030cb3d97fbb504e6babb6aef8ba7fe555092901479677af0625e`만 메타데이터로 비교한다. 업로드와 최종 모델 잠금은 이 스크립트가 수행하지 않는다. 저장 ZIP 재생 PASS를 이 cold 실패의 완료 표시로 전용하지 않는다.
