# P2 새 후보 — 학습부터 생성까지 완료, 아직 업로드 안 함

선택모델은 **raw v23 DeepSets + T5/S5 연속결측증강, 3seed 평균**이다. 기존bin17/anchor답안, 공식점수역산보정, 과거OOF가 학습·추론의 필수입력이 아니다. 원본관측을 학습해새모델을저장한뒤 **별도Python프로세스**에서로드하여결과를생성했다. 기존사용자최종패키지는변경하지않았고이번경로는별도새후보다. 내부0.920346→0.859250℃,공식미채점,업로드0.

## 로컬 구분

| 역할 | 경로 |
|---|---|
| 배포데이터 | 환경변수 `P2_DATA_DIR`, 저장소밖운영진 `P2_profile_restore` 원본폴더(read-only immutable).수정·복사편입·재배포금지 |
| 학습/추론코드 | `scripts/run_p2_score_repair_deploy_20260905_v1.py` 및 고정research함수 |
| 고정config | `configs/experiments/p2_score_repair_deploy_20260905_v1.json` |
| 새모델3개 | `artifacts/p2_score_repair_deploy_20260905_v1/model_seed20260901.pt`, `...20260902.pt`, `...20260903.pt` |
| 정답형식후보 | `artifacts/p2_score_repair_deploy_20260905_v1/submission_p2_v23_blockmask_3seed.csv` |
| 학습영수증 | [train-result.json](train-result.json) |
| 제출형식/추론QA | [predict-result.json](predict-result.json) |
| 내부테스트결과 | [P2 research report](../p2_score_repair_20260905_v1/report-source.md) |

학습:2024-05-01~2026-01-01KST의전체eligible배포target166,268행에증강51,354행을더하되총가중치166,268유지. 3seed×60epoch, 각17.468/16.906/17.250초. 학습중공식키/공식sample값/hiddentruth접근0.

추론: 모델설정/조건을고정하고모든학습을끝낸뒤 `test_index.csv`와 `sample_submission.csv`의 **station,layer,time 키3열만** 읽음. sample의수온placeholder와공식baseline값은읽지않음. 반복inference절대오차0,CSV직렬화최대오차5.00009e−11℃.

## 어디에 어떤 파일을 쓸지

- 홈페이지 **문제2 / OCN-02 / 중간층수온연직구조복원**에 위후보CSV를선택한다. P1/P3에제출하지않는다.
- 제목:`P2 clean v23 blockmask 3seed`
- 한줄설명:배포관측만으로scratch학습한rawDeepSets3seed평균;T5/S5연속결측증강,과거답안·공식역산계수미사용.
- CSV열은 `station,layer,time,temp`,26,061행,키중복0,공식키집합일치, sample순서일치, 수온모두finite.
- SHA-256:`46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071`.
- **이문서는제출완료증명이아니다.현재업로드0.** 모델가중치·raw관측·예측NPZ·후보CSV는Git에stage하지않는다.

## 재현 명령과 주의

프로젝트루트에서 `.venv-p1`환경및 `P2_DATA_DIR`를설정한다.원본source SHA와research코드/configSHA를검증한다.

```powershell
.venv-p1\Scripts\python.exe -m pytest tests/test_p2_score_repair_20260905_v1.py tests/test_p2_score_repair_deploy_20260905_v1.py -q
.venv-p1\Scripts\python.exe scripts/run_p2_score_repair_deploy_20260905_v1.py --stage train --execute --root-qa-pass
.venv-p1\Scripts\python.exe scripts/run_p2_score_repair_deploy_20260905_v1.py --stage predict --execute
```

위두실행은이미완료되어현재artifact의attempt lock이소비됐다. **현재디렉터리에서그대로재실행/lock삭제하지말것.** 독립재현이필요하면root승인하에별도깨끗한검증위치/실행ID로동일코드·config를옮긴다.이는모델재학습성능튜닝승인이아니다.6시간제한에대한현재호스트전체실행시간은충분히짧으나최종운영진검증환경의CPU/GPU조건은따로확인해야한다.

최종패키징시추론에서import되는원본함수와연관소스모듈도함께동봉해야한다.현재research코드는기존 `p2_pipeline`, v12 `VerticalDeepSet`, `p2_restore`유틸을그대로import한다.이문서는별도공식최종ZIP완료를주장하지않는다.
