# Portable 기준 패키지 — 사용/재학습 안내

2026-09-06 검증 완료본. P1/P2/P3 로컬 빈 폴더 재생성 및 답안 검증을 완료했다. P3는 별도 복구 비교와 최신 ZIP의 실제 추출·저장 모델 추론까지 검증했다. [CLEANROOM_RESULT](CLEANROOM_RESULT.md)는 검증 범위와 한계를 기록한다. **답안 CSV 채점, 재현 ZIP 첨부, 최종 모델 지정은 서로 다른 작업이다.** 이번 작업에는 업로드/최종 잠금/commit/push가 없다.

## 파일 선택

아래 경로는 저장소 `C:/Users/cedis/PycharmProjects/PythonProject` 기준이다.

| 문제 | 답안 채점용 CSV | 코드+모델 재현용 ZIP | 주의 |
|---|---|---|---|
| P1 / OCN-01 | `artifacts/portable_cleanroom_20260906_v1/P1/run_a/05_answer/P1_submission.csv` | 같은 `run_a/06_docs/P1_PORTABLE_REPRODUCIBILITY.zip` | 169,011행. run_b는 독립 재검증본, 새 후보가 아님 |
| P2 / OCN-02 | `artifacts/portable_cleanroom_20260906_v1/P2/05_answer/submission_p2_clean_C3.csv` | 같은 `P2/06_docs/P2_PORTABLE_REPRODUCIBILITY.zip` | 26,061행. replay CSV는 검산본, 새 후보가 아님 |
| P3 / OCN-03 | `artifacts/portable_cleanroom_20260906_v1/P3/v2_run_b_completed/05_answer/submission.csv` | `artifacts/portable_cleanroom_20260906_v1/P3/archives_v3/`의 아래 두 ZIP | 1,200행. run_b는 빈 폴더 전체 재생성본, recovery는 별도 연구 검증본 |

원본 배포 데이터는 재현 ZIP에서 제외했다. 원본을 `P1_DATA_DIR`, `P2_DATA_DIR`, `P3_DATA_DIR`로 지정한다. `01_data`=원본 참조, `02_code`=코드, `03_model`=이번 학습 산출물, `05_answer`=그 모델의 추론 답안, `06_docs`=실측/계보/QA이다. 원시 데이터·가중치·답안·ZIP은 로컬 보존 대상이며 Git에 넣지 않는다.

## P1 — 원 저장소 없이 빈 폴더부터 재학습

완성 패키지/ZIP의 기존 모델을 삭제하지 않는다. 아래처럼 새 경로에 코드와 README만 복사한다. ZIP을 쓴다면 먼저 풀어서 그 안의 `P1`을 `$verifiedP1`로 지정한다. 코드의 requirements는 검증 환경의 버전 목록이며 본 시험은 Python **3.12.10**에서 수행했다. 새 venv/OS 차단망 시험과 wheel bundle 준비는 아직 별도다.

```powershell
$verifiedP1 = 'C:\path\to\verified\P1'
$coldP1 = 'C:\path\to\new\P1_cold_run'
if (Test-Path -LiteralPath $coldP1) { throw '새 빈 경로를 지정하세요' }
New-Item -ItemType Directory -Path $coldP1 | Out-Null
Copy-Item -LiteralPath (Join-Path $verifiedP1 '02_code') -Destination $coldP1 -Recurse
Copy-Item -LiteralPath (Join-Path $verifiedP1 'README.md') -Destination $coldP1
foreach ($folder in '01_data','03_model','04_logs','05_answer','06_docs') {
    New-Item -ItemType Directory -Path (Join-Path $coldP1 $folder) | Out-Null
}
$env:P1_DATA_DIR = 'C:\path\to\organizer_P1_dataset'
Set-Location -LiteralPath $coldP1
python 02_code/run.py self-test
if ($LASTEXITCODE) { throw 'self-test 실패' }
python 02_code/run.py train
if ($LASTEXITCODE) { throw '학습 실패: 기존 산출물을 보존하고 조사' }
python 02_code/run.py infer
if ($LASTEXITCODE) { throw '추론 실패' }
python 02_code/run.py verify
if ($LASTEXITCODE) { throw 'replay 불일치' }
Get-FileHash -Algorithm SHA256 -LiteralPath '05_answer/P1_submission.csv'
```

정상 종료 기준 SHA는 `5971e145f1ac38b8ee3e34cfd302973ba7a64b8873db11c354d3331221fdb28a`. 불일치하면 검증 실패를 기록한다. 과거 답안을 복사하거나 점수에 맞춰 threshold를 조정해 SHA를 맞추지 않는다. 원 저장소 코드 접근 차단 시험을 반복하려면 환경변수 `P1_DENY_REPO`에 원 저장소 경로를 지정한다(배포 데이터와 실행 venv는 예외). Python audit hook은 OS 차단망을 대신하지 않는다.

## P2

완료 패키지 root에 포함된 `build_package.py --output <새폴더>`로 코드/config/문서만 복사하여 시작한다. `06_docs/RETRAIN_FROM_EMPTY.md`와 README의 명령을 그대로 따른다. 이번 실측은 CUDA, CPU thread 1 및 기존 v6 3-seed/60 epoch 설정이다. 새 CPU-only 레시피로 조용히 바꾸지 않는다. 재학습/추론/replay 과정이 끝나면 SHA `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071` 및 QA를 대조한다.

P2 ZIP의 `04_logs/*_ATTEMPT_LOCK.json`은 완료된 검증 실행의 영수증이다. 기존 경로에서 학습을 반복하려고 삭제하지 않는다. 위 `build_package.py`는 코드·config·문서의 허용목록만 새 폴더로 복사하며 모델·답안·이전 lock은 복사하지 않는다. 따라서 새 전체 재학습은 항상 이 새 폴더에서 시작한다.

## P3 — 전체 학습과 저장 모델 재생 구분

최신 검증본은 `artifacts/portable_cleanroom_20260906_v1/P3/archives_v3/`다. 이전 `archives/`의 ZIP은 이력이며 선택하지 않는다.

- `P3_cold_start_code_v2.zip`: **빈 모델부터 전체 학습**용 코드. 새 빈 폴더에 풀고 `P3_DATA_DIR`를 지정한 뒤 포함 README의 `--RUN_TRAINING --gpu-approved` → `--replay` → `audit.py --training-only` → `--RUN_INFERENCE --official-approved` → `--verify-answer --official-approved` → `audit.py` 순서로 실행한다. 모델·답안은 미동봉이며 학습으로 생성한다. 코드 ZIP SHA `7629ff948dd5c939d4100b8fa712f68f18dfddab8bd5c471136b20eb443bbfdb`.
- `P3_saved_model_replay_v2.zip`: **이미 학습한 모델 재생**용. 새 폴더에 풀고 같은 환경/데이터 경로를 지정한 뒤 `python -I <package>/02_code/archive_infer.py --official-approved`만 실행한다. 결과는 `05_answer/replayed_submission.csv`다. 원 `run.py`의 학습/검산 명령을 이 비어 있지 않은 경로에 실행하지 않는다. 저장 모델 ZIP SHA `bb4367cd6e291cb6cb373cb41669a1f3c280e6410a077676937fbe9f71abb4c6`.

저장 모델 ZIP을 실제 새 OS 임시 폴더에 풀고 PID40344에서 추론3.668초/학습0회로 1,200행 SHA `6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7`를 재현했다. 이는 저장 모델 재생 검증이지 새 전체 재학습 1회를 추가한 것이 아니다. 원 학습 코드/manifest는 불변이며 별도 추론 어댑터도 해시 검증한다.

저장 모델 ZIP에는 고정 로더의 해시 확인에 필요한 이번 학습 모델9개와 train-derived probe 특징·모델 예측·열 목록이 포함된다. 원시 배포 데이터, 학습 target/OOF truth 테이블, credentials는 포함하지 않는다. 이 파생 probe/모델/ZIP도 Git에는 넣지 않는다. 오프라인 dependency wheel 묶음과 새 PC 시험은 미완료다. [P3 상세 보고서](../../reports/portable_cleanroom_20260906_v1/P3/report-source.md)와 [ZIP manifest QA](../../reports/portable_cleanroom_20260906_v1/P3/archive-build-v3-qa.json)를 참조한다.

## 제출 전 마지막 확인

후속 P1 bracket-only 후보가 별도로 학습·답안 생성·독립 검증을 완료했다. 신규 SHA `9031c84e…ad93a`의 정확한 경로와 범위는 [P1 후보 안내](P1_BRACKET_CANDIDATE_HANDOFF_20260906.md)를 따른다. 아직 공식 채점/영구 재현 패키지 교체는 하지 않았으며, 위 표의 기준 답안·패키지는 fallback으로 그대로 보존한다.

이미 채점된 SHA와 같은 P1/P2/P3 기준 답안이다. 같은 답안을 다시 올려 새 성능 증거를 얻었다고 하지 않는다. 개선 실험이 만든 후보는 이 기준 패키지와 섞지 않고 새 계보/내부 검증/재현/공식 채점 절차로 판단한다. 포털에 실제 올릴 때는 [현재 실행서](../OFFICIAL_SUBMISSION_RUNBOOK_20260905.md)에 따라 당일 횟수·마감·첨부 제한과 사용자 승인 범위를 확인한다. 모델 잠금 전에는 특히 후속 업로드 제한을 확인한다.
