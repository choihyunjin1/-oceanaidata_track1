# P3 clean 기준선 빈 모델 폴더 재생성

현재 실행 상태의 단일 근거는 이 폴더의 `prepare.json`, `training-result.json`, `fresh-process-replay.json`, `answer-qa.json`, `answer-replay-qa.json`, `independent-qa.json`이다. 존재하지 않는 단계는 완료로 해석하지 않는다. 이 작업은 **기존 기준선의 재생성 검사**이고 새로운 성능 개선/업로드/최종 운영진 검증을 뜻하지 않는다.

## 고정 정책과 계보

`p3_corrected_repeated_forward_catboost_v2.json`의 source-only residual single/multi CatBoost를 재학습한다. 역사 검증3fold×2모델=6fit, 전체 데이터 single/multi2fit, 이전 완료 fold만 쓰는 router2fit 및 전체 OOF router1fit이다. 총 CatBoost8 + router3이다. TabPFN 후보는 완전히 제외해 합성 사전학습 가중치도 읽거나 복사하지 않는다.

`alpha=10`은 `src/p3_wave/loss_router.py`의 Ridge 정규화 계수다. 12/18/24시간의 persistence shrink0.2는 고정 v2 연구 recipe이며 이번 결과나 공식 점수로 역산하지 않는다. 과거 refined-alpha/axis contract/기존 model·OOF·답안은 실행 입력이 아니다. 현재 known-clean 내부 RMSE와 답안 SHA는 **출력 대조용 metadata**일 뿐 학습/예측에 사용되지 않는다. 값이 다르면 차이를 보고하며 맞추기 위한 재튜닝은 없다.

## 분리된 로컬 구조

저장소 root 기준 `artifacts/p3_clean_regeneration_20260905_v4/`:

| 폴더 | 내용 |
|---|---|
| `01_data/` | 허용된 배포 train_wave/train_atmos CSV의 SHA 동일 복사본. Git 제외 |
| `02_code/` | 새 runner/config와 고정 의존 코드의 SHA 봉인 snapshot. 실행 source와 매 단계 대조 |
| `03_model/` | 시작 시 비어 있었음. 실제 재학습한 historical6 및 full single/multi/router 모델 |
| `04_validation/` | 원본에서 새로 계산한 특징/anchor/key/OOF, 학습 모델 검증용 local replay 자료 |
| `05_answer/submission.csv` | 별도 승인된 fresh-process 추론으로 생성한 P3 리더보드 답안. 업로드하지 않음 |

공식 추론 입력은 `P3_DATA_DIR`의 `test_context.parquet`와 `test_index.csv`에서 공개 predictor/key 허용 열만 읽는다. sample/hidden/baseline CSV는 사용하지 않는다. 원본 공식 파일은 `01_data`에 복사하지 않는다.

## 실행 입구

PowerShell에서 저장소 root(`C:\Users\cedis\PycharmProjects\PythonProject`) 기준이다. **이미 실행한 같은 폴더에서 아래 training/prepare를 재실행하지 않는다.** lock·모델·답안 삭제로 재시작하지 말고 다른 새 재현 checkout/출력 계약을 명시적으로 준비한다. 이번 실행은 이 entrypoint를 사용해 실제 빈 모델 폴더에서 검증한다.

```powershell
$env:P3_DATA_DIR = 'C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast'
# 신규 빈 출력 경로일 때만 source prepare와 실제 학습. GPU 독점 승인 필요.
.venv-p1\Scripts\python.exe scripts/run_p3_clean_regeneration_20260905_v4.py --RUN_TRAINING --gpu-approved

# 반드시 학습 종료 후 별도 프로세스의 저장 모델 재생.
.venv-p1\Scripts\python.exe scripts/run_p3_clean_regeneration_20260905_v4.py --replay
.venv-p1\Scripts\python.exe scripts/qa_p3_clean_regeneration_20260905_v4.py --training-only

# 별도 공식 local-inference 승인 후, fresh process로 답안 생성.
.venv-p1\Scripts\python.exe scripts/run_p3_clean_regeneration_20260905_v4.py --RUN_INFERENCE --official-approved

# 다시 별도 프로세스에서 CSV bytes까지 exact 재생 검사(추가 학습0).
.venv-p1\Scripts\python.exe scripts/run_p3_clean_regeneration_20260905_v4.py --verify-answer --official-approved
.venv-p1\Scripts\python.exe scripts/qa_p3_clean_regeneration_20260905_v4.py
```

GPU 예약 전에 CPU source 재처리만 분리할 경우 `--prepare`를 먼저 실행한다. 이후 `--RUN_TRAINING --gpu-approved`는 이 새 폴더에서 방금 만든 자체 prepare 자료만 사용한다. 기존 연구 cache를 가져오는 resume이 아니다.

이번 실행의 CPU 예산2threads는 모델 설정·BLAS 환경변수로 지정한다. 역사 helper의 일부 native predict 기본값을 포함해 상한을 강제하려고 training 프로세스 OS affinity도2logical CPUs로 제한한다. GPU 설정은 기존 MultiRMSE 학습 경로 그대로이며 새 가설/모델 변경이 아니다. 외부 장치의 GPU/CPU 성능 차이와 프로세스 예약 대기시간은 모델 실행 시간과 구분한다.

## 검증과 제출 범위

1. SHA 동일한 배포 train 두 파일에서 24,360anchors/591features를 새로 만든다. 기존 feature/OOF/model/answer 읽기0을 경로 가드와 단계 봉인으로 제한한다.
2. 동일181cases/1,086행의 six-lead historical 평가, 78h 정점별 간격 및 episode 분리, strictly past-only router를 재생성한다.
3. full2모델과 전체 OOF router를 새로 학습하고 저장한다. 별도 프로세스에서 181localcases 예측을 최대오차0으로 재생한다.
   공식 입력을 열기 전에 `--training-only` 독립 QA로 새 OOF target/SSE/시각순서/모델·source hash까지 먼저 검산한다.
4. `05_answer/submission.csv`는 `case_id,station,lead_h,hs_pred`, 1,200행, 키/순서/중복/finite0..30/SHA QA 대상이다. **P3/OCN-03 파고 예측 답안**이다. `03_model` 파일을 리더보드 CSV 입력란에 올리지 않는다.
5. `--verify-answer`와 독립 QA를 통과해도 업로드는 별도 root 작업이다. 운영진 최종 제출용 코드·환경 ZIP, OS 네트워크 차단, 다른 머신/경로 이동, 운영진 하드웨어6시간 검증은 별도이며 자동 완료 표시하지 않는다.

현재 runner는 이 저장소 상대경로를 기준으로 실행되며 `02_code`는 봉인 snapshot이다. snapshot만 떼어 다른 곳에 복사한 최종 portable ZIP이 검증된 것은 아니다. 기존 `03_model`에서 추론만 재현한 PASS와 **이번 빈03_model에서 실제 학습→답안 생성 PASS**도 서로 다른 행으로 보고한다.

이번 `02_code`는 hash 검증 편의를 위해 `src/p3_wave` 전체를 담은 **넓은 연구 소스 snapshot**이다. 따라서 실행하지 않은 과거 ERA5/Chronos/Public계보 연구 코드도 텍스트로 포함된다. 코드의 존재를 해당 외부 관측·가중치 사용으로 해석하지 않으며, 새 runner의 active 호출 경로에는 그 실험의 실행/모델·데이터 로딩이0이다. 운영진 제출용 portable package를 만들 때는 실제 실행 의존성만 선별해야 한다. 현재 봉인 snapshot을 삭제·수정하거나 최종 제출 pack으로 표시하지 않는다.

소스/모델/OOF/답안·로그·lock은 Git 제외 연구 artifact다. 현재 worktree나 이전 최종 제출 폴더는 삭제/덮어쓰기하지 않는다.
