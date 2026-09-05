# P3 portable clean reproduction — 완료

**원 연구 저장소 밖의 빈 모델 폴더에서 배포 원자료부터 학습한 run_b가 20.94분 만에 clean 답안 1,200행을 그대로 재생성했다.** 답안 SHA는 `6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7`이며 full QA 42/42 PASS, 별도 PID 답안 재생도 exact이다. 중단된 run_a의 새 모델을 이어 쓴 recovery도 44/44 PASS 및 동일 답안이다. 이는 **uninterrupted cold-start 1회 + 별도 recovery 1회**이며 두 번의 uninterrupted cold-start 성공은 아니다.

최신 저장 모델 ZIP을 실제 새 폴더에 풀어 별도 PID 40344에서 3.668초 동안 무학습 추론한 결과도 같은 SHA였다. 최종 패키지·receipt·모델 hash QA는 **134/134 PASS**이다. 이번에는 기존 clean 결과의 재현 경로를 완성했으며 점수 개선을 주장하지 않는다. 업로드·최종 모델 잠금은 하지 않았다. 사용 파일은 [ARCHIVES_CURRENT.md](ARCHIVES_CURRENT.md)의 답안 하나와 최신 역할별 ZIP 두 개로 한정한다.

## 고정 범위

- 기준은 P3 clean v4의 3-window/181 case/1,086 lead-row 정책이다. 별도 CPU-only 5-fold numeric-lead 실험을 재시작하거나 해당 성능으로 대체하지 않는다.
- pristine run_b는 배포 train_wave.csv/train_atmos.csv에서 24,360 anchor × 591 feature를 다시 생성하며 이전 모델, OOF, cache, 답안을 읽지 않는다. 별도 recovery만 중단 직전에 이 사이클에서 새로 만든 7개 모델 및 own feature/OOF를 exact-hash 재사용한다. legacy/public-inverse 모델·계수·답안 재사용은 없다.
- 원래 각 full 실행은 historical CatBoost 6 + full-data CatBoost 2 = backbone 8 fits; earlier-OOF loss router 2 + full router 1 = 3 small fits였다. 실제 실행은 run_a **7 successful + 1 interrupted backbone attempt**, run_b 8 fits, recovery 새 full multi 1 + full router 1이다. 전체 16 successful backbone + 1 interrupted attempt = 17 started attempts, router 총 6회이며 중단 fit를 숨기지 않는다. 단순 single/multi 등가 평균만이 아니라 새 OOF loss router 및 고정 장기 persistence shrink 0.2를 포함한다.
- CPU threads 2, multi CatBoost GPU 독점. 각 실행의 source-to-answer 6시간과 2회 검증 총시간을 구분한다. 실행 중 모델·seed·분할·비율·허용치를 바꾸지 않는다.
- 학습 공식 입력 0. training-only 독립 QA 후에만 승인된 local inference에서 공개 context 열과 index key를 읽는다. sample 값·hidden truth·이전 답안·외부 자료·upload는 0이다.

## 패키지와 실행 진입점

현재 pristine 패키지는 `scripts/portable_20260906/P3/technical_repair_v2/`이다. manifest에 열거한 로컬 코드만으로 실행하며, runtime `p3_wave` import를 금지한다. 원본 source 경로는 출처 메타데이터일 뿐 runtime import나 탐색 경로가 아니다.

`01_data/`에는 배포 데이터 참조 및 해시만 둔다. `02_code/`, 처음 비어 있는 `03_model/`, `04_logs/`, `05_answer/`, `06_docs/`를 구분하며 원자료는 배포하지 않는다. `P3_DATA_DIR`로 사용자가 배포본 경로를 제공한다.

```text
python -I <package>/02_code/run.py --RUN_TRAINING --gpu-approved
python -I <package>/02_code/run.py --replay
python -I <package>/02_code/audit.py --training-only
python -I <package>/02_code/run.py --RUN_INFERENCE --official-approved
python -I <package>/02_code/run.py --verify-answer --official-approved
python -I <package>/02_code/audit.py
```

실제 학습은 OS temporary directory의 서로 다른 `run_a`, `run_b`에서 수행했다. 완료 cold-start는 `artifacts/portable_cleanroom_20260906_v1/P3/v2_run_b_completed/`, 완료 recovery는 같은 부모의 `v3_recovered_completed/`, 중단된 v2 run_a는 `v2_run_a_interrupted/`에 보존했다. 기존 실행이나 lock을 삭제·덮어써 재시작하지 않았다. pristine v2의 `RUN_TRAINING`은 prepare가 없으면 원자료 준비도 수행한다. `05_answer/submission.csv`는 P3 / OCN-03용 1,200행 `case_id,station,lead_h,hs_pred`이며, 여기서는 재현 검증용 local 파일이다. `recovery_v3`는 연구용 중단 복구 도구이고 빈 `03_model`에서 동작하는 최종 사용자용 진입점이 아니다.

두 최신 ZIP은 `artifacts/portable_cleanroom_20260906_v1/P3/archives_v3/`에 있다. `P3_cold_start_code_v2.zip`은 모델 없는 전체 재학습 패키지다. `P3_saved_model_replay_v2.zip`은 저장 모델 재생 전용으로 **`python -I <package>/02_code/archive_infer.py --official-approved`만 제공 명령**이다. 저장 ZIP의 기존 `run.py`, manifest와 receipt는 불변이며, 기존 `--replay`의 exclusive-write receipt와 원 학습 시각을 사용하는 `--verify-answer` 때문에 이를 미래 재생 명령으로 안내하지 않는다. 독립 보조 엔트리는 고정 모델/recipe/원 코드/새 companion manifest를 검증하고 원래 `official_frame` 및 predict 함수를 그대로 호출한다. 이전 답안 CSV는 읽지 않고 기록된 SHA만 대조한다.

저장 ZIP에는 원 `load_models`의 hash 검사를 만족시키는 9개 모델과 3개 train-derived probe 파일을 동봉했다. `replay_cases.parquet`는 과거 특징, `replay_expected.npz`는 모델 예측, `feature_columns.json`은 schema이며 target truth가 아니다. raw CSV, 전체 train feature/anchor, OOF target table은 제외했다. 모델·probe·ZIP은 Git 제외다. cold ZIP의 `03_model/04_logs/05_answer`가 실제 압축 해제 후 모두 비어 있고, 두 ZIP 모두 `01_data`부터 `06_docs`까지 존재함을 확인했다.

## 동등성 및 재현 판정

패키징 전 선택한 원본 함수·상수 AST 60개를 비교해 동일함을 확인했다. 기능 변경 없이 필요한 함수와 상수만 `p3_clean`으로 복사했다. original clean config SHA는 `e5c2eff7bc9fcd44759d0bc30d965c86eca10c038807f186b1379131aed9b169`이다.

v2 runner SHA: `e87a040997194d30ebd19132199b4c3a523ee693d162ef88329b611a540ffcae`.
v2 config SHA: `80d5086ea8709d25255b7c148e923fd725e23b8ac2f9c33d3ed289a4cadc3af5`.
v2 manifest SHA: `998e5e8b4e78076457f52e2683704a656bc6a53291fd6647d8dcc92695ff556c`.

고정 비교 참조는 clean CSV SHA `6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7`와 과거 local RMSE 0.7791048399763751 m이다. 둘은 QA용 스칼라만 사용한다. 과거 CSV/OOF를 열거나 그 값으로 새 학습·계수를 조정하지 않는다. SHA가 다르면 같은 공식 채점 후보라는 주장을 하지 않는다.

완료한 cold run_b와 recovered run_a는 동일 키·행 순서·training target을 확인한 뒤 OOF, full-model historical probe, 두 **새** 답안을 비교했다. 1,086행 OOF RMSE는 둘 다 **0.7791048399763751 m**, 독립 SSE는 **659.2067259186298 m²**이고 예측 차이는 0이다. 같은 historical probe에서 full single과 GPU full multi를 따로 예측한 값도 각각 exact였다. 1,200행 새 답안 역시 exact였다. 허용치는 0 그대로이며 추가 재학습으로 일치를 찾지 않았다.

8개 CBM의 binary SHA는 두 실행에서 다르고 router joblib SHA는 같다. 그러나 동일 입력의 single/multi 수치 예측은 모두 exact였다. 따라서 모델 파일 hash 차이를 곧바로 예측 비결정성으로 해석하지 않는다. full single의 중단 전 save-time digest 부재는 동일 파라미터·same-input native 예측 비교로 보완했지만, acceptance-time hash라는 원래 한계는 지우지 않았다. 상세 산식·hash 비교는 [repetition-comparison.json](repetition-comparison.json)에 있다.

## 0.2의 출처와 한계

`artifacts/p3/long_persistence_shrink/metrics.json`의 기록 시각은 **2026-08-17T15:54:42.599176+09:00**이다. line 16의 0.2 및 line 17의 `single bounded scalar chosen after local diagnostics; not a virgin holdout result`가 원래 선택 성격을 명시한다. 해당 집계 JSON SHA는 `41ed5676fc993fc6debefd271f74f28be69340bb4b7a5621ab5d83322a81c46e`이다.

`scripts/run_p3_long_persistence_shrink.py` lines 67/74는 local training OOF 입력, lines 97–104는 target_hs/기존 local prediction/persistence를 이용한 제한된 0.15/0.20/0.25 진단이며, lines 113/135–137은 local adaptive 선택 및 external/hidden/official-score 비사용을 기록한다. source SHA는 `d907eab7a3f1ad8191703a12045f3ae52313893171837c4d803c6f530aa2342c`이다. 이 과거 OOF 자체는 현재 읽지 않았다.

`src/p3_wave/persistence_shrink.py` lines 11/18/54의 고정 상수와 convex 식을 복사했으며 SHA는 `f6afbb6f3f7cf6ee3f7f39b54dbb0d47d943ee8804921de5787af34253b64914`이다. 이는 Public-score 역산 alpha/axis와 다른 **local adaptive 후 고정** 계보이다. 원래 선택을 virgin holdout 또는 진단 이전 사전등록으로 과장하지 않는다. 자세한 출처는 패키지 `06_docs/shrink-provenance.json`에 동봉했다.

## 보존한 기술 실패와 정정

첫 v1 패키지는 24,360 anchor 원자료 특징 준비 후, 잘못 추가한 `previous OOF target ready < current 48h context start` 단언 때문에 종료했다. PID 38332, failure UTC 2026-09-05T17:41:42.326228+00:00, model 파일 0, TRAIN_LOCK 없음, 모든 model/router fit 0이었다. `artifacts/portable_cleanroom_20260906_v1/P3/run_a/FAILURE_RECEIPT.json` 등은 그대로 보존한다.

원 clean v4 QA는 이전 OOF의 마지막 target 시간이 **fold-start fit 시점보다 이전**인지 검사한다. context-start보다 앞서야 한다는 조건은 원 계약에 없었다. root 승인에 따라 새 ID/새 빈 경로의 v2에서 이 adapter 단언만 원래 조건으로 정정했다. 모델·행·분할·seed·계수는 바꾸지 않았고 v1 특징 cache도 재사용하지 않았다. v2에는 fold-start 이전 target 허용 / fold-start 이후 target 차단 경계 테스트를 추가했다.

## 검증 결과와 보존된 운영 오류

- v2 focused synthetic pytest **11 PASS**, Ruff PASS. 이 중 작고 합성인 CPU CatBoost save/load 검증만 포함하며 실제 학습 성공을 대체하지 않는다.
- 두 실행 비교기 focused synthetic pytest **5 PASS**, Ruff PASS. 중복·순서·target 불일치 및 nonfinite를 차단한다.
- v1 synthetic 9 PASS였지만 실제 시간 경계 실패를 놓쳤다. 이전 정적 리뷰와 합성 PASS를 실행 PASS로 인용하지 않는다.
- v2 run_a: source prepare **344.889873초**, historical 6 fits 및 earlier-OOF router 2회, full single 1 fit를 마친 후 full multi에서 중단. terminal/failure receipt 및 프로세스는 없으며 종료 원인은 단정하지 않는다. 실행기 중단 요청과 시간적으로 이어지지만 exit receipt가 없으므로 사용자 중단을 확정 원인으로도 기록하지 않는다.
- v2 run_b: source prepare **334.800947초**, historical 6 fits **613.287494초**, prepare+training **1199.567407초**, source→answer wall **1256.162552초**. 실제 PID 21764 → model replay 27744 → inference 25640 → answer replay 41432. training-only QA 36/36, full QA 42/42 PASS.
- recovery v3: 새로운 21개 input SHA, 원래 6개 historical 모델의 save-time hash, own OOF/prepare/source/config 연결, native CBM 7개 tree/feature schema 검사 PASS. full single은 수락 시점 digest만 있으며 중단 전 save-time digest는 없었다. full-router material 1,086행을 기존 own OOF에서 생성했으며 preflight fit 0. 14 synthetic PASS/Ruff PASS. 실제 새 full multi+router 학습 **24.554942초**; 원 prepare부터 중단/대기 포함 source→answer wall **4526.626833초**는 새 cold-start 속도가 아니다. PID 19536 → model replay 37016 → inference 38788 → answer replay 26160. training-only QA 38/38, full QA 44/44 PASS.
- recovery 비교기: 5 synthetic PASS/Ruff PASS. 중단 attempt를 0으로 숨기거나 recovery를 empty-model 성공으로 표시하면 차단한다.
- terminal 후 independent math.fsum SSE/RMSE, 1,086 historical target의 배포 원자료 일치, 78h split/episode 분리, prior-OOF availability, own hash/PID, official schema/order/finite/range, fresh CSV replay를 확인했다.
- recovery의 첫 추론은 누락된 `05_answer` 디렉터리 검사에서 FileNotFoundError로 종료했다. INFERENCE_LOCK 생성·official_frame·CSV 이전이었으며 fit 0, official 0이다. 원 failure receipt를 보존한 채 승인된 빈 디렉터리만 생성하고, 같은 모델/봉인 코드로 아직 미소비였던 추론을 한 번 실행했다. [inference-directory-repair.json](inference-directory-repair.json)에 전후 모델·코드 불변 근거가 있다. 패키지 builder의 6개 디렉터리 존재 합성검사로 재발을 점검했다.
- 첫 ZIP 안내의 기존 `--replay` 재사용 문구는 이미 보존된 exclusive-write receipt와 충돌한다는 정적 리뷰를 받아 폐기 안내로 남겼다. 원 ZIP은 이력으로 보존하고 **새 archives_v3만 최신 사용본**이다. 저장 ZIP의 실제 추출/새 PID 무학습 추론은 3.668310초, 1,200행 exact. archive synthetic **5 PASS**, Ruff PASS.
- 최종 집계 [independent-qa.json](independent-qa.json): **134/134 PASS**, 실패 0. 이는 기존 cold 42/recovery 44 실행 QA를 새로 대신한 것이 아니라 그 receipt/model/ZIP/hash 연결과 실제 추출 결과를 감사한 검사 수다. focused synthetic 합계 40 PASS(11+5+14+5+5), 실제 backbone fit와 별도 집계다. 추가 개선 실험은 실행하지 않았다.

## 검증 범위 제한

현재 실행은 이동된 코드 + 현재 Python 3.12.10 환경 + 현재 Windows/GPU의 검증이다. Python `-I`와 로컬 import 경계를 확인하지만 별도 새 venv 설치나 심사 하드웨어 재현까지 증명하지 않는다. pinned runtime dependencies와 사용법은 동봉했으나 offline wheel bundle은 제공하지 않았다.

Python-level network audit hook과 파일 경계 검사는 OS 수준 완전 인터넷 차단 또는 과거 시점 전 시스템 감사의 증명이 아니다. 6시간은 각 단계 측정 및 checkpoint에서 검사하며 native GPU fit을 강제 종료하는 OS hard-timeout 보장은 아니다. CatBoost는 GPU의 부동소수점 합산 순서 때문에 학습이 비결정적이라고 명시한다. 따라서 두 새 실행을 직접 비교하며, 두 번 일치해도 모든 하드웨어·환경의 보편적 결정성을 증명하지 않는다. [CatBoost 공식 GPU 학습 문서](https://catboost.ai/docs/en/features/training-on-gpu), 2026-09-06 KST 확인. 내부 181 case와 최신 5-forward benchmark/공식 전체 분포를 혼동하지 않는다.
