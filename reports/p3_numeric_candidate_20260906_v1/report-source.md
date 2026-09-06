# P3 numeric-lead 후보 — 최종 학습·내부 QA·답안 재생 완료

**1,200행 새 로컬 제출 후보가 완성됐으며 별도 PID의 CSV 전체 SHA 재생이 정확히 일치했다.** 이 작업은 hmax 제거가 아닌 **numeric-lead 단일 변경** 후보다. 기존 clean 기준 파일은 보존했고 이 lane은 업로드하지 않았다. 공식 점수는 root의 별도 제출/채점 receipt로 확인해야 한다.

- 후보: `artifacts/p3_numeric_candidate_20260906_v1/05_answer/submission.csv`
- SHA-256: `ff42a6a08c76f0d58ed2f3a9ea31a08819ada5fa6e9007af942fe0b891937960`
- 형식: `case_id,station,lead_h,hs_pred`, 200 cases × 6 leads, 1,200행, UTF-8/LF, 공식 index 키/순서 동일, 중복 0, finite 0..30m.
- focused pytest **28 PASS**, Ruff PASS, full native/model/provenance QA **24/24 PASS**, [root 별도 산술·답안 QA](../parallel_core_training_20260906_v1/p3-root-independent-qa.json) **28/28 PASS**.

## 실제 수행 및 분리 계상

| 단계 | 실제 범위 | 결과 |
|---|---|---|
| 기존 numeric 역사 검증 | 15 backbone + 8 prior-OOF router, 103,602행 | 149 QA/새 PID 재생의 exact SHA 확인; **재실행 0** |
| 새 full single | CPU2, 700 trees, seed 20260817, 146,160행 | 223.940초 |
| 새 full multi | GPU0 Plain, 1,200 trees, 같은 seed, 24,360 anchors | 25.043초 |
| 새 full router | 검증된 동일 실험 OOF 103,602행, alpha10/temperature2/strength0.5 | 1 fit |
| full 학습 전체 | 새 **2 backbone + 1 router**, 빈 full 모델 폴더 | **254.056초**, PID4544 |
| 내부 저장 모델 재생 | train-derived 128 cases × 6 leads | PID14304, 768 predictions, max absolute difference **0** |
| 공식 입력 추론 | training QA 이후 허용 공개 context/index | PID10780, 3.763초 |
| 최종 CSV 재생 | 새 PID, 동일 모델과 동일 입력 | PID37156, 3.753초, **전체 bytes SHA exact** |

학습은 13:04:20 KST경 시작해 13:08:34경 종료했고, 마지막 답안 replay receipt는 13:09:57 KST에 생성됐다. seal의 `started_utc`→마지막 receipt 파일 UTC 기준 운영 대기/QA 포함 **337.491초(약5분37초)**다. 이 숫자는 source feature prepare와 과거 23 fits를 새로 수행한 시간이 아니다. 3,600초 watchdog은 신규 fit 프로세스에만 적용됐다. GPU는 실제 학습 프로세스 종료 확인 후 root/P2에 해제 통보했고 이후 추론은 CPU였다.

## 유지한 정책과 내부 근거

새 single은 실제 `lead_h` float를 사용하며 native category index는 `[0]`(station만)이다. multi의 station category와 **591개 특징 및 hmax 원형/파생 특징은 그대로** 유지한다. router에도 `hmax_current`가 남아 있다. numeric+hmax 제거 조합, 새 seed 탐색, threshold/weight retuning은 하지 않았다.

기존 [numeric 검증 보고서](../p3_numeric_lead_forward_gpu_20260906_v2/report-source.md)의 pooled RMSE는 **0.6840385926 → 0.6835381743m**, Δ **−0.000500418m**다. 90% paired station/high-run cluster 구간은 **[−0.001879861, +0.000809684]m**로 0을 포함한다. 최악 quarter Δ는 **+0.004406613m**이며 일부 long-lead/wind-missing slice가 악화됐다. 따라서 작은 평균 개선 연구 후보이지 안정적 최고점 보장이 아니다. 이 수치는 원 103,602행의 이전-fold-only/purge 검증 결과를 인용한 것이고, final router 자신의 학습 OOF/저장 probe로 재평가해 낸 성능이 아니다.

full router를 모든 검증된 training OOF로 다시 학습하는 것은 최종 배포 단계다. 역사 평가 router의 earlier-fold-only 계약과 달리 배포용 full router는 모든 해당 OOF를 사용하지만, 이 자체로 새로운 일반화 성능을 주장하지 않는다.

12/18/24h persistence 0.2는 기존 clean recipe의 local-adaptive 선택 후 고정 계보다. [선행 provenance 감사](../portable_cleanroom_20260906_v1/P3/report-source.md)의 원 선택 시각은 2026-08-17 15:54:42 KST이며, 집계 SHA `41ed5676fc993fc6debefd271f74f28be69340bb4b7a5621ab5d83322a81c46e`, 선택 코드 SHA `d907eab7a3f1ad8191703a12045f3ae52313893171837c4d803c6f530aa2342c`에 연결된다. 이를 virgin/preregistration-before-any-diagnostic로 과장하지 않으며 Public 역산 alpha/axis와 구분한다. 그 과거 예측/답안은 이번 학습 입력으로 읽지 않았다.

이번 마무리 감사에서는 선행 설명만 인용하지 않고 [실제 원본 집계](../../artifacts/p3/long_persistence_shrink/metrics.json)와 [그 집계를 생성한 코드](../../scripts/run_p3_long_persistence_shrink.py)를 직접 읽고 위 SHA와 대조했다. 코드는 lines67/74에서 training-derived router OOF를 입력으로 받아 182 cases/1,092행을 검사하고, lines97–104에서 local `target_hs`와 prediction/persistence를 대상으로 0.15/0.20/0.25 민감도를 계산한다. 원 집계는 0.2와 `single bounded scalar chosen after local diagnostics; not a virgin holdout result`, 원 OOF SHA `21bef173731c89362a567baceb1df7fac383104f60746776ebae47889ae3598f`, external/hidden/official-score 비사용을 기록한다.

따라서 이것은 단순히 과거 문서에 등장한 숫자가 아니라 **실제 local OOF 진단 산출물에 기록된 고정 계수**다. 다만 현재 full 모델이 0.2를 학습한 것도, 이 스크립트가 0.2를 자동 argmin으로 골랐다는 증명도 아니다(집계에는 0.25 진단값도 있다). 이번 확인에서 그 과거 OOF 값은 열거나 원 실험을 재실행하지 않았으므로 과거 원자료 계보 전체의 신규 재현 증거로 확대하지 않는다. 새 numeric 사이클에서는 이미 고정된 0.2를 변경하지 않았다.

## 해시·QA 근거

| 근거 | SHA-256 |
|---|---|
| 현재 scoped driver | `12bc75266f6958f9e465f437e5c6db4d5010a885359643f6d174bd4c91f44fe3` |
| 현재 config | `5ec865055171eb46d53b76b1e7fb766b1fc968180520836d82ebededd6469384` |
| 불변 full materializer | `de74d271e45a7dcc520ee553ca1bc86c915ccc30d7c3f3250a77cb8b35fcd5ea` |
| 역사 result | `66b65d55c09f7eae557a4055a9b7cbe0ba2401c579a5b58f5d56144adbf1d8f3` |
| full training result | `2311332a7321a11b4183840d0c23cc1742c670ab31c1e74477e1e99b241fa5ec` |
| numeric independent QA | `5dff664363967be9815d043844f63b9781badb84999f91c39b631d04480a086c` |
| 새 PID 내부 모델 replay | `61c450ecd5fcfdc6324d45693ce3e439fca9fba4889de7cdf6131b54bd185483` |
| 답안 QA | `17ecdcccf24e85be8a95ed83ae774b5dd2de5d8dbbea005484ea73a29cbf6a8b` |
| 새 PID 답안 replay QA | `9392e72551241bc7040a18ae17e4115b83d722a55259a622f3226cbf91aa90eb` |
| root 별도 28-check QA | `a2c960d42b210dbca42e2881a053c15bdea88ca8bded728e4c03e3e16970f3e6` |

세 모델·공식 입력·source closure의 세부 SHA는 [result.json](result.json) 및 artifact `scoped-seal.json`/`04_logs`에 있다. 공개 context/index는 각 추론 전후 exact SHA가 동일했다. sample 값, hidden, 외부 관측/외부 가중치, 공개 점수 역산 입력, upload는 0이다. exact training-derived cache와 자신의 과거 numeric OOF 읽기는 허가된 재사용이며 “이전 예측 읽기 전부 0”으로 숨기지 않는다.

## 기술 오류 보존과 재현 한계

첫 preflight에서 과거 QA의 `checks`가 list인데 dict로 가정한 metadata 검사 오류가 있었다. 당시 full 모델 폴더/fit lock은 없고 fit/official 입력 모두 0이었다. `preflight-failure.json`과 첫 code-QA를 보존한 채, 학습 봉인 전에 실제 `check/pass` list schema와 false/malformed 회귀검사를 정정했다. 그 뒤 28 tests/Ruff 및 PID1392 preflight, 별도 PID36904 해시/closure check를 통과하고 나서만 학습했다. 실제 모델 fit 재시도는 없다.

**통과한 것은 빈 full 모델 폴더의 최종 3 fits 및 같은 환경 저장 모델 재생이다.** 모든 원자료부터 5-fold를 다시 학습하는 whole-cold, 두 번째 scratch 학습, 별도 venv/OS 물리적 차단망/심사 하드웨어 검증은 수행하지 않았다. source snapshot은 현재 repo 경로와 원 exact cache/OOF 계보를 요구하므로 최소 portable ZIP 완성으로 표시하지 않는다. GPU CatBoost 학습의 byte 결정론을 주장하거나 old CSV SHA를 맞추기 위해 재학습하지 않았다.

실행 방법과 디렉터리 역할은 [README](README.md)에 있다. root의 공식 채점 여부와 무관하게 현재 모델/설정/CSV는 동결하며 추가 학습/튜닝/업로드를 수행하지 않는다.
