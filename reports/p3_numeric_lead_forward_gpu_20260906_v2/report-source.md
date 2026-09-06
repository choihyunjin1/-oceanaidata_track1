# P3 numeric lead: 작은 평균 개선 후보를 보존, 확정 개선은 아님

새 동일 자원 기준 정책 대비 내부 pooled RMSE는 **0.684038593 → 0.683538174 m**, 차이는 **−0.000500418 m**다. 사전 고정한 평균 개선 기준으로 후보를 보존한다. 그러나 paired station-episode bootstrap CI90은 **[−0.001879861, +0.000809684] m**로 0을 포함하고, 일부 시기·정점·장기 lead·바람 결측 구간이 악화했다. 자동 승격이나 공식 점수 상승 확정 근거가 아니다.

학습은 15 backbone + 8 small router fits, 1,955.608초(32.59분)에 정상 완료했다. 새 PID에서 저장한 15 backbone + 8 router의 103,602행 예측을 다시 계산하여 max absolute difference **0**을 확인했고, 독립 산술·해시·분할·누출·재생 검사 **149/149 PASS**다. 공식 입력·제출 CSV·upload는 이 실험에서 모두 0이다.

## 비교 계약

- 기준: 이번 실행에서 새로 학습한 categorical-lead CPU single + GPU Plain multi + 과거 OOF loss router + 고정 장기 persistence 0.2 완성정책.
- 후보: single의 `lead_h`만 문자열 범주형에서 실제 시간값 float로 바꾼다. 같은 실행에서 학습한 multi 예측은 exact 재사용하고, router는 각 arm의 OOF로 따로 재적합한다. 별도 비율·threshold·lead별 선택은 없다.
- CPU single은 2 threads, 700 trees/depth 6/lr 0.035/l2 8/random strength 0.2. GPU0 multi는 1,200 trees/depth 7/lr 0.03/l2 10/random strength 0.15/Plain. full-data 학습은 본 historical runner의 범위가 아니다.
- [v5 평가 계약](../../configs/evaluation/ocean_forward_v5.json): Q1 2024 warmup 이후 정확 5 forward folds, 24,360 training anchor 후보 중 validation 17,267 anchors × 6 leads(3/6/9/12/18/24h) = 103,602행. 591개 past-only feature, 78h purge 및 raw 20분 station high-run episode exclusion.
- router는 완료된 이전 fold OOF만 사용하고, 현재 fold의 78h/episode 조건으로 다시 제외한다. 실제 target-ready 시간과 현재 48h context footprint의 분리를 독립 검증했다. Q2 2024 첫 평가 fold에는 이전 OOF router가 없으므로 고정 가중치를 사용한다.
- 원 source CSV SHA 및 기존 fresh clean prepare cache의 source/config/hash/key 계보를 확인했다. current/6 targets 전체와 45개 raw context의 feature를 배포 데이터로 재계산했다. 이 cache 재사용은 이전 모델·예측·답안 재사용이 아니다.
- 과거 CPU-only 2-fit resource stop이나 기존 3-window/181-case 공식 관련 점수를 이 새 5-fold baseline으로 승계하지 않는다. 숫자형 변경과 별도로 수행하는 hmax 제거 후보는 이 categorical baseline만 재사용하며 숫자형 변경과 결합하지 않는다.

## 결과와 안정성

주지표는 전체 행의 unweighted SSE를 전체 행수로 나눈 뒤 제곱근을 취한 pooled RMSE다. fold/lead별 RMSE의 단순 평균이 아니다.

| 항목 | 기준 RMSE m | 후보 RMSE m | 후보−기준 m |
|---|---:|---:|---:|
| 전체 103,602행 | 0.684038593 | 0.683538174 | −0.000500418 |
| Q2 2024 | 0.728157913 | 0.726656169 | −0.001501743 |
| Q3 2024 | 0.685556713 | 0.686276067 | +0.000719354 |
| Q4 2024 | 0.676355833 | 0.675346054 | −0.001009779 |
| Q1 2025 | 0.680023118 | 0.678495092 | −0.001528027 |
| Q2 2025 | 0.694785380 | 0.699191993 | +0.004406613 |
| G-ORS | 0.671348408 | 0.669719861 | −0.001628547 |
| I-ORS | 0.649089314 | 0.650413792 | +0.001324478 |
| S-ORS | 0.727415075 | 0.726849912 | −0.000565164 |
| 3h | 0.465212030 | 0.464746263 | −0.000465766 |
| 6h | 0.578470717 | 0.577100176 | −0.001370540 |
| 9h | 0.641900018 | 0.639751976 | −0.002148042 |
| 12h | 0.689443183 | 0.689053439 | −0.000389744 |
| 18h | 0.802276121 | 0.802500406 | +0.000224285 |
| 24h | 0.851696933 | 0.852190576 | +0.000493644 |
| 바람 관측 특징 없음 | 0.721513742 | 0.722369830 | +0.000856089 |
| 바람 관측 특징 있음 | 0.667645085 | 0.666526589 | −0.001118496 |

기준 SSE 48,476.287099355 m², 후보 SSE 48,405.385991647 m²다. 재표집은 같은 951개 station-episode cluster를 양 arm에 paired 적용하고 각 표본의 실제 행수를 분모로 사용한 2,000회(seed 20260906) 계산이다. 음의 delta 표본 비율 0.752는 기술적 재표집 요약이지 공식 개선 확률이나 0.8 승격 gate가 아니다. raw high-run cluster가 독립 기상 폭풍이라는 보장은 없고, dense 인접 anchor 및 정점 간 의존성이 남는다. onset/greedy 진단은 계약상 `NOT_ENABLED`다.

공식 예상 점수는 **미산정**이다. 이 dense training-derived 평가와 공식 익명 200-case 분포의 대응을 확보하지 않았으며, 공개 점수를 역산하거나 후보를 그 점수에 맞추지 않았다.

## 자원·재현·제출 준비도

첫 예정 single 139.413초 + multi 20.723초를 실제 본 실험에 재사용했다. 성능을 열기 전 규모 선형×1.2 margin으로 3,281.112초(54.69분)를 예측하여 90분 한도 안에서 진행했고, 실측은 32.59분이다. preflight 3.870초, saved replay 15.787초는 학습 시간과 구분한다. synthetic compatibility 검사 3 tiny fits는 15 historical fits와 별도다.

GPU 학습 자체는 부동소수점 합산 순서 때문에 비결정적일 수 있다. 같은 저장 모델의 exact replay를 전체 재학습 동일성으로 해석하지 않는다. GPU fit은 callback이 아닌 fit 전후 시간 검사와 외부 프로세스 감시를 사용하므로 OS hard timeout이라고 주장하지 않는다. [CatBoost 공식 GPU 학습 문서](https://catboost.ai/docs/en/features/training-on-gpu)

| 준비 항목 | 판정 |
|---|---|
| 새 5-fold 기준·후보 학습 | 완료, 15 backbone + 8 router |
| 내부 target/key/chronology/산술 QA | 149/149 PASS |
| 같은 환경 저장 모델 새 PID 재생 | PASS, PID 8552→40848, 103,602행 maxdiff 0 |
| 승인된 후보 full2 + full router1 | PENDING, hmax 독립 비교 후 별도 출력 경로에서 수행 |
| 이 후보의 빈 폴더 whole-cold 12 backbone 재학습 | PENDING, 아직 실행하지 않음 |
| 공식 입력·답안 CSV·업로드 | 0 / 0 / 0 |

기존 clean portable baseline의 cold-start PASS는 이 신규 numeric 후보의 PASS가 아니다. 같은 사이클 OOF를 사용하는 full 모델 생성과 전체 source→OOF→full 재생성도 구분한다. 후보를 공식적으로 평가하더라도 안정성 악화를 별도로 유지한다.

운영 후속: hmax QA 완료 시 새 whole-cold 후보 패키지가 준비되어 있으면, 중복 warm full2+router1을 생략하고 선정된 단일 후보의 전체 12 backbone+5 router 경로 안에서 full/QA/CSV/replay를 수행하는 방향을 우선 검토한다. 현재는 어느 후보의 추가 full/cold 학습도 시작하지 않았으며, parent가 자원과 패키지 준비도를 확인한 뒤 결정한다.

## 계보와 검증 파일

모델 상수는 [clean recipe](../../configs/experiments/p3_corrected_repeated_forward_catboost_v2.json), 이번 실행의 정확 변경·자원은 [sealed config](../../configs/experiments/p3_numeric_lead_forward_gpu_20260906_v2.json)에 있다. 장기 persistence 0.2는 2026-08-17의 제한된 **local adaptive 진단 후 고정** 상수이며 virgin holdout 사전 선택이나 Public-score inverse alpha/axis가 아니다. 당시 집계 SHA `41ed5676fc993fc6debefd271f74f28be69340bb4b7a5621ab5d83322a81c46e`와 원 선택 코드 SHA `d907eab7a3f1ad8191703a12045f3ae52313893171837c4d803c6f530aa2342c`의 근거는 [기존 출처 감사](../portable_cleanroom_20260906_v1/P3/report-source.md#02의-출처와-한계)에 기록되어 있다. 과거 OOF는 이번 학습 입력으로 읽지 않았다.

- [result.json](result.json): `66b65d55c09f7eae557a4055a9b7cbe0ba2401c579a5b58f5d56144adbf1d8f3`
- [independent-qa.json](independent-qa.json): `dd64d4434e8c437e367b11c2ea82c45fec5dad5f65046197d463ebf4bbc3863e`
- [fresh-process-replay.json](fresh-process-replay.json): `9981e18dd79fa8edf6096e3d88521acd5cfbf7beb593d1de394eb4df46ba22e2`
- runner SHA: `d9c3b5d6e628d768f66725ea091f91175e1949eb5e38bf344da81aedb5568563`; config SHA: `0800f97d1be20d297b763e05a233b6884ab2aa71421403d60d7afc036c8490dc`.
- [preflight.json](preflight.json), [resource-pilot.json](resource-pilot.json), [사전등록](preregistration.md), [focused test 기록](code-qa.json).

Runner/result의 `same_environment_saved_model_replay=PENDING`는 완료 당시 불변 기록이며, 그 이후의 실제 PASS는 별도 fresh-process-replay/independent-QA 영수증으로 연결한다. 원 sealed 결과를 덮어쓰지 않는다. 학습 raw data, 모델, OOF 예측, lock/log는 연구용 ignored artifact이며 출판할 집계 보고서와 구분한다. 본 작업에서 commit/push/upload는 하지 않았다.
