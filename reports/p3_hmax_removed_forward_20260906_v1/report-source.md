# P3 hmax 제거: 이번 두 후보 중 평균 개선폭이 커서 우선 보존

동일 categorical-lead 기준 대비 내부 pooled RMSE는 **0.684038593 → 0.682467282 m**, 차이는 **−0.001571311 m**다. 이번에 별도로 시험한 numeric lead의 −0.000500418 m보다 평균 개선폭이 크므로, 자원과 패키지 준비도를 고려할 때 완전 재학습할 단일 후보의 우선순위가 높다. 다만 **CI90 [−0.003978547, +0.000703372] m가 0을 포함**하며, S-ORS 및 바람 관측이 없는 구간은 악화했다. 확정 일반화나 공식 점수 상승이라고 주장하지 않는다.

10 backbone + 4 router 신규 fits가 **903.260초(15.05분)**에 정상 완료했다. 독립 산술·키·target·chronology·hash·재생 QA **134/134 PASS**, 별도 PID에서 10 backbone + 4 router의 103,602행 예측을 재생한 max absolute difference **0**을 확인했다. 이 단계의 공식 입력/CSV/upload/full fits는 **모두 0**이며 추가 학습은 아직 시작하지 않았다.

## 고정한 변경과 대조

기준은 [새 numeric 실험](../p3_numeric_lead_forward_gpu_20260906_v2/report-source.md)에서 실제로 다시 학습한 **범주형 lead** CPU single + GPU Plain multi + 과거 OOF router + 고정 persistence 0.2 완성정책이다. 그 실험의 numeric 후보는 여기 사용하지 않는다. 기준 result, QA 149 PASS, fresh replay, source/config, baseline OOF의 SHA를 사전 확인하고 hmax 실행 seal에 포함했다.

단일 가설은 **hmax 관련 정보집합 전체 제거**다. single/multi의 591개 특징 중 `hmax_`로 시작하는 64개를 제거하여 527개를 사용한다. raw hmax 및 hmax/hs ratio의 current/lag/window 통계/valid 파생값을 포함하며, loss router의 `hmax_current`도 제거한다. raw hmax를 값/NaN으로 교란해도 남은 특징과 router 입력이 변하지 않는지 합성 검증했다. hmax/hs 비율이 일정해 보인다는 관찰을 제거의 안전성이나 정당성 증거로 사용하지 않았다.

다른 특징, target, current_hs, sample weight, seed, epoch/tree 수, split, loss/router/shrink는 유지했다. 숫자형 lead나 lead별 배합을 추가하지 않는다. CPU single은 2 threads/700 trees, multi는 exclusive GPU0 Plain/1,200 trees다. params의 정확한 정의는 [sealed config](../../configs/experiments/p3_hmax_removed_forward_20260906_v1.json)와 [clean recipe](../../configs/experiments/p3_corrected_repeated_forward_catboost_v2.json)에 있다.

v5의 5 forward folds, 24,360 전체 anchor 후보, validation 17,267 anchors × 6 leads(3/6/9/12/18/24h) = 103,602행, 78h purge 및 raw station high-run episode exclusion을 유지한다. 최초 fold의 router는 고정 가중치이고, 이후 router는 현재 fold의 target/context footprint와 겹치지 않는 완료된 이전 fold OOF만 학습한다. full-data router는 이 historical 연구에 없다. onset/greedy는 `NOT_ENABLED`다.

이번 hmax 실행은 **10 candidate backbone + 4 candidate router만** 새로 학습했다. 앞 실행의 10 baseline backbone + 4 baseline router 및 그 OOF는 검증 후 재사용했다. 따라서 `old_prediction_reads=0`은 과거/승인되지 않은 예측 재사용이 없다는 뜻이며, 승인된 **동일 사이클 baseline 예측 103,602행** 읽기는 `authorized_same_cycle_baseline_prediction_rows`로 별도 명시한다. 이전 답안·공식 점수 역산 계수·외부 데이터·과거 모델은 입력이 아니다.

## 결과와 위험 구간

주지표는 각 행의 squared error를 모두 합해 전체 행수로 나눈 뒤 제곱근을 취한 unweighted pooled RMSE다.

| 항목 | 기준 RMSE m | 후보 RMSE m | 후보−기준 m |
|---|---:|---:|---:|
| 전체 103,602행 | 0.684038593 | 0.682467282 | −0.001571311 |
| Q2 2024 | 0.728157913 | 0.727989236 | −0.000168676 |
| Q3 2024 | 0.685556713 | 0.688178416 | +0.002621703 |
| Q4 2024 | 0.676355833 | 0.675310731 | −0.001045102 |
| Q1 2025 | 0.680023118 | 0.677331014 | −0.002692104 |
| Q2 2025 | 0.694785380 | 0.687921949 | −0.006863431 |
| G-ORS | 0.671348408 | 0.667856089 | −0.003492318 |
| I-ORS | 0.649089314 | 0.645457952 | −0.003631362 |
| S-ORS | 0.727415075 | 0.729623721 | +0.002208646 |
| 3h | 0.465212030 | 0.465435603 | +0.000223573 |
| 6h | 0.578470717 | 0.579278903 | +0.000808186 |
| 9h | 0.641900018 | 0.642615158 | +0.000715140 |
| 12h | 0.689443183 | 0.687999320 | −0.001443862 |
| 18h | 0.802276121 | 0.797986714 | −0.004289407 |
| 24h | 0.851696933 | 0.848112666 | −0.003584267 |
| 바람 관측 특징 없음 | 0.721513742 | 0.725148039 | +0.003634297 |
| 바람 관측 특징 있음 | 0.667645085 | 0.663692449 | −0.003952636 |

기준 SSE는 48,476.287099355 m², 후보 SSE는 48,253.832309487 m²다. 5개 분기 중 4개는 개선, 1개는 악화했다. 단기 3/6/9h는 소폭 악화하고 장기 12/18/24h는 개선했다. 이 관찰을 이용한 단기/장기 후보 혼합은 **이번 실험에서 하지 않는다**.

CI는 같은 951 station-episode cluster를 paired 재표집한 2,000회(seed 20260906) pooled RMSE 차이이며, 표본마다 실제 행수가 분모다. 음의 delta 표본 비율 **0.8605**는 기술적 요약이지 공식 점수 개선 확률이나 0.8 hard gate가 아니다. 평균 개선 후보를 보존하되 위 악화 구간은 별도 위험표로 유지한다.

raw high-run cluster는 독립 폭풍과 같지 않고, dense 인접 anchor와 정점 간 의존성이 남는다. 사전 고정된 두 후보를 같은 historical 평가면에서 비교한 연구이므로 이 중 우선 후보를 고른 뒤의 효과크기는 선택 낙관성을 가질 수 있다. 이번 block CI는 두 후보 선택 전체의 다중비교 보정이나 virgin holdout 증명이 아니다. GPU multi는 각 arm에서 별도 학습했으므로 GPU 학습 변동도 단일 실행 차이와 완전히 분리하지 않았다. [CatBoost 공식 GPU 학습 문서](https://catboost.ai/docs/en/features/training-on-gpu)

공식 예상 점수는 **미산정**이다. 배포 정답이나 공식 점수에서 RMSE/계수를 역산하지 않았고, 이 dense validation과 공식 익명 200-case의 분포 대응도 확보하지 않았다. 내부 long-lead 개선을 공식 최고점 갱신으로 미리 치환하지 않는다.

## 시간·재현·다음 실행

첫 예정 single 122.944초 + GPU multi 20.625초로 결과 열람 전 전체 시간을 1,572.591초(26.21분)로 예측하여 60분 cap 안에서 진행했다. 이 두 fits는 실제 후보 학습에 재사용했다. 최종 신규 single5 합계 784.797초, multi5 합계 108.666초, router/집계 포함 실행 903.260초다. preflight 3.937초와 saved replay 9.201초는 구분한다. GPU 시간 한도는 fit 전후 검사와 외부 감시이며, OS hard-kill 한도라고 주장하지 않는다.

| 항목 | 판정 |
|---|---|
| 새 hmax 후보 historical 학습 | 완료, 10 backbone + 4 router |
| 기준 재사용 | 같은 사이클 categorical baseline exact-hash, 추가 baseline fit 0 |
| 독립 산술·키·분할·해시 QA | 134/134 PASS |
| 같은 환경 저장 모델 새 PID 재생 | PASS, PID 29584→39476, 103,602행 maxdiff 0 |
| 이 후보 full2 + router1 | 아직 0 fits |
| 이 후보 whole-cold 12 backbone + 5 router | PENDING, 아직 미실행 |
| 공식 입력·답안 CSV·업로드 | 0 / 0 / 0 |

Parent 지시에 따라 warm full 학습은 보류했다. 별도 담당자가 준비하는 원 repo 밖의 source-only whole-cold 패키지가 준비되면 가장 유망한 단일 후보를 명시적으로 고정하고, 그 안의 historical10+router4→full2+router1→fresh QA→local CSV→새 PID answer replay를 우선 실행하여 중복 full fit을 줄인다. 이 보고서 작성 시점에는 추가 cold 학습을 하지 않았다. 기존 clean baseline의 cold-start PASS는 이 신규 후보의 PASS가 아니다. 기존 baseline 패키지·sealed 원실험·실패 이력은 보존한다.

## 해시·출처

- [result.json](result.json): `d575a1214d31af5220679d48dd90ea67fe54babdca00d2956f3724f29b74d1aa`
- [independent-qa.json](independent-qa.json): `26bfc5076e9a05c80dd170bac5830569e1f537d678dd30716a3a98fdb6505c10`
- [fresh-process-replay.json](fresh-process-replay.json): `846490e9ba6c657a810a7ea8c163403a8c0f632d3f016efea283b12dc40dc2ac`
- runner SHA: `3011ad945208196d9cca1468f39d29c9746bf26e2d892810fb4ab7efedfb63ad`; config SHA: `c56d5c639b08664cbccdaacbf1a1f5fb60d8de9df88b7bf4d93a2bd837e5a366`.
- baseline OOF SHA: `a9dab0d59d32e8ab2dbf27c3207bd7d945c563b9d944ce1313cee10452e782aa`. Baseline result/149-QA/replay/source 연결은 [preflight.json](preflight.json)의 `baseline_pins`에 있다.
- [사전등록](preregistration.md), [resource-pilot.json](resource-pilot.json), [focused test 기록](code-qa.json).

고정 0.2는 원래 제한된 train-only local adaptive 진단 후 고정한 상수다. 원 선택 시각/집계 SHA/선택 코드의 계보는 [기존 출처 감사](../portable_cleanroom_20260906_v1/P3/report-source.md)에 있으며, Public-score inverse alpha/axis와 다르다. 과거 OOF 자체는 hmax 신규 학습에 읽지 않았다.

불변 result의 saved replay `PENDING` 필드는 당시 상태이며, 이후 PASS는 별도 재생·QA 영수증으로 연결한다. raw data/모델/OOF/locks/logs와 작은 집계 보고서는 구분하고 본 작업에서 commit/push/upload는 하지 않았다.
