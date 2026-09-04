# 최신 리더보드가 증명하는 문제별 개선여지

## 기술 요약

결론은 **사용자의 판단이 맞다**. 2026-08-29 01:55:20 KST 공식 Public 리더보드에서 분당독고다이는 81.047226점으로 4위이고, 1위는 84.722067점이다. 3.674841점 차이는 실제 Public 데이터에서 더 높은 해법이 존재한다는 직접 증거다. 더욱 강한 증거는 8월 27일 이후 우리가 문제별 최고와의 격차를 합계 2.097452점 줄였다는 사실이다. 당시의 headroom은 단순한 점수 산술이 아니라 실제로 일부 회수 가능한 개선여지였다.

다만 리더보드가 증명하는 범위는 Public까지다. 상위팀 코드가 공개되지 않았고, P3 Public은 66 cases/396 rows뿐이며 P1도 Public 이상 이벤트 수가 적다. 따라서 상위 점수는 “구조적 개선을 시도할 가치”를 강하게 증명하지만 “같은 개선이 Private에 재현된다”거나 “상위팀 구조를 알고 있다”는 뜻은 아니다.

실행 우선순위는 **P1 구조 재설계 > P2 성층 상태 잔차모형 > P3 KMA 기반 MOS 잔차모형**이다. P1이 1위와의 총격차 중 84.25%를 설명하고, 최근 P1 미세 후처리의 공식 이득은 +0.007978점에 불과했다. P2와 P3는 기존 최고 모델을 보존한 채 하나의 새 구조축만 시험해야 한다.

## 핵심 수치

| 문제 | 우리 점수 | 문제별 최고 | 점수 여지 | 완주팀 내 우리 순위 | 우리 공식 원지표 | 최고 원지표 추정* | 필요한 원지표 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 28.909341 | 32.005398 | **3.096057** | 7/18 | F1 0.833548 | F1 0.950037 | **+0.116489 F1** |
| P2 | 27.934759 | 28.633273 | 0.698514 | 6/18 | RMSE 0.430250°C | RMSE 0.374581°C | **−0.055669°C (−12.94%)** |
| P3 | 24.203126 | 24.784043 | 0.580917 | 2/18 | RMSE 0.575262m | RMSE 0.538661m | **−0.036601m (−6.36%)** |

\* 최고 원지표는 공식 변환식이 아니다. 우리 공식 제출 이력에서 원지표→점수를 OLS로 적합해 역산한 계획용 추정치다. 관측 구간에서 최대 점수 잔차는 0.000012 이하지만, 반올림과 구간 밖 비선형 가능성이 남는다.

민감도 해석은 유용하다. 현재 관측 구간에서 대략 P1 F1 +0.01은 +0.266점, P2 RMSE −0.01°C는 +0.125점, P3 RMSE −0.01m는 +0.159점에 대응한다. 따라서 P1 3점은 단순 threshold 소수점 조정으로 닫을 규모가 아니다.

## 리더보드가 증명하는 것과 증명하지 못하는 것

증명하는 것:

- 상위 Public 성능은 실재한다. 문제별 최고점 합은 85.422714점으로 우리보다 4.375488점 높다.
- 문제별 전문화가 중요하다. P3 최고팀은 종합 7위이고, 완주 18팀에서 P2–P3 점수 상관은 0.032803으로 거의 0이다. 하나의 보편적 모델 역량보다 문제별 파이프라인 차이가 크다는 신호다.
- 격차는 회수 가능했다. 8월 27일 대비 P1/P2/P3 최고점 격차를 각각 0.416605/1.302950/0.377897점 줄였다.

증명하지 못하는 것:

- 상위팀이 어떤 모델을 썼는지는 알 수 없다. 공개 GitHub와 웹에서 검증 가능한 참가자 코드를 찾지 못했다.
- Public 최고가 Private 최고라는 보장은 없다. 반복 제출로 동일 Public holdout에 적응하면 모델 선택 기준 자체를 과적합할 수 있다.
- 문제별 최고를 한 팀이 동시에 달성한 것도 아니다. 문제별 최고점 합은 실제 단일 팀 점수가 아니라 oracle ceiling이다.

## P1: 유형별 사건 모델로 재구성

현재 병목은 전체를 한 점수/한 모델로 처리한 뒤 임계값과 GI 후처리를 미세 조정해 온 데 있다. 공식 데이터 생성 과정은 spike, noise, flatline, offset, drift의 시간 형태와 지속시간이 다르고 서로 겹칠 수 있음을 명시한다. 이런 구조에서는 범용 anomaly score 하나보다 **유형별 검출기 + 겹침 가능한 사건 결합 + 시간 디코더**가 더 직접적이다.

권고 구조:

1. spike/noise용 짧은 창 변화량·분산 head, flatline용 run-length·극저분산 head, offset/drift용 change-point·누적기울기 head를 분리한다.
2. 수온 층간·기상/해양 관련 변수의 일관성 head를 추가한다. 실제 KORS 품질관리 연구에서도 단일 gradient 검사는 군집 이상을 놓치고 thermocline 자연변동을 오탐해, 다중 이동창과 관련 변수 일관성 검사를 제안했다.
3. head 출력을 OOF prediction으로 만든 뒤 작은 logistic/GBM stacker와 지속시간 priors로 결합한다. 라벨 겹침을 허용하고 최종 평가는 공식과 같은 label=1 F1로 한다.
4. 단순 비지도 Anomaly Transformer를 그대로 쓰기보다, 풍부한 합성 라벨과 유형 정보를 감독학습에 활용한다. Anomaly Transformer의 association discrepancy는 보조 특징/비교군이지 정답 구조로 간주하지 않는다.

실험 1은 incumbent의 확률에 5개 유형 head의 OOF 확률을 추가하는 residual stacker다. 합성 이상 사건 단위로 fold를 분리하고, 동일 생성 사건이 train/validation에 걸치지 않게 한다. 실험 2는 모델을 고정한 뒤 duration-aware decoder만 붙여 raw 모델과 직교적으로 비교한다. 두 실험 모두 시간블록·연도·관측소별 sign이 일치하지 않으면 공식 제출로 승격하지 않는다.

중단 기준: 유형별 사건 recall이 올라도 전체 precision 붕괴로 F1이 세 hard fold 중 두 곳에서 악화되거나, decoder의 이득이 한 유형/한 fold에만 집중되면 중단한다.

## P2: 계절 성층 상태에 조건부인 수직 잔차 복원

현재 OAS/앙상블 incumbent는 강하다. 최신 cross-fit 보정은 공식 +0.012572점에 그쳐, 같은 미세 veto 축의 한계가 보인다. 반면 SORS 연구는 8–9월 완전 성층과 10–11월 약화를 관측했고, 하층은 표층과 다르게 안정적이었다. 결측 기간이 바로 9–10월이므로 계절 성층 regime을 명시하지 않는 보간은 핵심 상태를 평균낼 위험이 있다.

권고 구조:

1. 현재 OAS 예측을 base로 고정한다.
2. 공개층 1/5/6/7/8의 수직 온도·염분 기울기, 조석 위상(12.42h), 시간/계절, 기상 변수를 사용해 mixed / weakening-stratified 상태를 저차원 gate로 추정한다.
3. 2024 동계절 완전 프로필과 2025 결측 외 구간에서 층별 OAS 잔차를 학습한다. 출력은 절대 수온이 아니라 layer 2/3/4 residual이며, 수직 smoothness와 층 순서 제약을 둔다.
4. gate의 확신이 낮으면 residual을 0으로 shrink해 incumbent로 되돌린다.

실험 1은 두 regime만 가진 저차원 residual ridge/GBM이다. 실험 2는 같은 OOF prediction에 수직 저랭크 basis를 추가한 모델이다. 일반 CMFPCA를 다시 크게 돌리는 대신, 이미 강한 base의 잔차와 9–10월 성층 전이에 초점을 좁힌다.

중단 기준: 2024 가상 9–10월 mask와 2025 pre/post-gap pseudo-mask에서 모두 개선하지 못하거나, layer별 RMSE 중 하나가 크게 악화되면 중단한다.

## P3: 스칼라 KMA 다음은 풍파 상태별 MOS 잔차

우리 P3는 18개 완주팀 중 2위다. KMA 40% 결합은 공식 +0.136958점으로 유효했지만, α=0/0.2/0.4의 Public 세 점으로 복원한 MSE 곡선 정점은 α≈0.424865이고 추가 이득은 RMSE 0.0000299m뿐이다. 스칼라 α 탐색은 종료할 근거가 충분하다.

권고 구조:

1. α=0.4 incumbent를 base로 고정하고 `y - base` 잔차만 예측한다.
2. past-only 48h의 파고·파주기·파향, 풍속·풍향, 기압 변화, lead, 계절/폭풍 regime을 입력한다. 방향은 sin/cos로 표현한다.
3. 표본이 작으므로 lead별 완전 분리보다 shared trunk + lead embedding 또는 저차원 N-HiTS/MLP/GBM residual을 우선한다. N-HiTS는 다중 속도 표본과 계층 보간으로 서로 다른 주파수 성분을 분해한다.
4. case/time-block purged OOF와 moving-block bootstrap으로 residual overlay의 분산을 확인한다.

실험 1은 shrinkage가 강한 tree/ridge MOS residual이며 가장 값싼 구조 검증이다. 실험 2는 동일 입력/분할의 소형 N-HiTS residual로 비선형 이득만 분리한다. 공식 제출은 base 대비 예측 벡터 상관이 지나치게 높으면서 로컬 이득이 극미하면 정보가치가 낮으므로 보류한다.

중단 기준: lead 3개 이상에서 개선하지 않거나, 폭풍 case bootstrap의 하위 25%에서 tail RMSE가 악화되거나, Public 한 번의 양성 결과가 로컬 sign과 반대면 구조 승격 대신 “Public 적응 가능성”으로만 기록한다.

## 승격 기준

기존의 “공식 점수 +3점”을 단일 로컬 gate로 쓰지 않는다. 3점은 P1에서 F1 약 +0.113에 해당하지만 P2/P3의 유효 구조 개선은 그보다 작을 수 있다. 새 기준은 효과 크기, 반복성, 정보가치를 분리한다.

1. **누수 방지:** P1은 합성 사건 단위 group split, P2는 연도·계절 contiguous mask, P3는 anchor case/time purged split을 고정한다.
2. **반복성:** 모든 사전등록 hard window의 과반에서 sign이 양수이고, aggregate 공식 metric도 개선되어야 한다.
3. **불확실성:** P1은 사건 block bootstrap F1 차이, P2/P3는 case/time block bootstrap paired RMSE 차이를 기록한다. 95% CI 전체 양수를 절대요건으로 삼기보다, 중앙값·worst quartile·tail 손실을 함께 본다.
4. **효과와 정보 분리:** 성능 후보는 안정적 양성 효과, 정보 후보는 한 구조 가설만 바꾸고 결과 해석이 가능한 직교 대조를 요구한다.
5. **Public 사전등록:** 제출 전 예상 방향, 실패 시 결론, 재시도 금지 조건을 적는다. Public 결과로 hyperparameter를 연속 보간하는 횟수를 제한한다.
6. **Private 보호:** Public 점수를 학습 target으로 사용하지 않고, 강한 incumbent는 언제나 보존한다. 새 구조가 실패해도 제출본과 챔피언을 덮어쓰지 않는다.

## 우선순위와 자원 배분

| 우선 | 문제 | 판단 | 다음 구조 실험 | 연구 자원 권고 |
|---:|---|---|---|---:|
| 1 | P1 | 1위 격차의 84.25%; 미세 후처리로 설명 불가 | 유형별 사건 head + OOF stacker | 60% |
| 2 | P2 | 잔여 0.0557°C; 성층 전이의 물리적 근거 | regime-gated OAS residual | 25% |
| 3 | P3 | 현재 2위; scalar α 포화 | wind-wave lead-aware MOS residual | 15% |

예상 점수를 임의로 약속하지 않는다. 대신 구조 실험의 planning band를 점수 변환 민감도로 표현한다. P1 F1 +0.02/+0.05/+0.10은 약 +0.53/+1.33/+2.66점, P2 RMSE −0.01/−0.03/−0.05°C는 약 +0.13/+0.38/+0.63점, P3 RMSE −0.01/−0.02/−0.03m는 약 +0.16/+0.32/+0.48점이다. 이는 가능성 예측이 아니라, 로컬 효과를 공식 점수 크기로 읽기 위한 변환표다.

## 한계와 반대 증거

- 리더보드는 Public 최고점의 합이다. 반복 제출 수와 후보 생성 과정이 팀마다 달라 동일한 과학적 비교가 아니다.
- P3 Public은 66 cases이고 P1 Public 이벤트 수도 제한적이다. 상위권 수십 분의 일 점은 sampling noise 또는 Public 적응일 수 있다.
- 문제별 최고 원지표는 공식 식이 아닌 경험적 역산이다.
- 타 참가자 구조를 검증하지 못했다. 팀명에서 Random Forest 등 모델을 추정하지 않는다.
- 외부 논문은 다른 해역·표본·평가다. 모델 후보의 원리 근거이지 대회 점수 보증이 아니다.
- 리더보드 3–7위 총점 범위가 1.229598점으로 조밀하다. 작은 Public 변동으로 순위는 쉽게 바뀔 수 있다.

## 독립 Gemini 딥리서치 교차검증

동일한 최신 점수·공개 저장소·요구사항을 Gemini Deep Research에 독립 입력해 두 번째 관점을 얻었다. 일치한 부분은 P1 최우선, P3 자원 축소, Public 반복 선택 편향 경계, P3의 선형 KMA 다음에 풍향·lead-aware residual/MOS가 필요하다는 큰 방향이다.

하지만 아래 주장은 공식 문제·우리 원장과 충돌하거나 재현 근거가 없어 **채택하지 않았다**.

- P2를 공간 격자 해수면온도 예측으로 해석하고 eddy·위경도 attention·연안 mask를 제안했다. 실제 P2는 S-ORS 수직 layer 2/3/4의 2025년 9–10월 결측복원이다.
- P1이 point-adjustment(PA) 평가라고 가정하고 minefield/무작위 spike 제출을 제안했다. 공식 평가는 `label=1` F1이며 PA 적용은 확인되지 않았다. 제출 holdout을 역공학하는 탐침은 Private 일반화와 제출 기회 가치에도 반한다.
- P3 case별 잔차 없이 표준오차 ±0.0501m와 z=0.73을 제시했다. RMSE 하나와 n=66만으로는 case-cluster RMSE 차이의 표준오차를 식별할 수 없다.
- P1 상위 F1을 약 0.96으로 제시했지만 우리 공식 제출쌍 OLS 역산은 약 0.950037이다.
- 공개 저장소 접근이 제한됐다고 했지만 저장소와 지정 commit은 실제로 공개 접근 가능하다.
- 문제별 낙관/기준 점수 범위는 검증 데이터나 학습 곡선 없이 제시돼 예측으로 채택하지 않았다.

이 독립 연구는 “공통 결론의 방향성 확인”과 “그럴듯하지만 문제 정의를 벗어난 일반론을 걸러내는 반증 장치”로 사용했다. 최종 전략의 정량 수치는 공식 리더보드·제출 원장·재현 가능한 계산만을 사용한다.

## 다음 실행 순서

1. P1 사건 단위 split과 5개 유형별 OOF confusion/segment recall 원장을 먼저 만든다.
2. incumbent prediction을 고정한 P1 residual stacker를 한 번 학습하고, 이어 decoder만 분리 평가한다.
3. 병렬로 P2 two-regime residual ridge를 값싼 구조 검증으로 실행한다.
4. P3는 α 탐색을 중단하고 가장 작은 MOS residual부터 실행한다.
5. 각 문제에서 로컬 기준을 통과한 후보만 “성능 후보”와 “정보 후보”로 구분해 공식 제출 판단표에 올린다.

## 추가 질문

- P1 생성기의 유형별 label 또는 anomaly provenance를 학습 데이터에서 직접 복원할 수 있는가?
- P2 2025 결측 전후 구간에 동일한 수직 센서 편향/보정 이력이 있는가?
- P3 KMA 예보의 issue time과 valid time 정렬이 모든 anchor에서 엄밀히 보장되는가?
- 운영본부 공지의 대학부 마감일 9월 7일과 공개 홈페이지의 9월 30일 충돌 중 어느 날짜가 유효한가?

## 주요 출처

- [공식 리더보드](https://oceanaidata.org/app/leaderboard), [P1](https://oceanaidata.org/app/problems/5), [P2](https://oceanaidata.org/app/problems/6), [P3](https://oceanaidata.org/app/problems/7), [공식 제출관리](https://oceanaidata.org/app/submissions), accessed 2026-08-29.
- Min et al., *Quality Control of Observed Temperature Time Series from the Korea Ocean Research Stations*, 2020, DOI 10.4217/OPR.2020.42.3.195.
- Lee et al., *Hydrodynamics and Sediment Transport at Socheongcho Ocean Research Station*, Water 16(1):23, 2024, [DOI](https://doi.org/10.3390/w16010023).
- Xu et al., *Anomaly Transformer*, ICLR 2022, [OpenReview](https://openreview.net/forum?id=LzQQ89U1qm_).
- Challu et al., *N-HiTS*, AAAI 2023, [DOI](https://doi.org/10.1609/aaai.v37i6.25854).
- Zhang et al., *Real-time Bias Correction of Significant Wave Height Forecasts*, Ocean Modelling 187, 2024, [DOI](https://doi.org/10.1016/j.ocemod.2023.102289).
- Cawley & Talbot, *On Over-fitting in Model Selection*, JMLR 2010, [paper](https://www.jmlr.org/papers/v11/cawley10a.html).
- Blum & Hardt, *The Ladder: A Reliable Leaderboard for Machine Learning Competitions*, ICML 2015, [PMLR](https://proceedings.mlr.press/v37/blum15.html).
