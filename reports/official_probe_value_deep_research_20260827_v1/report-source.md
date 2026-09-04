# 로컬 점수에 종속되지 않는 공식 테스트 승격 기준

작성일: 2026-08-27 KST  
대상: 분당독고다이 P1·P2·P3 연구·제출 의사결정  
상태: `RESEARCH_COMPLETE_NO_UPLOAD`  
결정 질문: 로컬 지표가 약하거나 나빠도 최종 성능급 후보를 공식 Public에서 검증할 가치가 있는가?

## 직접 결론

**그렇다. 기존 로컬 통과/실패 gate를 공식 제출 여부에 그대로 적용한 것은 지나치게 보수적이었다.**

우리의 14개 local–official contrast에서 부호 일치는 6개뿐이었고, 비교 가능성이 상대적으로 높은 5개에서도 3개뿐이었다. P1 Router의 공식 효과는 로컬 효과의 10.84배, P2 L4는 12.76배였으며, P3 reverse 계열은 관측된 세 contrast 모두 로컬과 공식 방향이 반대였다. 이 정도면 로컬 점수는 후보의 결함과 위험을 찾는 데는 유용하지만, 공식 성능의 단독 탈락 기준으로는 부적합하다.

다만 반대 극단도 틀리다. 같은 Public leaderboard를 반복 사용하면 그 표면에도 적응적으로 과적합한다. 유한한 CV 기준을 반복 최적화할 때 생기는 선택 편향은 실제 알고리즘 간 차이와 비슷한 규모가 될 수 있고, 반복 leaderboard 질의는 숨은 holdout과 후보를 종속시킨다([Cawley & Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html), [Blum & Hardt, 2015](https://proceedings.mlr.press/v37/blum15.html)).

따라서 앞으로 두 결정을 분리한다.

1. **최종 채택 gate**: fresh local confirmation, 재현성, slice 안전성, official 방향까지 요구한다.
2. **공식 탐색 제출 gate**: 로컬 우월성을 필수로 하지 않는다. 대신 최종 배포 가능한 후보인지, 공식 점수를 움직일 규모인지, 기존 제출과 다른 정보를 주는지, 점수에 따라 다음 행동이 실제로 달라지는지를 본다.

현재 후보를 이 기준으로 다시 보면, `20260827 Round E 3×3`은 좋은 **공식 정보획득 세트**이지만 전부가 `+3점` 목표에 맞는 최종 성능급 세트는 아니다. P2 U와 P3 reverse-axis 세트는 제출 가치가 있다. P1의 G/I/GI 미세 분해 3개를 모두 쓰는 것보다, 새 MS-TCN e150의 full-train 배포형을 1~2개 포함시키는 편이 `+3점` 돌파 가능성을 더 잘 산다.

## 확인된 공식 상태

2026-08-27 로그인된 공식 화면을 읽기 전용으로 확인했다.

- 팀 `분당독고다이`: 7위, 총점 `78.919104`
- 현 최고: P1 `0.817873 / 28.492736점`, P2 `0.536536 / 26.601139점`, P3 `0.599072 / 23.825229점`
- 3위: `81.005143점`, 1위: `84.128471점`
- 2026-08-27 제출관리에는 오늘자 제출이 없었다.
- P1 문제 화면은 `오늘 남은 제출 3/3`을 명시했다. P2/P3도 문제당 하루 3회라는 사용자 확인을 운영 전제로 둔다.

공식 리더보드는 문제별 Public 최고 점수의 합으로 집계된다([공식 리더보드](https://oceanaidata.org/app/leaderboard)). 문제 목록은 각 문제를 33점으로 표시한다([공식 문제 목록](https://oceanaidata.org/app/problems)).

현재 리더보드에서 관측되는 문제별 최고 팀과 우리 점수의 차이는 다음과 같다.

| 문제 | 우리 점수 | 현재 관측 최고 | 입증된 headroom |
|---|---:|---:|---:|
| P1 | 28.492736 | 32.005398 | **+3.512662** |
| P2 | 26.601139 | 28.602603 | +2.001464 |
| P3 | 23.825229 | 24.784043 | +0.958814 |
| 합계 | 78.919104 | 문제별 최고 합 | +6.472940 |

이는 `+3점` 목표의 주 레버가 P1임을 뜻한다. P2·P3 개선도 합산에는 중요하지만, 현재 관측된 성능 범위만 보면 어느 한 문제도 P1만큼 큰 점수 상승을 증명하지 못했다. 이는 가능성의 상한이 아니라 현재 리더보드에서 입증된 headroom이다.

## 왜 로컬 hard veto가 실패하는가

### 우리 데이터의 직접 증거

| 문제·contrast | local benefit | official benefit | 결과 |
|---|---:|---:|---|
| P1 Router vs B | +0.002230 F1 | +0.024163 F1 | 같은 방향, 공식 10.84배 |
| P1 Intersection vs B | -0.001594 F1 | +0.009218 F1 | 방향 역전 |
| P2 global -t vs O | +0.004883℃ | +0.003847℃ | 같은 방향 |
| P2 L2 -t vs O | +0.001121℃ | -0.000832℃ | 방향 역전 |
| P2 L4 -t vs O | +0.000357℃ | +0.004549℃ | 같은 방향, 공식 12.76배 |
| P3 reverse global | -0.002166m | +0.007999m | 방향 역전 |
| P3 reverse 18/24h | -0.002037m | +0.007689m | 방향 역전 |

여기서 benefit은 클수록 좋게 통일했다. P2/P3는 `RMSE_incumbent - RMSE_candidate`다. 전체 원장은 [`local_official_calibration.json`](C:/Users/cedis/PycharmProjects/PythonProject/reports/next_day_breakthrough_deep_research_20260827_v1/local_official_calibration.json)에 있다.

전 문제를 하나의 배율로 보정하는 것도 정당화되지 않는다. 표본이 작고, 후보들이 family 안에서 상관되어 있으며, P2/P3 일부는 정확히 같은 배포 계보가 아니다. 사용 가능한 결론은 **family별 공식 운송성을 따로 학습해야 한다**는 것이다.

### 문헌의 경계

- Cawley와 Talbot은 유한 검증 기준의 분산을 무시하고 모델 선택까지 최적화하면 성능 평가가 낙관적으로 편향될 수 있으며, 그 크기가 알고리즘 간 차이와 비슷할 수 있음을 보였다([JMLR 원문](https://www.jmlr.org/papers/v11/cawley10a.html)). 이는 반복 사용한 우리 OOF를 최종 진실로 취급하면 안 된다는 근거다.
- Blum과 Hardt는 반복 제출로 후보가 leaderboard holdout에 적응하면 Public 점수가 더 이상 독립 평가가 아니라고 정식화했다([PMLR 원문](https://proceedings.mlr.press/v37/blum15.html)). 이는 공식 점수를 HPO oracle로 무제한 사용하면 안 된다는 근거다.
- 고정 예산 best-arm identification은 제한된 횟수에서 어떤 대안을 시험할지 자체가 별도의 순수 탐색 문제임을 다룬다([Qin, 2022](https://proceedings.mlr.press/v178/open-problem-qin22a.html)). 본 보고서의 3-slot 배치는 그 아이디어를 대회 운영에 맞게 단순화한 실무 규칙이며, 해당 논문의 정리나 최적 알고리즘을 직접 적용했다는 뜻은 아니다.

## 새 공식 탐색 제출 gate

로컬 개선량 `> 0`은 hard gate에서 제거한다. 대신 다음을 적용한다.

### P0 — 제출 불가

하나라도 실패하면 official probe를 하지 않는다.

- CSV schema, 행·키·순서, 값 범위, hash, lineage가 검증됨
- test label/숨은 target 사용 없음
- 전체 train에서 재현 가능한 최종 배포 recipe가 있음
- 단순한 checkpoint 관찰점이 아니라 실제 test 추론까지 완결 가능함
- 후보가 기존 제출과 동일하거나 사실상 동일하지 않음

### P1 — 점수를 움직일 능력

다음 중 하나 이상을 요구한다.

- 같은 family의 공식 contrast가 방향 또는 곡률을 이미 지지함
- 변경 행·구간·예측 RMS가 충분히 커서 `+3점` 또는 그 실질적 일부를 만들 가능성이 있음
- 새 backbone·표현·외부 context처럼 기존 incumbent가 못 보던 오류 집합을 겨냥함
- 로컬과 공식의 체계적 역전이 확인된 family라서 로컬 열세가 오히려 탈락 근거가 되지 않음

### P2 — 정보가 다음 행동을 바꿈

업로드 전에 점수 구간별 행동을 적는다.

- 개선: 현 Public best 교체, 같은 축의 후속 최적화
- 비열등/근소 열세: 앙상블·subset·threshold의 제한된 후속
- 큰 열세: 해당 exact deployment family 종료

어느 점수가 나와도 다음 행동이 같다면 제출 정보가치는 낮다.

### P3 — 3개 슬롯 구성

문제마다 매일 다음 역할을 원칙으로 한다.

1. **Exploit**: 같은 family 공식 증거가 가장 강한 최종 성능 후보
2. **Structural challenger**: 기존 모델과 오류 구조가 다른 full-scale 후보
3. **Diagnostic pair/ablation**: 1·2번의 결과를 해석하거나 다음 날 최적점을 정하는 후보

세 개 모두 미세한 동일 축 후보이거나, 세 개 모두 검증되지 않은 새 구조인 배치는 피한다. 첫 점수 전에 파일·hash·순서·점수 구간별 행동을 동결하고, 같은 날 중간 점수로 나머지 파일을 바꾸지 않는다. 이 규칙은 Public 과적합 위험을 줄이면서도 사용하지 않으면 사라지는 일일 슬롯의 정보가치를 보존한다.

## 현재 후보 재판정

### P1

| 후보 | 새 판정 | 이유 |
|---|---|---|
| MS-TCN e150 full-train + Router union | **`BUILD_THEN_OFFICIAL_PROBE`** | pooled `+0.003887 F1`, Q3 `+0.017209`, Q4 `-0.015441`, CI90 `[-0.01315,+0.02114]`. 최종 채택 근거는 혼합적이나 새 backbone이고, P1이 유일하게 +3.5점의 입증된 headroom을 가짐. 과거 P1은 작은 local 효과가 official에서 10배 확대된 전례가 있어 로컬 CI만으로 버릴 수 없음 |
| MS-TCN e150 G/S-only additions | **`BUILD_THEN_OFFICIAL_PROBE`** | station local delta가 G `+0.039618`, S `+0.007326`, I `-0.010648`. full candidate와 paired하면 모델 자체와 I-ORS transport 위험을 분리함. 단, Q3/Q4를 본 뒤 만든 exploratory variant임을 명시 |
| Round E GI no-removals | **`LEFTOVER_PROBE`** | 공식 Router 구성요소를 분해하지만 217행 미세 postprocess이며 +3점 규모의 주 후보는 아님. P1 세 슬롯 중 최대 한 슬롯 권고 |
| Round E G-only, I-only | **`DEFER_IF_MSTCN_READY`** | 셀별 정보는 주지만 세 슬롯 모두 소비할 만큼 최종 성능 규모가 크지 않음 |

중요한 운영 조건: 현재 e150 결과는 OOF 진단만 있고 test submission CSV가 없다. `BUILD_THEN_OFFICIAL_PROBE`는 업로드 승인이 아니라 full-train 3-seed refit, test inference, Router-preserving union, 두 variant의 독립 QA가 끝난 뒤 공식 1회씩 검증할 가치가 있다는 뜻이다.

### P2

| 후보 | 새 판정 | 이유 |
|---|---|---|
| Round E U | **`SUBMIT_EXPLOIT`** | official all-row quadratic이 이전 global 예측을 `2.64×10^-7 RMSE` 오차로 재현했다. U 예상 RMSE `0.535750480`, 현 best 대비 `0.000785520℃` 개선. +3점급은 아니지만 성공확률이 가장 높은 exploit |
| Round E endpoint envelope E | **`SUBMIT_DIAGNOSTIC`** | U에 대한 bounded physical ablation. 로컬은 약한 순서 증거로만 사용하며 공식값이 다음 후처리 선택을 바꿈 |
| Round E full PAVA F | **`LEFTOVER_PROBE`** | 단조 수온 제약은 밀도 안정성과 동일하지 않아 위험이 더 큼. 다른 full-scale 후보가 없을 때 세 번째 슬롯으로 사용 |
| checkpoint p0.85 | **`NO_AS_IS`** | r3 대비 `-0.047112℃` 개선 신호는 있으나 outer 결과를 본 뒤 고른 prefix이며 full checkpoint는 `+0.010618℃` 악화. p0.85는 아직 최종 배포 recipe가 아니므로 그대로 공식 제출할 수 없음 |

P2의 핵심은 로컬을 믿지 않는 것이 아니라 **공식 RMSE 이차대수처럼 실제 Public에서 식별된 축을 우선**하는 것이다. 새 구조가 준비되면 F보다 그 구조를 세 번째 슬롯에 넣는다.

### P3

| 후보 | 새 판정 | 이유 |
|---|---|---|
| Round E long α=-2 | **`SUBMIT_EXPLOIT_GUARD`** | 예상 RMSE `0.598986994`; early lead를 no-op으로 두고 이미 공식 개선된 12/18/24h를 합침 |
| Round E long α=-4 | **`SUBMIT_CURVATURE_PROBE`** | current local analogue가 official과 전부 역전됐으므로 official 곡률 확인 자체가 높은 정보가치 |
| Round E 18/24h α=-4 | **`SUBMIT_CURVATURE_PROBE`** | 12h와 18/24h 기여를 분리해 다음 날 최적 subset을 결정 |
| nested checkpoint 4/7/1 | **`NO_AS_IS`** | fixed8보다 `+0.003637m`, incumbent보다 `+0.041491m` 악화이고 같은-family official transport 근거가 없음 |
| ERA5 context-transfer | **`WAIT_UNEVALUATED`** | 별도 고정 실험이 진행 중이다. 결과 전에는 성공·실패·공식 가치 어느 쪽도 선언하지 않음 |

P3 세 축은 `+3점`을 직접 만들 가능성은 낮다. 현재 리더보드에서 입증된 P3 headroom은 약 `0.96점`이다. 그러나 local selector가 이 family에서 반대로 작동했으므로 공식 3점 곡률 배치는 장기적으로 가치가 있다.

## 권고하는 2026-08-27 배치

### MS-TCN 배포형이 시간 내 QA 완료되는 경우

- P1: e150 full / e150 G·S-only / Round E GI no-removals
- P2: U / endpoint envelope / PAVA 또는 새 full-scale 구조 1개
- P3: long -2 / long -4 / 18·24h -4

### MS-TCN 배포형이 시간 내 준비되지 않는 경우

- 이미 QA PASS인 Round E를 그대로 제출할 수 있다.
- 다만 P1 G/I/GI 세 점은 `+3점 exploit 세트`가 아니라 다음 full-scale P1 모델을 위한 공식 support-transport 실험으로 기록한다.

## 승격 문구 규칙

- 로컬이 나쁨: `LOCAL_RISK_SIGNAL`, 자동 `NO_OFFICIAL_ACTION` 아님
- official probe 가치 있음: `OFFICIAL_PROBE_ELIGIBLE`, 최종 채택 아님
- Public 최고 갱신: `PUBLIC_BEST_ONLY`, Private 일반화 증명 아님
- fresh local + reproducibility + official 방향: `PRIVATE_READY`

이 네 문구를 섞지 않는다.

## 한계와 중단 기준

- 14개 contrast는 작고 family 상관이 강해 local→official 확률모형을 안정적으로 적합할 수 없다. 숫자 확률 대신 등급 판정을 사용했다.
- authenticated UI의 Public 점수는 같은 숨은 표본을 재사용한다. 오늘 3회가 새 독립 검증 3회라는 뜻은 아니다.
- 현재 리더보드 headroom은 다른 팀이 이미 달성한 범위일 뿐 이론적 상한이 아니다.
- P1 MS-TCN e150은 아직 official CSV가 없어 `BUILD_THEN_*` 상태다. 이를 준비하지 않고 기존 OOF 숫자만 제출 가치로 과장하면 안 된다.
- 후보가 큰 공식 개선을 내더라도 다음 날에는 같은 Public에 과도하게 세부 튜닝하지 않고, family-level 방향·subset·곡률만 업데이트한다.

## 조사 범위와 종료 이유

로컬 공식 점수 원장, Round D 9개 공식 결과, Round E frozen manifest, checkpoint retrospective, 로그인된 공식 리더보드·제출관리·P1 문제 화면, 그리고 model-selection/leaderboard/fixed-budget exploration 1차 문헌을 확인했다. 핵심 결론은 서로 독립적인 세 증거군—우리의 실제 sign reversal, 공식 score headroom, 반복 검증 과적합 문헌—에서 수렴했다. 추가 일반 문헌은 후보별 판정을 바꿀 가능성이 낮아 여기서 중단했다.
