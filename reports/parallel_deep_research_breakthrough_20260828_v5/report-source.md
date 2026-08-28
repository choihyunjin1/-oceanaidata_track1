# P1·P2·P3 새 돌파구 병렬 딥리서치 v5

작성일: 2026-08-28 KST
성격: 내부 기술 의사결정 및 다음 실행 사전등록
범위: 공개 1차 문헌·공식 코드·현재 저장소 실패 계보 교차검증

## 결론

새 돌파구 후보는 세 개로 좁혔다.

1. **P2 BayOTIDE형 동적 저차원 상태공간 공동 복원**: 공식 OAS 상승이 보여 준 프로필 공분산 신호를 시간축 latent state로 확장한다. 새 구조 중 1순위다.
2. **P1 TS2Vec형 조건부 normal-prototype proposal generator**: 라벨이 부족한 event ranker 대신 무라벨 연속구간에서 표현을 학습해 proposal 자체를 새로 만든다.
3. **P3 TimeXer형 past-exogenous direct 6-lead**: sparse expert router를 버리고 파고와 과거 기상을 비대칭 encoder로 직접 결합한다.

현재 즉시 공식 검증 가치가 있는 산출물은 새 구조가 아니라 기존 **P2 seasonal OAS alpha=0.40 probe**다. 이 파일은 26,061행·lineage·PAVA·해시 QA를 통과했고, 공식 alpha=0.10 및 0.20 관측과 제출 벡터 기하로 alpha=0.20 대비 약 +0.23~+0.44점의 조건부 기대가 기록돼 있다. 그러나 업로드는 이번 요청의 범위가 아니며 정확한 파일 재승인 없이 실행하지 않는다.

이번 연구로 공식 test/sample/submission 경로를 열거나 새 제출 CSV를 만들거나 업로드하지 않았다. 세 새 구조는 `RESEARCH_PREREGISTERED_NOT_AUTHORIZED_FOR_EXECUTION` 상태다.

## 왜 이번 후보가 이전 실패의 반복이 아닌가

### P1 — reconstruction이 아니라 contrastive representation

기존 masked reconstruction TCN은 full-fraction에서 incumbent보다 크게 나빴고, direct interval set은 recall을 늘리는 대신 대규모 false positive를 만들었다. 동결 83-bank ranker는 qualification 양성 proposal 1개로 학습 전 종료됐다.

TS2Vec는 augmented context view 사이의 hierarchical contrastive agreement로 각 timestamp representation을 학습한다. 따라서 label이나 synthetic anomaly를 encoder loss에 넣지 않고도 정상 문맥의 다중 시간척도 표현을 만들 수 있다. 이번 후보는 논문의 SPOT·test anomaly ratio·point adjustment를 가져오지 않고, train-normal prototype/kNN 거리와 고정 event decoder만 사용한다. [TS2Vec 논문](https://ojs.aaai.org/index.php/AAAI/article/view/20881), [공식 코드](https://github.com/zhihanyue/ts2vec)

핵심 위험은 명확하다. 원 논문의 anomaly 응용은 P1의 장기 offset/drift event와 같은 검증 계약이 아니다. 그러므로 qualification에서 새 eligible long event 2개 이상, 최소 2개 cell 방향 일치, FP/day cap, 7-day block bootstrap 개선확률 0.8을 모두 요구한다.

### P2 — static profile이 아니라 dynamic functional state

CMFPCA는 profile별 정적 latent score를 조건부 복원했지만 frozen OOF에서 OAS20보다 0.022734°C 악화했다. public-only heave는 strongest common OOF보다 0.000011°C 악화했고 활성률이 0.0687%뿐이었다.

BayOTIDE는 다변량 시계열을 여러 functional low-rank factor의 결합으로 표현하고, GP prior를 equivalent SDE/state-space로 바꿔 전역 trend와 periodic pattern을 선형 비용으로 갱신한다. P2에서는 actual-depth로 정렬한 T/S 채널에 3개 Matérn trend factor와 12.42h·24h periodic factor를 고정하고, 공개층 관측이 target blackout 동안에도 latent state를 갱신하게 한다. [BayOTIDE 논문](https://proceedings.mlr.press/v235/fang24d.html), [공식 코드](https://github.com/xuangu-fang/BayOTIDE)

반증은 논문 benchmark의 결측 메커니즘이다. 주된 실험은 P2의 61일 target-channel blackout과 다르다. 따라서 random point mask는 주 검증으로 쓰지 않고, 세 target layer의 T/S를 동시에 가린 기존 3개 historical block과 7일 purge를 그대로 사용한다. strongest common OOF 대비 pooled delta ≤ -0.003°C, CI90 상한 <0, 2/3 fold 개선, worst-layer 악화 ≤ +0.005°C를 요구한다.

### P3 — expert selection이 아니라 direct endogenous/exogenous forecasting

v4 ERA5 router는 I4에서 -0.001150m 신호가 있었지만 CI90 상한이 0이고 intervention 1.389%, outer support 1/3에 불과해 모든 prediction이 incumbent fallback됐다. Chronos-2와 TSMixer 계열도 기존 outer에서 명확히 악화했다.

TimeXer는 endogenous temporal patch의 self-attention과 exogenous variate의 cross-attention을 분리한다. P3에서는 미래 ERA5나 미래 기상을 쓰지 않고 과거 48시간의 hs를 endogenous patch로, tp·hmax·풍속·풍향·기압·기온·습도·mask를 exogenous token으로 사용한다. 6개 lead residual을 한 번에 직접 출력해 expert-support 병목을 없앤다. [TimeXer 논문](https://arxiv.org/abs/2402.19072), [공식 코드](https://github.com/thuml/TimeXer)

가장 큰 위험은 작은 표본 Transformer 과적합이다. 3개 seed, train-only inner-best checkpoint, outer prediction seal을 고정하고 pooled delta ≤ -0.005m, 2/3 fold 개선, case-bootstrap CI90 상한 <0, worst-station +0.01m 이하, 12/18/24h 모두 비악화를 동시에 요구한다.

## 실행 우선순위

### 단기 점수 확인

P2 OAS alpha=0.40은 이미 생성·독립 QA가 끝난 공식 probe다. 새 구조 연구와 별개로, 사용자가 정확한 파일을 승인하면 오늘의 첫 공식 검증 슬롯 후보가 된다. 공식 alpha=0.10→0.20의 상승과 실제 벡터 기하가 근거이며, 로컬 OOF는 반대로 악화했으므로 champion 확정이 아니라 local-official transport를 측정하는 probe다.

### 새 구조 실행

1. P2 BayOTIDE bounded run: OAS 공식 신호와 직접 연결되고 계산비용이 낮거나 중간이다.
2. P1 TS2Vec bounded run: 30~90분 GPU screen으로 proposal support를 먼저 판정한다.
3. P3 TimeXer bounded run: 3 outer×3 seed로 약 2~4시간을 예상하며 앞의 두 결과와 병렬 실행할 수 있다.

세 후보 모두 결과 기반 factor 수, window, epoch, threshold, patch width를 다시 고르는 재실행은 허용하지 않는다. 실패하면 2순위 구조로 넘어가되 같은 계열의 gate 완화는 하지 않는다.

## 공개 구현 및 라이선스 경계

- TS2Vec 공식 저장소는 MIT license를 표시한다. 프로젝트에 소스를 그대로 복사하기보다 논문 구조를 현재 코드 규약에 맞춰 독립 구현하고 출처를 남긴다.
- ImputeFormer는 MIT 공식 구현이 있으나 BayOTIDE 실패 후 2순위다. actual-depth embedding이 원 구현과 다른 변형이라는 점을 명시해야 한다.
- MOMENT checkpoint는 학습자료 provenance와 권리 검토 전 HOLD다.
- BayOTIDE 공식 저장소는 README에서 아직 갱신 중임을 밝힌다. 논문 구조와 현재 dependency 호환성을 smoke test한 뒤 사용한다.
- TimeXer 공식 코드는 future covariate가 가능한 일반 setting을 포함하지만 P3 변형은 past-only로 제한한다.

## 대회 운영상 새로 확인한 점

2026-08-28 현재 공개 공식 홈페이지는 대학부 예선 종료·결과물 제출 마감을 **2026-09-30**으로 표시한다. 저장소의 9월 7일 메모와 충돌하므로, 제출·최종모델 잠금 같은 실제 행동 전에는 로그인 후 최신 공지와 문제 상세를 다시 확인해야 한다. [공식 홈페이지](https://oceanaidata.org/)

## 이번 사이클 산출물과 경계

- 사전등록 config 3개를 추가했다. 모두 실행 미승인 상태다.
- `gap-matrix.md`, `claim-ledger.json`, 이 `report-source.md`를 canonical research record로 남겼다.
- 공식 입력 접근 0, 새 prediction/submission CSV 생성 0, 업로드 0, Git commit/push 0이다.

## 최종 의사결정

전체 모델 공간을 다 본 것은 아니지만, 이번 두 파동에서는 과거 실패 계열과 겹치지 않고 공식 코드가 있으며 현재 컴퓨터로 bounded 검증 가능한 세 구조까지 수렴했다. 가장 합리적인 다음 행동은 **P2 OAS40의 공식 probe 여부를 별도로 결정하면서, 새 구조는 BayOTIDE → TS2Vec → TimeXer 순으로 실행하는 것**이다.
