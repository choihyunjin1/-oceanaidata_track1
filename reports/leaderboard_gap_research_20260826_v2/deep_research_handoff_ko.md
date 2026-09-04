# 해양 해커톤 2–3점 개선 연구 중간 종결 및 Deep Research 인계서

작성일: 2026-08-26  
팀: 분당독고다이  
범위: P1·P2 로컬 연구와 리더보드 운송성 검토. P3 ERA5 고정 실험은 별도 자동화로 유지하며 본 연구에서 변경하지 않았다.

## 기술 요약

결론부터 말하면, **리더보드에서 총 2–3점을 개선할 수 있다는 문제 설정은 여전히 타당하지만 이번 사이클에서는 새 모델의 과학적 우열을 판정하지 못했다.** P1은 72-fit 장기 이벤트 재점수 실험을 실행하기 전에 Windows 격리 실행기의 권한 경계 결함이 발견됐고, 사용자 지정 중단선에 따라 r8에서 멈췄다. P2는 기존 trajectory 계열의 비식별 원인을 확인한 뒤 식별 가능한 raw-Celsius 2-window 설계를 완성하고 독립 QA까지 통과했지만, 명시적 구현 승인 전 상태로 정지했다.

현재 공개점수 스냅샷에서 팀 분당독고다이는 78.092863점으로 3위이며 1위 82.678604점과 4.585741점 차이다. 문제별 관측 headroom은 P1 3.035360점, P2 1.817999점, P3 0.374402점이다. P1 제출 후보급 최소 효과와 P2 선호 효과를 함께 달성하면 약 +1.63점, P3의 관측 headroom까지 더하면 약 +2.00점이다. P1 stretch 효과까지 포함한 계획 상한은 약 +2.79점이다. **이 값들은 달성 점수가 아니라 사전등록된 효과 크기를 공개점수 환산식으로 옮긴 계획 범위다.**

이번 사이클의 가장 중요한 성과는 모델 성능 수치가 아니라 다음 세 가지다.

1. P1·P2에서 소수점 변동이 아니라 공개점수 약 +0.7점 이상에 해당하는 로컬 효과 기준을 고정했다.
2. P2의 과거 실패가 단순 하이퍼파라미터 부족이 아니라 학습 창 비식별성과 target 정의 문제임을 분리했다.
3. P1은 잘못된 one-shot 실행으로 연구 기회를 소모하기 전에 권한·재실행·bootstrap 결함을 차단했고, claim과 fit을 모두 0으로 보존했다.

## 현재 점수 여지와 채택 기준

### 공개점수 스냅샷

| 항목 | 값 | 해석 |
|---|---:|---|
| 팀 총점 | 78.092863 | 분당독고다이, 3위 |
| 1위 총점 | 82.678604 | 총점 차이 4.585741 |
| P1 관측 headroom | 3.035360 | 가장 큰 개선 여지 |
| P2 관측 headroom | 1.817999 | 두 번째 개선 여지 |
| P3 관측 headroom | 0.374402 | 별도 ERA5 실험 외 우선순위 낮음 |

리더보드-로컬 근사 운송식은 다음과 같이 고정했다.

- P1: `points ≈ 6.7536890869 + 26.5800428671 × F1`
- P2: `points ≈ 33.3333252308 − 12.5475122444 × RMSE_C`

P1은 pooled F1 개선 `+0.0255` 이상, paired bootstrap CI90 하한 `+0.012` 이상을 제출 후보급 최소 기준으로 삼았다. 이는 계획상 약 `+0.6778점`이다. 연구 진행 최소 기준은 F1 `+0.0161`, CI 하한 `+0.007`이지만 이 수준만으로는 제출하지 않는다.

P2는 RMSE 차이 `candidate − incumbent ≤ −0.060°C`, paired-day CI90 상한 `≤ −0.040°C`를 의미 있는 기준으로 정했다. 공개점수 환산은 약 `+0.7529점`이다. 선호 기준은 `−0.0759°C / CI 상한 −0.050°C`이며 약 `+0.9524점`이다.

이 기준들은 결과를 본 뒤 낮추지 않는다. 공개 리더보드와 로컬 지표 사이에는 P1 A 제출의 방향 반전 사례가 있고 P2에는 양의 운송 사례가 부족하므로, 작은 로컬 개선은 신뢰하지 않는다.

## P1: 과학 가설은 미검증, 실행 인프라는 r8에서 종결

### 고정된 과학 가설

P1 Cycle 1은 장기 offset/drift 이벤트의 segment proposal과 경계·문맥 점수를 기존 예측에 재점수하는 실험이다. 과학 계약은 다음과 같다.

- anchor fit 9개, inner fit 54개, outer fit 9개: 총 물리 fit 72회
- 과학 materialization 21회
- inner chronological window 정확히 3개, outer locked window 정확히 3개
- 특징, seed, 모델, decoder, postprocess, decision gate 고정
- 공식 test/sample/submission/candidate 접근 0
- 결과 기반 재실행·튜닝 0

과학적 RESEARCH_GO는 pooled F1 `+0.0161`, CI90 하한 `+0.007`, 3개 outer fold 중 최소 2개 개선, 3개 station 중 최소 2개 개선, Q3와 station G 비악화, noise recall·false positive·long-event recall·interval precision 안전장치를 모두 요구한다. 제출 후보급은 더 높은 `+0.0255 / +0.012`를 요구한다.

### r7에서 확인된 실행 경계 문제

r7은 10,745개 서드파티 runtime 파일, 58개 private snapshot 파일, tzdata·pyc·native DLL, canonical root와 one-shot claim을 봉인했다. 테스트와 두 번의 preflight는 통과했지만 독립 red-team에서 다음 P0가 발견됐다.

- 숨은 worker가 canonical live authority를 직접 열지 않고 복제된 seal·QA·authorization을 신뢰할 수 있음
- 격리 runtime 활성화 전 `ZoneInfo("Asia/Seoul")` 호출로 첫 one-shot claim을 0-fit 상태에서 소모할 수 있음
- Python bootstrap이 읽는 `VCRUNTIME140.dll`과 live stdlib가 봉인 밖에 있음

따라서 r7 authorization은 생성하지 않았다.

### r8에서 닫은 부분과 마지막 P0

r8은 다음을 구현했다.

- 격리 Python home 4,044개 파일과 서드파티 runtime 10,745개 파일의 byte-copy·hash 봉인
- root VC runtime, Python DLL, stdlib, tzdata, Defender provider의 정확한 inventory
- pipe-only 32-byte capability, 실제 parent process handle, canonical claim handle
- worker의 canonical auth·QA·seal·claim 직접 열기와 lifetime hold
- private bundle RX-only ACL로 native-loading 중 DLL 삽입 차단
- exact 21,600초 wall contract, 64자 lowercase nonce, future/stale/reuse 차단
- direct/copied hidden worker, parent/PID, clock/nonce, ACL insertion, isolated tzdata zero-fit 공격 테스트

봉인 테스트는 88개 중 87 PASS, 기존 Windows symlink 권한 skip 1이었고 Ruff와 두 번의 byte-identical preflight도 통과했다. 그러나 봉인 후 실제 프로세스 adversarial test에서 정상 프로세스의 `NtQueryInformationProcess` 호출이 실패했다. sealed runner에서 ctypes `argtypes/restype`가 빠진 것이 유력 원인이다. copied-interpreter initial-map probe도 test-side ctypes 정의 문제 가능성을 남긴 채 미해결이다.

이 결함은 코드 몇 줄로 보일 수 있지만, true-parent 검증은 hidden-worker 자기승인을 막는 핵심 권한 경계다. 사용자와 합의한 “r8 마지막 수정” 규칙에 따라 r9를 만들지 않았고, P1 actual authorization·claim·fit·materialization·score는 모두 0이다. **따라서 P1 장기 이벤트 가설은 실패한 것이 아니라 아직 검증되지 않았다.**

## P2: 비식별 실패를 분리하고 실행 가능한 최종 설계를 확보

### 기존 trajectory Cycle 1의 구조적 종료

trajectory DTW/curvature 계열은 사전등록된 2024년 3월 inner window에서 target layer 2·3·4의 truth 지원이 모두 0이었다. March key는 4,464개였지만 각 target layer의 truth/anchor/both가 `0/4,454/0`이었다. 따라서 모델 우열을 비교할 수 없었고 `FINAL_PRECLAIM_NO_GO_INNER_WINDOW_UNIDENTIFIABLE`로 종료했다. local 3-window 실행과 결과 기반 재설계는 하지 않았다.

기존 raw-Celsius v1도 1–2월과 3–4월의 target layer 2·3·4가 모두 빈 그룹이었다. frozen `FAIL_CLOSED` 정책 때문에 빈 창을 결과 후 임의로 삭제하거나 합칠 수 없었다.

### raw-Celsius Cycle 2/2 FINAL 설계

결과 수치가 존재하기 전에 식별 가능한 두 창만 사용하는 append-only 설계를 고정했다.

- 학습 창: 2024년 5–6월, 2024년 7월–8월 24일
- 학습 행: 총 45,935개
- target layer: 2·3·4, 총 6개 layer-window group 모두 비어 있지 않음
- 특징: 기존 public-only 순서 그대로 55개
- target: `target_temperature_c − public interpolation baseline_c`
- seed: 20260823, 20260824, 20260825
- LightGBM 파라미터 고정, 물리 fit 정확히 3회
- blend, inference gate, postprocess, stage 2 없음
- 평가: 2024년 9–10월 exact OOF 26,273행, 8,779 timestamps, 61 KST days
- incumbent row-pooled RMSE: 0.4477930822°C
- paired-day bootstrap 5,000회, seed 20260826

독립 QA는 파일·lineage·source hash, 45,935행, 6개 group/day support, raw group weight 각 `1/6`과 전체 `1`, 정규화 weight 평균 `1`, 55개 특징, exact OOF key/truth/prediction digest, nanosecond roundtrip, gate와 점수 환산을 모두 재구성했다. 판정은 `P0=0 / P1=0`이었다.

다만 이 설계는 명시적 governance 승인 전이며 implementation·3-fit 실행은 0회다. 이번 사이클에서는 P1 인프라 종결과 사용자 중단 지시를 우선해 P2를 실행하지 않는다.

## 로컬-공식 채점 운송성의 현재 한계

로컬 지표를 공식 점수의 정밀 대리변수로 아직 볼 수 없다.

- P1 제출 A는 로컬 방향과 공개점수 방향이 뒤집혔다.
- P1 제출 B는 약한 양의 신호만 보였다.
- P2 공식 점수는 명백한 나쁜 모델을 배제하는 rejection surface로는 유용하지만, 작은 양의 로컬 개선이 그대로 운송된다는 사례가 없다.
- 각 문제당 남은 제출 기회가 제한되어 있어 소수점 개선 탐색보다 큰 효과·일관된 CI·세그먼트 안전장치를 요구해야 한다.

따라서 다음 연구에서도 공개점수로 하이퍼파라미터를 역추정하거나 제출 결과를 로컬 tuning label로 사용하지 않는다. 사전등록된 2–3회 비교와 높은 효과 기준을 유지한다.

## 왜 지금 Deep Research가 필요한가

현재 병목은 계산량이 아니라 탐색 관점이다. 로컬 연구는 이미 다음을 소진했다.

- P1: 장기 이벤트 representation/decoder, station-season robust objective, matched-budget 비교의 로컬 설계
- P2: trajectory/curvature transfer, normalized residual, raw-Celsius residual과 식별성 검토
- P3: 별도 고정 ERA5 context-transfer 실험 진행 중

추가 로컬 버전은 인프라·계약을 더 정교하게 만들 수는 있지만 새로운 과학적 가설을 공급하지 않는다. 특히 P1 r7→r8 반복은 모델 개선이 아니라 one-shot 연구 거버넌스 방어였다. 이제는 외부 논문, 검증된 대회 해법, 해양 물리 기반 interpolation, event-detection 방법을 체계적으로 조사해 **현재 feature/model family 밖의 후보를 좁히는 편이 기대가치가 높다.**

## Deep Research에 넘길 핵심 질문

### P1: 희소 장기 이상 이벤트 탐지

1. 불규칙·희소 positive event에서 pointwise binary classifier보다 event-level F1과 duration/segment 경계를 직접 개선한 최근 방법은 무엇인가?
2. semi-Markov, change-point detection, temporal convolution/transformer, weak supervision, structured prediction 중 작은 데이터와 강한 distribution shift에서 가장 재현성 높은 계열은 무엇인가?
3. 긴 offset/drift 이벤트 recall을 올리면서 spike·flatline 예측을 정확히 보존하는 constrained decoder 또는 residual correction 방법은 무엇인가?
4. station·season shift에서 worst-group F1을 개선하되 pooled F1을 희생하지 않는 GroupDRO, distributionally robust calibration, hierarchical partial pooling의 실증 근거는 무엇인가?
5. F1을 직접 최적화하거나 differentiable surrogate로 다루면서 시간 누수 없이 threshold·duration을 선택하는 방법은 무엇인가?

### P2: 해양 수직 프로파일 보간과 domain transfer

1. 관측 가능한 상·하부 layer만으로 중간 수온을 복원할 때 spline, Gaussian process, kriging, functional data model, neural operator, state-space 모델의 비교 근거는 무엇인가?
2. TEOS-10·밀도 안정성·수온약층 같은 물리 제약을 leakage 없이 학습 objective 또는 postprocess에 넣어 RMSE를 줄인 사례는 무엇인가?
3. raw-Celsius residual과 normalized residual 중 계절·수심·profile scale shift에 더 안정적인 target parameterization은 무엇인가?
4. layer × season × day 불균형에서 group-balanced MSE, worst-group risk, quantile/heteroscedastic loss 중 공개 분포 운송성이 높은 방법은 무엇인가?
5. 작은 공개-only feature set에서 trajectory similarity를 쓰되 빈 historical window와 missing layer를 견디는 causal analog 또는 metric-learning 방법은 무엇인가?

### 공통: 제한된 제출과 로컬-공식 괴리

1. 2–3번의 제출만으로 local-public transport를 보수적으로 업데이트하는 Bayesian 또는 sequential decision rule은 무엇인가?
2. leaderboard probing 없이 public score variance와 selection bias를 감안해 제출 후보를 고르는 기준은 무엇인가?
3. 동일한 local OOF에서 old fully-tuned model과 new under-tuned family를 공정하게 비교하는 budget-matched protocol은 어떻게 설계해야 하는가?

## Deep Research 출력 요구사항

Deep Research는 단순 논문 목록이 아니라 다음 형식으로 반환해야 한다.

1. P1 후보 최대 3개, P2 후보 최대 3개
2. 후보별 핵심 메커니즘, 현재 접근과의 구조적 차이, 필요한 특징·데이터
3. 시간 누수·공식 데이터 접근·external-data 규정 위험
4. 예상 구현 난이도와 fit budget
5. 현재 동결 incumbent와 비교할 단 하나의 primary metric·CI·segment guard
6. 2-cycle 안에 반증 가능한 최소 실험
7. 실패 시 무엇을 배울 수 있는지와 명확한 stop rule
8. 원 논문·공식 문서·재현 가능한 코드 등 1차 출처 우선

## 다음 실행 규칙

Deep Research 이후에도 무한 탐색은 하지 않는다.

1. P1과 P2에서 각각 가장 구조적으로 다른 후보 1개만 선택한다.
2. 결과를 보기 전에 windows, features, target, seeds, budget, gates를 고정한다.
3. 문제당 최대 2-cycle, 결과 기반 분기 1회만 허용한다.
4. P1은 F1 `+0.0255 / CI 하한 +0.012`, P2는 RMSE `−0.060°C / CI 상한 −0.040°C`를 제출 후보급 하한으로 유지한다.
5. local gate를 통과해도 별도 독립 QA와 local-public 운송성 검토 전에는 제출 파일을 만들지 않는다.
6. 공식 test/sample/submission/candidate 접근과 P3 고정 실험 변경은 각 별도 권한 범위에서만 수행한다.

## 재현 근거

- 리더보드 연구 스냅샷: `reports/leaderboard_gap_research_20260826_v1/research_snapshot.json`, 7,350B, SHA-256 `943ff9ac9723c36445a2a6af2112073799a4bf52185fde91d1e6e5bad9bdfb45`
- P1 과학 설계: `configs/experiments/p1_long_event_segment_proposal_rescore_v1_design.json`, 14,389B, SHA-256 `31b0bde27d8ef7e2b42135709563cca0bcca61c6ec6fdabefbb3530906869563`
- P1 r7 독립 NO_GO: `reports/p1_long_event_segment_proposal_rescore_v7_independent_preexecution_qa_20260826.json`, 9,059B, SHA-256 `066d51a32a16175a1e0fb5829ff44d6699ea398901d38a1fbbd92e51a20fe6ff`
- P1 r8 terminal NO_GO: `reports/p1_long_event_segment_proposal_rescore_v8_terminal_no_go_20260826.json`, 5,927B, SHA-256 `aefed99f005bb213ba459df3c1e3112e4c1a5a58451e227104ea82acc7a2fe82`
- P2 최종 설계: `configs/experiments/p2_public_group_balanced_celsius_residual_v2_final_design.json`, 15,386B, SHA-256 `814f0f23780159b3fa6c93026a7ef19145c8745311f44bb451f43297763808d1`
- P2 식별성 증명: `reports/p2_public_group_balanced_celsius_residual_v2_identifiability_certificate_20260826.json`, 5,847B, SHA-256 `d2aed5e14d04600d50b49346f24a040eb66e2ce12a8b6c087b2c43587a2d8930`

## 최종 상태

- P1 Cycle 1: `NOT_EXECUTED`, infrastructure terminal `NO_GO`, scientific claim 미평가
- P2 Cycle 2/2: design QA PASS, implementation·3-fit 미승인·미실행
- P3: 별도 고정 ERA5 자동화에 위임, 본 연구에서 무변경
- 생성된 submission/candidate/upload: 0
- 권고: **현재 로컬 연구를 중지하고 본 문서를 입력으로 Deep Research를 실행한다.**
