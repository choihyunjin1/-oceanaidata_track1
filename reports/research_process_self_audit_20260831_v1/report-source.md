# 2026-08-31 연구 프로세스 자기감사

## 결론

**계속 돌파구를 못 찾은 주원인은 딥리서치 프롬프트의 문장 품질이 아니라, 이미 여러 번 노출된 로컬 선택면에서 새 후보를 계속 고르고 그 결과를 일반화 신호로 해석한 연구 구조다.** 프롬프트는 실패 목록·평가지표·중단 조건까지 상당히 구체적이었지만, `새 돌파구`를 요구하는 목적 함수가 남아 있어 외부 연구가 `실험하지 않음`보다 새로운 방법을 내놓도록 압박했다. 동시에 실패 원장은 사람이 읽는 넓은 이름 목록이어서, 의미상 같은 정보축이 다른 모델명으로 재진입하는 것을 자동 차단하지 못했다.

이 판단을 뒷받침하는 가장 강한 내부 증거는 세 문제에서 모두 나타난 **선택/프록시 양수 → 독립 확인/공식 음수의 방향 역전**이다.

| 문제 | 선택 또는 로컬 프록시 | 독립 확인 또는 공식 결과 | 판정 |
|---|---:|---:|---|
| P1 | Sobol trial18 Q2 pooled ΔF1 `+0.000565637` | sealed Q3/Q4 ΔF1 `-0.011889120` | 선택면 과적합 또는 시기 수송 실패 |
| P2 | copula historical proxy ΔRMSE `-0.010616065°C` | Public ΔRMSE `+0.012050°C` | 명시적 부호 역전 |
| P3 | CatBoost selection ΔRMSE `-0.0228625m` | 182-case confirmation ΔRMSE `+0.0079741m` | 전 fold·station·lead 비개선 |

반대로 실제 Public 개선은 대형 신규 모델보다 **이미 확인된 축을 저차원으로 분해한 고정 보정**에서 나왔다.

- P1: e150의 고정 15개 G-ORS 추가행은 제거 대비 표시 Public F1 `+0.004519`였다. 238개 S-ORS 추가행의 표시 marginal은 `0`이었다.
- P2: rank-1 계절 보정 중 bin17-only가 prior champion 대비 RMSE `-0.000015°C`, 점수 `+0.000187`로 새 Public 최고가 됐다. bin18은 반대 방향이었다.
- P3: 균일 KMA `alpha=0.425`는 prior champion 대비 RMSE `-0.000029m`, 점수 `+0.000473`을 얻었고, station 제거 실험은 G/I/S 세 station 모두 양의 기여임을 보였다.

따라서 다음 사이클의 기본 행동은 `새 모델을 더 많이 찾기`가 아니다. **새 후보를 제안하기 전에 후보 지문 중복, 식별 가능성, 계약 smoke, 미사용 확인면 존재 여부를 통과시키고, 통과하지 못하면 `NO_NEW_EXPERIMENT`를 정상 결론으로 허용**해야 한다.

## 무엇이 잘못됐나

### 1. 재사용된 선택면의 작은 이득을 구조적 신호로 과대평가했다

P1·P2·P3에서 선택 단계 이득이 모두 다음 단계에서 뒤집혔다. 이는 특정 알고리즘 하나의 실패보다, 후보 생성과 선택에 사용한 데이터면이 더 이상 독립적인 성능 추정면이 아니었음을 시사한다. Cawley와 Talbot은 유한 표본의 모델 선택 기준 자체가 과적합될 수 있고 그 크기가 알고리즘 간 차이와 비슷할 수 있다고 보였다. Blum과 Hardt는 반복적·적응적 리더보드 조회가 홀드아웃 과적합을 유발할 수 있음을 별도로 다룬다.

이 감사에서 말하는 `선택면 과적합`은 숨은 정답을 읽었다는 뜻이 아니다. 같은 과거창·프록시·Public 피드백을 보고 다음 가설을 고르는 적응 과정 전체가 추정치의 독립성을 약화했다는 뜻이다.

### 2. 프롬프트는 상세했지만 목적 함수가 `반증`보다 `발명`에 가까웠다

Gemini 교정 프롬프트는 metric, 우리 champion과 대회 leader의 구분, 정확한 실패 수치, 최대 세 가설, 누수 위험, 6~12시간 최소 실험, stop rule, 예상 점수 금지를 명시했다. 그럼에도 결과는 다음 문제를 반복했다.

1. 파일 수준 공식 A/B를 행별 TP·recall 증거로 확대했다.
2. 변동하는 제출 잔여 횟수를 고정 사실로 취급했다.
3. 이미 실패한 type/boundary cascade를 다시 제안했다.
4. 미확정 변환에서 예상 개선치를 만들려 했다.
5. row-level F1과 action-segmentation 지표를 혼동했다.

즉, `더 자세한 배경을 주면 해결된다`는 가설도 이미 부분적으로 반증됐다. 외부 연구 모델은 문헌 탐색과 가설 생성에는 유용하지만, 우리 실험 원장의 의미론적 중복 판정과 배포 결정을 스스로 맡길 수 없다.

### 3. 실패 원장이 넓은 문자열 집합이라 의미상 재실행을 막지 못했다

2026-08-30 원장은 exact closed group을 P1 `13`, P2 `10`, P3 `7`개 기록했다. 이는 정확한 재실행 금지에는 유용했지만 다음 필드를 기계적으로 비교하지 않았다.

- 사용 데이터축과 관측 가능 시점
- target 또는 residual 정의
- split/anchor/episode ID
- feature family와 모델 family의 분리
- postprocess·router·threshold
- comparator, metric, gate
- 실험 목적이 성능·정보·인프라 중 무엇인지

그 결과 모델명이 달라도 같은 정보축을 다시 쓰는 `의미상 재실행`을 사람이 매번 판정해야 했다.

### 4. 성능 후보, 정보 탐침, 인프라 수리를 한 승격 언어로 섞었다

- **성능 후보**는 독립 확인에서 이득과 불확실성을 통과해야 한다.
- **정보 탐침**은 점수가 오르지 않아도 서로 겹치지 않는 가설을 식별하면 가치가 있다.
- **인프라 수리**는 실행 계약을 고칠 뿐 성능 증거가 아니다.

P3 CatBoost v1의 incompatible grid와 v2 confirmation schema failure는 과학적 NO_GO가 아니라 기술 실패였다. 반대로 P3 contract repair 뒤 `+0.0079741m` 악화는 비로소 과학적 NO_GO였다. 이 세 종류를 같은 `성공/실패`로 부르면 외부 연구가 기술 복구를 성능 후보처럼 재포장하거나 정보 탐침을 제출 후보처럼 과대평가한다.

### 5. 고정 `3점` 승격 기준도, 단순히 `개선 > 0`으로 낮추는 것도 모두 부적절하다

P1 trial18은 Q2에서 작게 양수였지만 독립 확인에서 명확히 음수였다. 따라서 `+0이면 승격`은 이 후보를 잘못 살렸을 것이다. 반대로 P2 bin17과 P3 KMA처럼 작지만 정확히 동결된 저차원 공식 탐침은 유의미한 구조 정보를 줬다.

새 기준은 고정 점수 문턱이 아니라 단계별 증거 계약이어야 한다.

- discovery: 방향·sanity·계약 확인. 배포 주장 금지.
- confirmation: 제안에 사용하지 않은 window/episode에서 방향 안정성·불확실성·최악군 손상을 함께 평가.
- official information probe: 서로 겹치지 않는 고정 변형만 허용하고, 점수 최적화가 아니라 축 식별을 목적으로 한다.
- deployment: confirmation과 배포 계약을 모두 통과한 후보만 허용한다.

### 6. HPO의 실행 효율과 일반화 증거를 혼동했다

Random/Sobol search는 grid보다 유효한 설정을 효율적으로 찾는 도구다. 그러나 Bergstra와 Bengio의 결과는 `검색 효율`에 대한 근거이지, 재사용된 validation에서 찾은 최댓값이 새 분포로 수송된다는 보장이 아니다. Bouthillier 등도 sampling, initialization, hyperparameter choice의 변동이 비교 결론을 바꿀 수 있으므로 평균값 하나가 아니라 전체 파이프라인 변동을 고려해야 한다고 정리한다.

## 우리가 잘한 점과 과하게 한 점

### 유지할 것

- exactly-once lock, frozen hash, 공식/hidden 접근 카운터, 독립 QA는 잘못된 주장과 결과 기반 재튜닝을 억제했다.
- 기술 실패와 과학적 NO_GO를 구분했다.
- 공식 제출을 단순 점수 소비가 아니라 factorial information probe로 바꿨을 때 실제 구조 정보를 얻었다.
- `closed exact`와 `broad family closed`를 구분했다.

### 줄일 것

- 같은 증거를 여러 장문의 연구 문서에서 반복 설명하는 비용
- 후보 식별 전 대규모 HPO와 후속 confirmation 코드를 동시에 만드는 일
- 여러 연구자가 모두 `새 방법`을 내고 최종 중복 조정자가 없는 병렬화
- deadline 직전에 기술 계약 오류를 발견하는 순서

거버넌스는 폐기할 대상이 아니라 **앞단의 5분 지문·계약 검사로 압축**할 대상이다.

## 연구 프로토콜 v2

### A. 후보를 여섯 단계에서 걸러낸다

1. **문제 계약:** official metric, 현 champion, 데이터 가용 시점, 금지 입력을 한 화면에 고정한다.
2. **후보 지문:** `data_axes / target / split_ids / feature_family / model_family / postprocess / comparator / metric / gate / lane`을 생성한다.
3. **중복 감사:** exact match뿐 아니라 동일 data axis와 동일 comparator를 쓰는 semantic overlap을 표시한다.
4. **식별 가능성:** 현재 데이터에서 후보와 대립 가설이 다른 관측 결과를 내는지 확인한다. 아니면 실행하지 않는다.
5. **contract smoke:** synthetic 1-fit 또는 무학습 dry-run으로 schema·지원 parameter·출력 계약을 먼저 확인한다.
6. **최소 반증 → 미사용 확인:** 가장 싼 실험에서 살아남은 후보만, 제안 과정에 사용하지 않은 확인면으로 이동한다.

### B. 세 lane을 분리한다

| lane | 성공 정의 | 허용 결론 | 금지 |
|---|---|---|---|
| PERFORMANCE | 미사용 confirmation에서 안정적 개선 | 제출 후보 | discovery 최고점만으로 제출 |
| INFORMATION | 사전등록된 직교 변형이 축을 식별 | 다음 가설의 방향 | 한 결과로 행별/인과 추론 |
| INFRASTRUCTURE | 계약과 재현성 복구 | 새 experiment ID로 재검증 가능 | 성능 개선 주장 |

### C. validation stop rule을 추가한다

같은 proxy class에서 `selection 양수 → confirmation/official 음수`가 두 번 발생하면, 다음 모델을 찾지 않는다. 해당 proxy class를 `EXPOSED_FOR_SELECTION`으로 닫고 새 확인면·새 시간 블록·새 관측 축이 생길 때까지 performance lane을 중단한다.

### D. 이상치 제거는 기본 전처리가 아니라 별도 가설이다

이상치 제거는 센서 오류를 줄일 수 있지만 P1에서는 이상 자체가 target일 수 있고, P2/P3에서는 극값이 실제 해양 사건일 수 있다. 따라서 `제거 → 성능 개선`을 전제하지 않는다.

- 제거 기준은 target/official feedback을 보지 않고 train-only에서 고정한다.
- 삭제보다 robust loss, winsorized sensitivity, contamination flag를 먼저 비교한다.
- 제거 전후 support·station·season·lead 분포와 extreme-event recall을 함께 보고한다.
- 한 validation에서 좋아졌다는 이유로 cutoff를 조정하지 않는다.

## 딥리서치 프롬프트 v2

다음 템플릿을 문제별 연구의 기본 프롬프트로 사용한다.

> 역할: 당신은 새 모델 발명가가 아니라 반증 감사자다. 첫 번째 책임은 실험을 제안하는 것이 아니라 현재 증거로 새 실험이 식별 가능한지 판정하는 것이다.
>
> 고정 입력: official metric과 단위, 현 팀 champion, 사용 가능한 train-only 데이터축, 절대 금지 입력, closed candidate fingerprint 원장, 이미 노출된 selection surface, 아직 사용하지 않은 confirmation block, 시간·fit·공식 제출 예산.
>
> 1. 먼저 `NO_NEW_EXPERIMENT`가 최선인지 YES/NO로 답하라. YES이면 부족한 새 정보가 무엇인지 쓰고 종료하라.
> 2. NO이면 가설을 문제당 하나만 제시하라. 모델명보다 mechanism을 먼저 쓰라.
> 3. 후보 지문을 완성하고 실패 원장과 exact/semantic overlap을 표로 감사하라. overlap이 크면 후보를 폐기하라.
> 4. 현재 데이터가 후보와 null을 어떻게 구분하는지 예상 관측을 쓰라. 예상 공식 점수·보장 개선량은 쓰지 마라.
> 5. 30분 contract smoke와 가장 작은 반증 실험을 설계하라. 기술 실패와 과학적 NO_GO를 별도로 정의하라.
> 6. proposal/HPO에 한 번도 쓰지 않은 confirmation block을 명시하라. 없으면 PERFORMANCE lane 제안을 금지하라.
> 7. 성공, 실패, inconclusive 각각의 중단 규칙을 사전등록하라. 결과를 본 뒤 threshold·window·seed를 바꾸지 마라.
> 8. 공식 제출은 직교 information probe 또는 confirmation을 통과한 frozen candidate만 허용하라.
> 9. 핵심 근거는 1차 논문·공식 문서와 제공된 로컬 원장만 사용하고, 사실·추론·제안을 구분하라.

## 문제별 다음 판단

### P1

넓은 temporal model 탐색을 더 하는 것보다 G 15-row 효과와 S 표시 marginal 0을 설명하는 **행 특성의 train-only 기술 감사**가 먼저다. 단, 공식 A/B는 개별 행 정답이 아니므로 이를 pseudo-label로 학습하면 안 된다. 새 미사용 시기 블록이 없다면 성능 lane은 중단하고 정보 lane만 유지한다.

### P2

Gaussian copula exact recipe는 공식 부호 역전으로 닫는다. 현재 가장 신뢰할 정보는 bin17 양수·bin18 미세 음수의 분해다. 다음 연구는 왜 bin17 support가 수송되는지에 대한 support/season/profile 진단이어야 하며, 다시 전체 복잡 모델을 먼저 돌리지 않는다.

### P3

CatBoost challenger_21은 confirmation에서 닫혔다. 현재 수송이 확인된 축은 KMA 18/24h 보정이며 station 기여는 모두 양수다. 새로운 확인면 없이 model-family HPO를 반복하지 말고 lead×station calibration의 물리·support 안정성을 정보 lane에서 감사한다.

## 한계와 열린 질문

- 이 감사는 2026-08-30까지 저장된 로컬 원장과 Public 표시값을 사용한다. Private 성능은 모른다.
- `선택면 과적합`은 가장 잘 맞는 진단이지만, 각 역전에서 sampling noise와 실제 domain shift의 비중을 분리할 새 독립 표본은 없다.
- 파일 수(`딥리서치` 40, `돌파` 44, `예상 점수/개선` 23)는 검색어가 들어간 문서 수이며 독립 연구 횟수나 오류율이 아니다.
- Public의 작은 양수는 구조 힌트이지 최종 Private 개선 보장이 아니다.

## 1차 출처

- Cawley, G. C., & Talbot, N. L. C. (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR 11:2079-2107. https://www.jmlr.org/papers/v11/cawley10a.html
- Blum, A., & Hardt, M. (2015). *The Ladder: A Reliable Leaderboard for Machine Learning Competitions*. PMLR 37:1006-1014. https://proceedings.mlr.press/v37/blum15.html
- Bergstra, J., & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. JMLR 13:281-305. https://jmlr.org/papers/v13/bergstra12a.html
- Bouthillier, X. et al. (2021). *Accounting for Variance in Machine Learning Benchmarks*. MLSys 2021. https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf
