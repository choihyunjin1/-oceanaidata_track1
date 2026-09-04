# 허용치 재설계와 과거 실패 전수 재평가

## 결론

기존의 `P1 +0.003 F1`, `P2 0.005°C`, `P3 0.005m` 같은 고정 raw-unit 문턱은 폐기해야 한다. 세 문제 모두에서 이보다 작은 변화가 실제 Public 점수를 올린 반례가 이미 있다.

| 문제 | 과거 hard 문턱 | 실제 공식 개선 | 실제 점수 증가 | 판정 |
|---|---:|---:|---:|---|
| P1 | `+0.003 F1` | `+0.000300 F1` | `+0.007978점` | old gate false negative |
| P2 | `0.005°C` | RMSE `-0.001002°C` | `+0.012572점` | old gate false negative |
| P3 | `0.005m` | RMSE `-0.002409m` | `+0.038225점` | old gate false negative |

적절한 허용치는 하나의 숫자가 아니다. 다음 네 층으로 분리한다.

1. **유효성:** lineage·키·행·순서·누수·finite/domain·사전등록·무재시도는 계속 hard gate다.
2. **수치 재현:** hash/key는 exact, deterministic metric 재계산은 절대오차 `1e-12`, 6자리 표시상 apparent tie는 `<1e-6`로 다룬다. 이 값들은 효과 크기 문턱이 아니다.
3. **과학적 효과:** 근거 있는 SESOI가 없으면 방향 문턱은 `0`이다. pooled 주지표의 paired dependence-aware 90% 구간이 전부 개선 방향이면 strong challenger, 0을 지나면 inconclusive/exploratory, 전부 손실 방향이면 primary harm다.
4. **제출 행동:** raw metric hard 문턱을 두지 않는다. `개선확률 × 예상 점수 증가 + 정보가치 − 최선 대안 슬롯의 기회비용`으로 판단한다. 하루 3회는 독립 확인 3개가 아니라 같은 Public 표면에 대한 상관된 질의 3개다.

이 재평가는 **48개 과거 family(P1 17, P2 19, P3 12)**와 negative registry의 **35개 canonical closed/invalid/not-failure group(P1 13, P2 13, P3 9)**을 모두 교차했다. 두 원장은 정의 단위가 달라 합산하지 않는다. fit, raw/prediction row read, 공식 test/sample/submission/hidden/query read, CSV, upload는 모두 0이다.

## 최종 허용치 기록

| 층 | 권장값 | 쓰임 |
|---|---|---|
| Hash·key·row identity | exact | 재현·lineage 유효성 |
| Deterministic metric epsilon | `1e-12` absolute | 같은 aggregate 재계산 검증 |
| 6-decimal apparent tie band | `<1e-6` raw metric 또는 point | 표시 반올림 구분 |
| Scientific directional margin | `0` | 근거 있는 비영점 SESOI가 없을 때의 기본값 |
| Scientific uncertainty | paired dependence-preserving 90% CI | 방향성 증거 단계 |
| Equivalence/noninferiority margin | 미설정 | 비용·안전·repeatability로 결과 전에 정할 때만 허용 |
| Submission margin | 미설정 | 점수 EV·VOI로 결정 |

비열등성·등가성 경계는 관측 결과와 독립적으로 정해야 하며, 효과가 유의하지 않다는 사실만으로 등가성을 선언할 수 없다. [Lakens, Scheel & Isager 2018](https://journals.sagepub.com/doi/10.1177/2515245918770963) 선택 기준을 반복 최적화하면 그 기준 자체에 과적합할 수 있으므로, exposed surface의 winner는 untouched confirmation 없이 확정 후보가 아니다. [Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html) 반복 leaderboard query 역시 독립 검증을 추가하지 않는다. [Blum & Hardt 2015](https://proceedings.mlr.press/v37/blum15.html)

### 점수 단위 참고표 — hard gate가 아님

P1·P2는 저장된 공식 제출 이력의 OLS이고 공식 점수식이 아니다. P3는 현재 `RMSE < T_public=0.630065` 구간에서 README anchor가 주는 기울기 `10/T_public ≈ 15.871378점/m`와 저장 OLS가 거의 같다.

| 문제 | 0.01점 | 0.05점 | 0.10점 | 근거 |
|---|---:|---:|---:|---|
| P1 | 약 `0.000376 F1` | `0.001881 F1` | `0.003762 F1` | 경험 OLS, 공식식 아님 |
| P2 | 약 `0.000797°C` | `0.003985°C` | `0.007970°C` | 경험 OLS, 공식식 아님 |
| P3 | 약 `0.000630m` | `0.003150m` | `0.006301m` | Public README의 현재 구간 |

이 표가 보여주는 핵심은 `0.003 F1`, `0.005°C`, `0.005m`이 모두 대략 `0.06–0.08점`급을 요구해, 작지만 실제인 개선을 체계적으로 버렸다는 점이다. 점수 band는 `0–0.01 trace`, `0.01–0.05 small`, `0.05–0.10 material`, `≥0.10 large`로 기술할 뿐 승격 veto로 쓰지 않는다.

## 재개하거나 복원한 후보

### P1

| 후보 | pooled benefit | 불확실성 | 새 판정 |
|---|---:|---:|---|
| block inpaint | `+0.002591 F1` | CI90 `[-0.015669,+0.051219]` | `REOPEN_FROZEN_CONFIRMATION_ONLY` |
| dynamic peer reliability | `+0.004640 F1` | CI90 `[-0.001677,+0.011611]` | `REOPEN_FROZEN_CONFIRMATION_ONLY` |
| environment-balanced replay | `+0.0000426 F1` | 유효 CI 없음, single seed | 저우선 frozen confirmation only |
| segment-precision router core | `+0.000968859 F1` | Q3/Q4 방향 반전, CI 없음 | frozen confirmation only |
| window phase | default e150 대비 `+0.000241782 F1` | Q2 exposed | frozen confirmation only |
| Sobol selected `trial_18`, threshold `0.8` | `+0.000565637 F1` | 3 seed·모든 월 양수, CI 없음 | **frozen Q3/Q4 confirmation only** |

Sobol의 32개 탐색 공간이나 임계값을 다시 여는 것이 아니다. 사전 선택된 `trial_18/0.8`만 독립 표면에 고정 확인할 수 있다. 결과를 보고 더 좋아 보였던 `0.7`로 바꾸는 것은 금지한다.

G-ORS depth invariance는 aggregate `+0.002688`과 G-ORS 손실이 함께 있고 CI가 `[-0.009210,+0.003553]`라 `INCONCLUSIVE_RESEARCH_ONLY`다. 새 add-only LCB도 `-0.002381`, CI `[-0.017810,+0.013951]`로 불확실하다. 반면 fixed Group-DRO `-0.013481`, event-balanced SupCon `-0.164874`는 명확한 exact-recipe 손실로 유지한다.

### P2

| 후보 | RMSE benefit | benefit CI90 | 새 판정 |
|---|---:|---:|---|
| supervised rank-1 base | `+0.004798862°C` | `[+0.003107,+0.008506]` | high-value research challenger; proxy comparator 주의 |
| cross-fit rank-1 v2 | `+0.002453834°C` | `[+0.001570,+0.004331]` | high-value; 이후 공식 `+0.001002°C` 개선으로 action axis 검증 |
| nested PLS | `+0.002041992°C` | `[+0.001052,+0.003874]` | challenger with 729-eval/84-fit selection caveat |
| Gaussian copula v2 | `+0.010616065°C` | `[+0.007700,+0.017384]` | **가장 높은 frozen-probe 우선순위** |
| state-conditioned copula | `+0.003459176°C` | `[+0.001923,+0.006530]` | high-value research challenger |

과거 Nov–Dec, JJA, inner eligibility, fold 수 같은 실패는 transport risk로 남지만 pooled efficacy를 뒤집는 hard veto는 아니다. 다만 이 표면은 여러 family와 243/729-grid 선택에 노출됐으므로 `confirmed`가 아니다. exact grid 재탐색은 계속 닫고 frozen winner 확인만 허용한다.

Availability-aware copula v2는 benefit `-0.001990430°C`, CI `[-0.004967,-0.000662]`라 primary harm으로 유지한다. v1 guard failure는 기술 invalid이고 성능 판정이 아니다.

### P3

Lead-continuous는 active benefit `+0.004187822m`, benefit CI90 `[-0.001585,+0.010129]`이므로 `0.005m 미달 실패`가 아니라 `EXPLORATORY_CHALLENGER_RESEARCH_ONLY`다. 현재 Public 구간의 점수 민감도를 단순 적용하면 약 `0.066점` 규모지만, local→official transport 예측이 아니라 크기 참고일 뿐이다.

Sparse GP abstention은 benefit `-0.003475071m`, CI `[-0.009929,+0.003149]`라 harm 확정이 아니라 inconclusive다. CatBoost repaired confirmation은 `-0.007974131m`와 전부 불리한 CI, masked SSL은 `-0.314155m`와 전부 불리한 CI라 exact-recipe harm을 유지한다.

P3의 가장 중요한 구조적 사실은 local 방향과 Public 방향이 자주 뒤집혔다는 점이다. reverse-global과 ERA5 Hs²처럼 local improvement/harm을 공식 improvement/harm으로 일률 변환할 수 없다. KMA의 exact local cross-fit strategy는 닫아도 KMA 정보축 전체를 닫으면 안 된다.

## 과거 원장 전수 재판정

`failure-replay.json`에는 48개 family와 35개 canonical group이 한 행씩 들어 있다.

- 48개 family 중 기존 anchor/champion/official evidence 9개는 애초 실패가 아니다.
- 29개 exact family는 새 기준에서도 exact 범위 종료가 유지된다. 이는 넓은 모델 class 전체의 불가능을 뜻하지 않는다.
- family-level에서 P1 block/peer 2개는 frozen confirmation으로 복원, P1 depth·P2 density·P3 KMA/analog 4개는 harm이 아니라 inconclusive로 좁혔다. P3 lead-continuous는 exploratory challenger로 복원했다.
- canonical group 35개 중 20개 exact scope는 그대로 닫힌다. 3개 technical invalid는 성능 결론이 없고, 2개는 원래 not-failure다. 나머지는 group 전체가 아니라 표 안의 **정확한 subrecipe만** confirmation 후보로 부분 재개했다.

모든 재개는 같은 exposed surface 재튜닝이 아니라 `exact frozen candidate + 새 독립 표면` 또는 `결과별 행동표가 사전등록된 단일 official probe`만 뜻한다.

## 이상치 허용치

이상치 hard delete는 기본값으로 두지 않는다. 드문 물리값과 검증된 센서 오류는 다르며, 행을 삭제하면 평가 모집단이 바뀐다.

- candidate가 새 finite/domain violation을 만들면 Level-0 실패다.
- comparator에 이미 있던 극값에서 candidate가 exact no-op이면 행을 보존한다. P2의 18개 inherited extreme이 이 사례였다.
- 센서 오류 ground truth가 없으면 flag, robust loss, bounded correction, abstention을 진단 축으로 쓴다.
- hard delete는 독립 검증된 센서 오류 규칙과 동결된 estimand가 있을 때만 별도 사전등록한다.

## 실행 경계와 다음 행동

이번 감사에는 fit, raw row, prediction row, 공식 test/sample/submission/hidden/query row, CSV, upload가 없었다. 기존 aggregate receipt와 README만 사용했다. commit/push도 수행하지 않았다.

우선순위는 다음과 같다.

1. P2 Gaussian copula v2를 exact frozen probe 후보 1순위로 두되, lineage·중복·quota·결과별 행동표를 먼저 고정한다.
2. P1은 Sobol `trial_18/0.8`의 untouched Q3/Q4 confirmation이 가장 정직한 재개다. block/peer/segment/window-phase는 그다음이며 같은 표면 재튜닝은 하지 않는다.
3. P3는 lead-continuous의 fresh episode/block confirmation을 우선한다. 새 독립 표면이 없을 때만, 다음 행동을 바꾸는 단일 official probe로 정보가치를 평가한다.

재현 명령:

```powershell
.\.venv-p1\Scripts\python.exe scripts\audit_tolerance_recalibration_and_failure_replay_20260830_v2.py
.\.venv-p1\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_audit_tolerance_recalibration_and_failure_replay_20260830_v2.py
.\.venv-p1\Scripts\python.exe scripts\qa_tolerance_recalibration_and_failure_replay_20260830_v2.py
```
