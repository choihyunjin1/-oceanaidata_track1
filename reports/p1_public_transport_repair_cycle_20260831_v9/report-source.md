# P1 Public transport repair v9 — 결론 우선 보고서

## 결론

**제출 후보는 0개다.** v9의 세 계층형 CAPA benefit selector는 모두 사전등록된 조건에서 `NO_GO_BUDGET_SUPPORT`로 끝났다. 6/6 prequential selector fits를 수행했지만 Q3·Q4에 단 한 행도 추가하지 않았고, 내부 F1 변화와 raw 예상 점수 변화는 모두 0이었다. family-aware Public 수송 패널티 `0.3219056897594759`점을 적용한 보수 예상 점수 변화는 세 후보 모두 `-0.3219056897594759`점이다. 공식 covariate/hidden truth/CSV/upload 접근은 모두 0이다.

가장 직접적인 원인은 **Q2의 I-ORS×L1 CAPA 학습 proposal 7개가 anchor-negative 행에서 TP 0, FP 2,046**이었다는 점이다. Flat Beta(1,1)의 90% precision lower bound는 `0.0000514694`로, Q3에 동결 적용해야 하는 `Q2 anchor F1 / 2 = 0.389206788`에 크게 못 미쳤다. 계층형 model×duration 및 model×month×duration shrinkage도 이 부재한 이득 신호를 복구하지 못했다. 이는 허용치를 임의로 낮춰 해결할 문제가 아니라, 후보 proposal family가 시간 수송 가능한 정밀도 신호를 갖지 못했다는 반증이다.

## 실행 계약과 결과

| 항목 | 봉인 값 / 실측 |
|---|---|
| 실험 | `p1_public_transport_repair_cycle_20260831_v9` |
| 시간 검증 | Q2 → Q3, Q2+Q3 → Q4 |
| 대상 | I-ORS, layer 1, target-free frozen CAPA whole proposals |
| 후보 | flat Beta(1,1), model×duration strength 20, model×month×duration strength 50 |
| 선택 규칙 | posterior precision LCB90 > train anchor F1/2 |
| 예산 | proposal 전체 수용만 허용, KST day별 full-surface 행의 0.5% 이내 |
| family/tier | `P1_CAPA_HIERARCHICAL_BENEFIT_SELECTOR` / `HARD_CONDITIONAL_ROUTER` |
| Public 수송 보정 | penalty `0.3219056897594759`, raw gate `0.33190568975947593`, calibrated gate `>=0.01` |
| fits | 6/6 |
| 후보별 additions | 0 / 0 / 0 |
| 후보별 pooled ΔF1 | 0 / 0 / 0 |
| 후보별 raw 예상 점수 Δ | 0 / 0 / 0 |
| 후보별 calibrated 예상 점수 Δ | -0.321905690 / -0.321905690 / -0.321905690 |
| strict PASS / CSV | 0 / 0 |
| 실행 시간 | 13.354초 |
| 독립 QA | PASS |

## v5–v9 실패가 알려 준 것

1. **v5 G-ORS×L1 Logit/HGB**: 12 historical fits, 세 후보 모두 threshold를 통과한 addition 0. 기술적 JSON bool 오류는 결과 재실행 없이 복구했으며 과학적 결론은 PASS 0이다.
2. **v6 marginal-F1 selector**: 외부 평가 전, Q2 inner layer-1 anchor-negative calibration label이 한 클래스(전부 0)여서 fit 0 technical failure. 허용치 문제가 아니라 식별 가능한 양성 support가 없었다.
3. **v7 all-layer calibration → layer-2 deployment**: 24 fits. inner에는 두 클래스가 있었지만 isotonic score가 F1/2를 넘지 못해 addition 0.
4. **v8/v8r1 direct rank tail**: 최초 3 fits 뒤 missing import로 외부 metric 전에 종료; import 1줄만 고친 recovery는 사전 py_compile/Ruff/pytest 후 6 fits 완료. 고정 top fractions의 inner day-block LCB가 모두 음수라 세 후보 모두 abstain했다.
5. **v9 CAPA proposal benefit selector**: 모델 score calibration과 다른 proposal-level 베타-이항 계층화를 시험했지만, Q2의 TP support가 0이어서 다시 abstain했다.

따라서 “고정 허용치가 너무 엄격해서 좋은 후보가 탈락했다”는 설명만으로 v5–v9를 뒤집을 수 없다. v7/v8은 inner LCB가 지지하지 않았고, v9는 가장 이른 학습 블록 자체에서 TP가 0이었다. gate를 낮추면 empirical evidence가 개선되는 것이 아니라 검증되지 않은 false-positive 개입을 허용하게 된다.

## 방법론 근거와 한계

- 시간 순서를 보존한 forward split은 미래 정보를 학습에 섞지 않기 위한 핵심 계약이다. scikit-learn `TimeSeriesSplit` 공식 문서도 시간 순서 자료에서 통상적인 교차검증이 미래 학습/과거 평가를 만들 수 있음을 명시한다.
- F1을 최대화하는 calibrated-score cutoff가 현재 최적 F1의 절반과 연결된다는 결과는 inner-only `F1/2` marginal gate의 이론적 근거다. 다만 이 결과는 calibrated probabilities와 iid에 가까운 가정을 사용하므로, 여기서는 posterior lower bound와 chronological outer blocks를 추가했다.
- day-block bootstrap은 행 단위 iid resampling보다 시간 의존성을 덜 훼손하지만, 2개 outer quarter만으로 Public 분포 수송을 완전히 추정할 수는 없다.
- family-aware transport penalty는 관측된 공식 잔차의 경험적 guardrail이지 통계적 confidence interval이 아니다. 신규 hard conditional router가 기존 저복잡도 add-only family의 작은 패널티를 빌려 쓰지 못하게 하는 보수 장치다.

## 다음 판단

v9를 제출하거나 예산/LCB를 사후 완화하지 않는다. 다음 P1 실험은 같은 CAPA family의 threshold 조정이 아니라, **과거 Q2에서 실제 TP support를 먼저 보여 주는 구조적으로 다른 proposal source**가 필요하다. 실행 전 최소 조건은 (a) train-prefix proposal TP>0, (b) Q3/Q4 각 nonnegative, (c) pooled block-LCB `>=0.0005788103`, (d) family-aware raw point gate 충족이다. 이 네 조건을 사전검사하지 못하는 아이디어는 expensive fit으로 승격하지 않는다.

## 재현 명령

```powershell
.venv-p1\Scripts\python.exe scripts\run_p1_public_transport_repair_cycle_20260831_v9.py --validate-only
.venv-p1\Scripts\python.exe scripts\run_p1_public_transport_repair_cycle_20260831_v9.py --execute
.venv-p1\Scripts\python.exe -m py_compile scripts\run_p1_public_transport_repair_cycle_20260831_v9.py
.venv-p1\Scripts\python.exe -m ruff check scripts\run_p1_public_transport_repair_cycle_20260831_v9.py tests\test_run_p1_public_transport_repair_cycle_20260831_v9.py
.venv-p1\Scripts\python.exe -m pytest tests\test_run_p1_public_transport_repair_cycle_20260831_v9.py -q
```

`--execute`는 exactly-once 경로이므로 기존 artifact가 있는 상태에서 재실행하지 않는다.
