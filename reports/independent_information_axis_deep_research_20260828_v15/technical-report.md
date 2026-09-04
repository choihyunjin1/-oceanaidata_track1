# P1·P2·P3 Deep Research 실행 결론

## 결론

이번 사이클에서 새 공식 제출 후보는 만들지 않았습니다. 세 문제 모두 기존 방식과 다른 구조를 실제로 구현하고 1회 검증했지만, 사전 승격 기준을 완전히 통과한 모델은 없었습니다. 공식 데이터 접근·CSV 생성·업로드는 모두 0건입니다.

| 문제 | 관측 결과 | 판단 |
|---|---|---|
| P1 | 다층 RPCA window 0개, ΔF1 0 | 데이터가 layer별로는 충분해도 같은 시각에 겹치지 않아 구조가 성립하지 않음 |
| P2 | pooled ΔRMSE `-0.004799°C`, bootstrap CI90 `[-0.008506,-0.003107]` | 가장 유망하지만 Nov-Dec `+0.008592°C` 회귀로 제출 보류 |
| P3 | shadow 장기예보 ΔRMSE `-0.011197m` | 17 cases뿐이고 S-ORS가 악화해 outer 검증 중단 |

## 가장 중요한 발견

P2는 단순 α 미세조정과 거의 직교하는 correction을 만들면서 2개 fold와 3개 layer를 개선했습니다. 다만 exact 공식 α50 OOF가 없는 proxy 비교이고 한 계절에서 크게 회귀했습니다. 현재 제출하면 평균 개선 신호보다 계절 운반 실패 위험이 큽니다.

P1은 연구 전 데이터 coverage 해석이 잘못됐습니다. layer마다 긴 관측 구간이 있다는 것과 여러 layer가 같은 시각에 존재한다는 것은 다릅니다. 이 때문에 synchronous matrix 방법은 더 진행할 가치가 없습니다.

P3는 G/I station에서 wind-wave memory 신호가 있었지만 S station 운반성이 없고 표본이 너무 작았습니다. 결과를 보고 S만 빼는 식의 사후 튜닝은 하지 않았습니다.

## 다음 연구 우선순위

1. P2에서 correction 크기를 다시 맞추지 말고, train-only support/regime veto가 계절 회귀를 사전에 차단하는지 새 계약으로 검증합니다.
2. P3는 현재 20-feature family를 닫고, 독립 shadow가 충분할 때 pressure tendency·directional spectral memory를 결합한 새 family를 엽니다.
3. P1은 synchronous multi-layer 가정을 버리고 sensor별 event proposal을 asynchronous station state로 결합합니다.

세부 근거·한계·출처는 같은 폴더의 `report-source.md`, artifact 검증은 `independent-qa.json`에 고정했습니다.
