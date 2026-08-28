# P1·P2·P3 신규 구조 실행 결론 — 2026-08-28

## 결론

이번에 사전등록한 세 신규 구조는 모두 로컬 승격에 실패했다. 공식 입력, 제출 CSV, 업로드는 사용하지 않았다. 결과를 본 뒤 같은 구조의 window, posterior guard, epoch, threshold를 바꾸는 재실행도 하지 않는다.

| 문제 | 구조 | 로컬 판정 | 핵심 근거 | 제출 후보 |
|---|---|---|---|---|
| P1 | TS2Vec-style hierarchical contrastive + conditional normal prototype | `NO_GO_COVERAGE_NO_RERUN` | historical embedding coverage `0.908087 < 0.95`; label·Q2 개방 전 종료 | 없음 |
| P2 | BayOTIDE-style fixed dynamic low-rank SSM | `FAIL_GATE_STOP_NO_CSV_NO_RERUN` | posterior SD guard 때문에 active `0/69,850`; RMSE `0.7683674566 → 0.7683674566` | 없음 |
| P3 | TimeXer-style past-exogenous direct 6-lead | `TERMINAL_LOCAL_NO_GO` | incumbent `0.7799487225m`, candidate `0.8784363461m`, delta `+0.0984876237m`; 개선 fold `0/3` | 없음 |

## P1

- label-free contrastive 학습은 1회 정상 완료했고 best checkpoint는 사전등록 규칙에 따른 epoch 30이다.
- 유효 연속구간을 넘지 않는 embedding coverage가 90.81%여서 95% 하한을 통과하지 못했다.
- Q2 truth, Q3·Q4, 공식 데이터는 0행 열람했고 frozen Q2 anchor도 열지 않았다.
- 이 결과는 TS2Vec 표현의 F1 성능을 부정한 것이 아니라, 현재 512행 window 계약이 요구한 coverage를 충족하지 못했다는 terminal preflight 결과다.

## P2

- 3개 Matérn-3/2 trend factor와 12.42h·24h periodic factor를 고정하고 3개 historical block을 한 번 평가했다.
- posterior SD 중앙값 `2.82956°C`, p95 `3.76282°C`로 사전등록 absolute cap `1.0°C`를 넘었다.
- 모든 행이 bit-exact incumbent fallback이어서 delta와 bootstrap CI90이 모두 0이다.
- 불확실성 cap을 결과에 맞춰 늘리는 재실행은 금지한다.

## P3

- 3 outer fold × 3 seed, inner-best checkpoint를 prediction seal 전에 고정했다.
- pooled delta는 `+0.09849m`, bootstrap CI90은 `[+0.06890,+0.12846]m`, 개선 확률은 0이었다.
- 모든 fold, 모든 lead, 모든 station에서 incumbent보다 악화했다. 최악 station 회귀는 `+0.15355m`였다.
- seed RMSE spread는 약 `0.00295m`라 실패가 특정 seed 하나의 우연으로 보이지 않는다.

## 검증

- 신규·공통 회귀 테스트: `25 passed`
- Ruff: PASS
- P1·P2·P3 독립 QA: PASS
- 공식 입력 접근: 0
- 제출 CSV 생성: 0
- 업로드: 0
- Git commit/push: 0

## 다음 연구선

1. P1은 모델 retuning이 아니라 segment 경계에서도 정당한 row-level coverage를 만드는 별도 representation 계약이 필요하다.
2. P2는 현 full-profile dynamic replacement 계열을 닫고, 기존 OAS40 공식 probe와 별도로 검증한다.
3. P3는 Transformer direct 계열을 닫고, 다음 독립 구조로 `Hs²` energy source/sink trajectory residual을 검토한다.

세 항목은 이번 결과에 맞춰 즉시 재실행하지 않으며 새 사전등록이 필요하다.
