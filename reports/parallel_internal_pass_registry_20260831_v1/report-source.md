# P1/P2/P3 병렬 내부통과 제출본 레지스트리

## 결론

목표 하한을 충족했다. P1은 엄격 내부 PASS 1개, P2는 엄격 내부 PASS 2개, P3는 경쟁 기대값 PASS 1개를 보유한다. 총 4개 제출 CSV가 원본 test/test_index와 독립 구조 QA를 통과했으며 업로드는 하지 않았다.

P3는 과학적으로 확정됐다고 부르지 않는다. episode bootstrap CI90이 0을 조금 넘으므로 `SCIENTIFIC_INCONCLUSIVE`이고, 만료성 제출 기회의 기대값·정보가치 정책에서만 `COMPETITION_EXPECTED_VALUE_PASS`다.

## 후보와 예상 공식 점수

| 문제 | 후보 | 내부 변화 | 판정 | 예상 점수 변화 보수/중앙/낙관 |
|---|---|---:|---|---:|
| P1 | HistGBDT OOF add-only | F1 `+0.001381` | strict PASS | `-0.0067 / +0.0054 / +0.0113` |
| P2 | shallow profile residual | RMSE `-0.010243°C` | strict PASS | `+0.0045 / +0.0179 / +0.0225` |
| P2 | HGB absolute profile | RMSE `-0.015491°C` | strict PASS | `+0.0068 / +0.0271 / +0.0398` |
| P3 | ExtraTrees physical router | RMSE `-0.004559m` | competition PASS, scientific inconclusive | `-0.0126 / +0.0723 / +0.1575` |

예상 점수는 공식식이 아니라 우리 공식 제출 이력에서 얻은 경험적 환산이다. P1은 실제 변경이 네 행뿐이므로 내부 F1을 그대로 환산하지 않고 네 행의 가능한 TP/FP 시나리오로 제한했다. P2는 pooled 상대개선과 공식 유사 Sep-Oct 상대개선을 사용했다. P3는 historical delta 및 bootstrap 구간을 1:1 수송한 조건부 시나리오다. 과거 P3에는 부호 역전이 있었으므로 중앙 `+0.0723점`은 보장이 아니다.

현재 문제별 최고점 합계 `81.048404`를 기준으로 best-of-problem 포트폴리오 시나리오는 보수 `81.055179`, 중앙 `81.153235`, 낙관 `81.257023`이다. 낮은 후보 점수는 기존 최고를 대체하지 않으므로 총점을 내리지는 않지만 제출 슬롯을 소비한다.

## 병렬 학습 기록

- 직전 v3 전체 사이클: 49 fits
- 이번 후속 병렬 사이클: P1 4, P2 15, P3 v4 21, v5 21, v6 72, v6b 252, v7 full-fit 1 = 386 fits
- 누적: 435 fits
- 공식 제출 업로드: 0

P1 신규 boundary RF/segment ExtraTrees는 악화 또는 동률로 탈락했다. P2는 HGB가 새 PASS가 됐고 PLS2는 1/3 fold만 개선해 탈락했다. P3는 연속 alpha, physical routers, conformal LCB, uncertainty regression을 순차 검증했으며, strict scientific PASS는 없었다. 그중 v5 ExtraTrees가 pooled 개선, 5/6 block 개선, `P(improve)=0.9156`으로 경쟁 제출 가치가 가장 높아 exact frozen full-fit 1회로 최종 후보가 됐다.

## QA

네 CSV 모두 행 수, 열, 복합키 순서, 중복 0, finite, 저장된 SHA-256을 원본 official input key와 독립 재대조했다. P1 binary 및 P3 `[0,30]m` 범위도 통과했다. hidden truth read와 upload는 0이다.
