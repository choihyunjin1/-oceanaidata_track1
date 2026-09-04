# p2_gaussian_copula_conditional_mean_20260830_v2

## 결론

**비선형 copula 구조의 신호는 확인됐지만 지금 제출본으로 승격하지 않는다.** 사전등록 핵심 4개 기준은 모두 통과했다. 그러나 별도로 고정한 더 엄격한 fold/inner 안정성 기준이 깨져 최종 분류는 `PRIMARY_SIGNAL_PASS_STRICT_STABILITY_NO_GO_RESEARCH_ONLY`다.

- pooled ΔRMSE: `-0.010616065°C`
- paired KST-day bootstrap CI90: `[-0.017384397, -0.007700262]°C`
- 개선 fold: `2/3`
- target layer: `3/3` 개선
- strict failure: `2025_nov_dec +0.034266729°C`, 세 outer selection 모두 inner worst-group eligibility 실패
- conceptual copula fits: `30` (`27 inner + 3 outer`)
- runtime: `27.943s`
- official hidden/test/sample/submission 접근: `0행`; CSV/upload: `0/0`

따라서 **copula family 전체를 폐쇄하지 않는다.** 이번 exact recipe와 동일한 세 shrinkage·동일 split·동일 unconditional seasonal fallback의 재실행만 닫는다. 다음 연구는 `2025_nov_dec`의 regime 이동을 training-only 정보로 사전 식별할 수 있는지에 집중해야 하며, 결과를 보고 shrinkage만 다시 고르는 방식은 금지한다.

## v1 기술 실패와 v2 수리

v1은 첫 fold 점수 계산 전 `KeyError`로 종료됐다. `2024_sep_oct` 8,779개 시각 중 64개가 target layer 두 개만 가진다는 사실을 row mapper가 처리하지 못했다. prediction 파일 0개, 성능 metric 0개였으므로 `INVALID_TERMINAL_TECHNICAL_FAILURE`이며 과학적 NO_GO가 아니다.

v2는 새 ID로 다음 한 가지만 수리했다.

- 완전 3-layer profile: v1과 같은 copula 예측
- 불완전 profile: exact zero residual correction
- model hyperparameter, empirical marginal, Kendall covariance, shrinkage 후보, outer fold, gate: 변경 없음

독립 QA에서 incomplete profile은 각각 `64/77/9`개 시각이었고 maximum absolute correction은 모두 정확히 `0.0`이었다.

## 모델과 선택 계약

- conditioner: layer 2/3/4의 `current_blend50` 3축과 `alpha50 OAS - current_blend50` 3축
- response: layer 2/3/4의 `truth - alpha50 reference`
- marginal: training-only empirical CDF, meteorological-season model + global fallback
- dependence: Kendall tau를 `sin(πτ/2)`로 변환한 latent Gaussian correlation
- covariance: fixed shrinkage `[0.1, 0.3, 0.5]`, PSD projection, eigen/condition guard
- original-scale conditional mean: 15-point Gauss-Hermite quadrature
- selection: training-only chronological 3-way rotations
- outer truth: prediction NPZ와 commitment를 봉인한 뒤 결합

독립 QA가 재검산한 121개 model receipt의 minimum eigenvalue는 `0.088406933`, maximum condition number는 `63.378424`였다.

## Outer 결과

| fold | selected shrinkage | inner eligible | reference RMSE | candidate RMSE | ΔRMSE |
|---|---:|---:|---:|---:|---:|
| `2024_sep_oct` | 0.1 | false | 4.882261511 | 4.865779278 | -0.016482232 |
| `2025_jul_aug` | 0.3 | false | 1.185757449 | 1.174775890 | -0.010981559 |
| `2025_nov_dec` | 0.5 | false | 0.280732185 | 0.314998914 | +0.034266729 |

## Layer 결과

| layer | reference RMSE | candidate RMSE | ΔRMSE |
|---:|---:|---:|---:|
| 2 | 0.977824584 | 0.948033167 | -0.029791417 |
| 3 | 1.614595346 | 1.598408997 | -0.016186349 |
| 4 | 5.001512730 | 4.992822997 | -0.008689734 |

## Gate 해석

사전등록 핵심 승격 기준은 모두 통과했다.

- pooled ΔRMSE < 0: pass
- 2/3 fold 이상 개선: pass
- 어떤 target layer도 0.001°C 초과 악화하지 않음: pass
- paired bootstrap upper < 0: pass

실행 전에 더 보수적으로 넣은 안정성 guard는 실패했다.

- all inner selections eligible: fail
- no outer fold regresses: fail

이 구분 때문에 결과를 단순한 “가치 없음”으로 버리지 않되, 현재 산출물을 제출 후보라고 부르지도 않는다.

## 재현·해시

- base config SHA256: `cdbf74f9f9f67c17f585905b2c28ed44a5dea77dc9b4abd7f42cce52827b405d`
- v2 overlay config SHA256: `ff3fd20ac95fa571c15e4ed842ae2c54c704e2914405677ab572e1beb167ef06`
- prediction commitment SHA256: `a21a267c267d4aecc33be52c1c442e89d6c8fe792665bda6a8b75ad8abc0720a`
- result SHA256: `79894391b72ccf2d4aab65eeef5361b2658505dd0fba7adffa1fcaf3f6103de3`
- independent QA: `PASS`

모든 outer block은 과거 연구에 이미 노출됐으므로, 이 결과는 공식 일반화 증거가 아니라 다음 구조를 정하는 `RESEARCH_ONLY` 증거다.
