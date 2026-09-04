# P1 incumbent-preserving MS-TCN++/ASRF v2 실행 결과

상태: **NO_GO_CONFIRMATORY**  
결론: **Q3·Q4 확인 기준을 통과하지 못했다.** Q2 선택 성능과 무관하게 공식 제출 후보로 승격하지 않는다.

> 이 문서는 로컬 Q2/Q3/Q4 증거만 요약한다. 공식 제출은 생성·승인·업로드되지 않았다. 별도 승인된 공식 평가에서 F1 0.930749 이상을 관측하기 전에는 공식 +3점이 미확정이다.

## 선택 사양과 수렴 해석

- width `512`, batch `64`, epoch `125`, high threshold `0.9`
- seed: `20260827, 20260839, 20260863`; 표현: `raw_three_seed_ensemble_mean`
- 수렴 계약: `Q2_SELECTED_BEFORE_MAX_EPOCH_NO_CONVERGENCE_CLAIM`
- Q2 선택 epoch `125` / 최대 `300`. training loss는 진단용이며 holdout 수렴을 뜻하지 않는다.

## Q2 선택 전용 결과

| 기준 | Router | 후보 | 차이/보조 지표 |
|---|---:|---:|---:|
| F1 | 0.792275574 | 0.890432823 | +0.098157249 |
| 추가 행 precision | — | — | 0.884220 |
| long-event recall gain | — | — | +0.192401 |
| 정상 FP 비율 | — | — | 1.298795× |
| Router 양성 제거 | — | — | 0행 |

Q2 epoch `125`의 ΔF1 `+0.098157249`는 maximum-over-grid에서 선택된 **고립된 낙관적 peak**로 취급한다. 인접 checkpoint 최고 ΔF1 대비 차이는 `+0.019615064`이며, Q2는 사양 선택에만 사용하고 승격 증거에는 포함하지 않는다.

## Q3·Q4 확인 결과

| 구간 | Router F1 | 후보 F1 | ΔF1 |
|---|---:|---:|---:|
| Q3 | 0.894979508 | 0.908278885 | +0.013299377 |
| Q4 | 0.914341895 | 0.882857479 | -0.031484416 |
| Pooled | 0.902917024 | 0.897776915 | -0.005140109 |

추가 `753`행의 precision은 `0.381142`이고 Router 양성 제거는 `0`행이다. 21일 paired circular block bootstrap `10,000`회의 ΔF1 평균은 `-0.005624439`, CI90은 `[-0.028100026, +0.014066700]`다.

## Gate 판정

- 연구 성공: **FAIL**
  - `anchor_positive_removed_rows`: PASS
  - `ci90_lower_positive`: FAIL
  - `pooled_delta_f1`: FAIL
  - `q3_positive`: PASS
  - `q4_positive`: FAIL
- 강한 공식 probe 검토 기준: **FAIL**
  - `added_row_precision`: FAIL
  - `anchor_positive_removed_rows`: PASS
  - `ci90_lower`: FAIL
  - `pooled_delta_f1`: FAIL
  - `pooled_f1_endpoint`: FAIL
  - `q3_delta_f1`: PASS
  - `q4_delta_f1`: FAIL
  - `stations_improved`: FAIL

## Station별 재현성

| Station | Router F1 | 후보 F1 | ΔF1 |
|---|---:|---:|---:|
| G-ORS | 0.800962117 | 0.831092928 | +0.030130811 |
| I-ORS | 0.874433562 | 0.870083771 | -0.004349791 |
| S-ORS | 0.935037461 | 0.923869007 | -0.011168454 |

개선 station은 `1/3`이다.

## Optimizer 이력 진단

| Fit | Epoch | 최소 loss (epoch) | 마지막 loss | tail-25 slope | tail-50 slope | nonfinite |
|---|---:|---:|---:|---:|---:|---:|
| `q2_width_256_seed_20260827` | 300 | 0.00709842 (294) | 0.00712159 | -2.107e-06 | -3.972e-06 | 0 |
| `q2_width_256_seed_20260839` | 300 | 0.00717151 (230) | 0.118689 | -2.519e-04 | -1.024e-03 | 0 |
| `q2_width_256_seed_20260863` | 300 | 0.00701904 (292) | 0.00705322 | -2.847e-06 | -3.613e-06 | 0 |
| `q2_width_512_seed_20260827` | 300 | 0.00562305 (299) | 0.00563453 | -9.740e-07 | -3.981e-06 | 0 |
| `q2_width_512_seed_20260839` | 300 | 0.00672301 (107) | 0.00730617 | -1.204e-05 | -2.119e-06 | 0 |
| `q2_width_512_seed_20260863` | 300 | 0.00712773 (186) | 0.179833 | -3.940e-04 | -2.681e-04 | 0 |
| `q3_width_512_seed_20260827` | 125 | 0.0100076 (68) | 0.0587222 | -1.729e-04 | -3.028e-04 | 0 |
| `q3_width_512_seed_20260839` | 125 | 0.00873336 (122) | 0.0100679 | -7.409e-05 | -1.300e-03 | 0 |
| `q3_width_512_seed_20260863` | 125 | 0.0103727 (78) | 0.115199 | -1.742e-03 | +2.305e-03 | 0 |
| `q4_width_512_seed_20260827` | 125 | 0.00834417 (96) | 0.105123 | +4.945e-02 | +1.700e-02 | 0 |
| `q4_width_512_seed_20260839` | 125 | 0.00951636 (125) | 0.00951636 | -1.045e-03 | +4.192e-05 | 0 |
| `q4_width_512_seed_20260863` | 125 | 0.0107057 (74) | 0.284149 | +8.997e-03 | +3.475e-03 | 0 |

이 표는 optimizer training loss의 안정성·tail 형태만 진단한다. 일반화 성능과 holdout 수렴 판단은 Q3·Q4 metric과 사전 고정 gate를 따른다.

## 해석 한계

- Q2 epoch 125는 인접 checkpoint와 분리된 낙관적 최대점이므로 Q3·Q4 확인 결과와 분리해서 해석한다.
- optimizer training loss는 일반화 성능이나 수렴을 직접 증명하지 않는다.
- 로컬 F1과 공식 F1의 크기 운송은 보장되지 않는다. 공식 F1 0.930749 이상을 별도 승인된 공식 평가에서 관측하기 전에는 공식 +3점을 주장할 수 없다.
- bootstrap은 날짜 표본 변동을 다루지만 Q2 다중선택과 전체 HPO 불확실성을 포함하지 않는다.

## 시각 증거

1. `figures/figure_01_training_loss_convergence.png` — Q2 두 용량 6개 곡선과 Q3·Q4 선택 용량 6개 곡선
2. `figures/figure_02_q2_qualification_envelope.png` — Q2에서 각 epoch·width의 고정 threshold best ΔF1 envelope
3. `figures/figure_03_confirmatory_effects_and_gates.png` — Q3·Q4·pooled 효과, CI90, station별 부호
