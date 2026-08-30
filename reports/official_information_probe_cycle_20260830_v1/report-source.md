# 2026-08-30 공식 정보 probe 사이클 결과

## 결론

**오늘 남은 7개 슬롯을 모두 유효한 공식 질문으로 전환했고, P2에서 새 Public 최고를 얻었다.** 최종 리더보드는 5위, 문제별 점수 `28.909341 / 27.935464 / 24.203599`, 총점 `81.048404`다. 직전 총점 `81.048217`보다 `+0.000187`점 상승했다.

점수 상승은 작지만 정보 성과는 명확하다.

- **P1:** e150의 Public 상승은 G-ORS에서 발생했다. S-ORS 추가행은 G 존재 여부와 무관하게 표시상 효과가 0이었다.
- **P2:** rank-1 계절 보정의 이득은 bin17이 만들었고 bin18은 미세하게 해쳤다. bin17-only가 RMSE `0.430194`로 새 공식 최고다.
- **P3:** uniform KMA 보정은 G/I/S 세 station 모두 이득이었다. MSE 개선 기여는 표시 정밀도 기준 G `12.75%`, I `42.75%`, S `44.50%`다. 로컬에서 S 보정이 거의 불필요하다는 신호는 공식 분포에서 뒤집혔다.

오늘의 답안 업로드 슬롯은 P1/P2/P3 모두 `0/3`이다. 실제 모델 최종 제출은 누르지 않았으며, 공식 공지상 `2026-09-07`까지 재현 가능한 코드·가중치를 별도로 제출해야 한다.

## 공식 결과

### P1 — e150 station ablation

| 고정 후보 | SHA-256 | 변경 | Public F1 | 점수 | champion 대비 |
|---|---|---:|---:|---:|---:|
| G-ORS 제거 | `59d862b62b24a03c637f58a676d9dc99186f148a6f710f9d0cd3f162f6ca2ce8` | 15행 | `0.829029` | `28.789240` | `-0.004519 F1`, `-0.120101점` |
| S-ORS 제거 | `9f5e8ca7a19d17b317e969fb152f2cc77f50fb5d1cc53757de200b68be720983` | 238행 | `0.833548` | `28.909341` | 표시상 동일 |
| G+S 제거, I-only | `e6a54b0aaf6bf6227362a2706447e4c09bfe80c8510248c2206f204f1ad5c042` | 253행 | `0.829029` | `28.789240` | `-0.004519 F1`, `-0.120101점` |

표시된 F1에서의 고정 finite difference는 다음과 같다.

- G 기여, S 존재: `0.833548 - 0.829029 = +0.004519`.
- G 기여, S 부재: `0.833548 - 0.829029 = +0.004519`.
- S 기여, G 존재/부재: 모두 `0.000000`.
- G×S difference-in-differences: `0.000000`.

이는 숨은 정답을 읽은 분석이 아니라 세 사전 동결 제출의 표시된 Public aggregate 비교다. 플랫폼이 6자리만 표시하므로 S 효과는 정확한 0이 아니라 `표시 정밀도에서 0`이라고만 결론낸다. 현재 공식 champion 파일은 그대로 유지한다.

### P2 — rank-1 season-bin decomposition

| 고정 후보 | SHA-256 | 변경 | Public RMSE | 점수 | 기존 champion 대비 |
|---|---|---:|---:|---:|---:|
| bin17-only | `99c6925cec605905c80f2924c5655b3dd83ed712c9f27853c58d6d9e0f74e2e2` | 762행 | **`0.430194`** | **`27.935464`** | `-0.000015°C`, `+0.000187점` |
| bin18-only | `0d213e97b9435862bbc892ac358afdc99b0a8834740915720f96bd420761d557` | 2,464행 | `0.431267` | `27.922001` | `+0.001058°C`, `-0.013276점` |

기준 alpha50는 `0.431252`, 기존 bin17+18 champion은 `0.430209`다. 표시 RMSE를 제곱한 공간에서:

- 전체 MSE 감소: `0.000898503823`.
- alpha50 대비 bin17 직접 관측 MSE 증분: `-0.000911409868`(감소), 전체 개선의 약 `101.44%`.
- alpha50 대비 bin18 직접 관측 MSE 증분: `+0.000012937785`(증가), bin17 이득의 약 `1.42%`를 잠식.
- 두 직접 증분의 합은 `-0.000898472083`이고, 관측 champion 증분 `-0.000898503823`과의 항등식 residual은 `-3.174e-8`이다. 이는 6자리 RMSE 반올림 envelope 안이다.

따라서 새 공식 후보는 **bin17-only**다. 다만 개선폭이 표시 RMSE `0.000015°C`이므로 Private 안정성이나 통계적 유의성을 주장하지 않는다. 최종 모델 패키지는 이 exact composition을 재현해야 한다.

### P3 — KMA station ablation

| 고정 후보 | SHA-256 | 변경 | Public RMSE | 점수 | champion 대비 |
|---|---|---:|---:|---:|---:|
| S-ORS 보정 제거 | `a5a16ba207ed1cccf16383e1de7b932417666917b0eb2b9c54a00fdb7ab67351` | 120행 | `0.579102` | `24.142185` | `+0.003869m`, `-0.061414점` |
| I-ORS 보정 제거 | `868d18d7a2d62d49b6d97712e686db7a55bbb16e38cea2659584a0c397275f4f` | 140행 | `0.578951` | `24.144591` | `+0.003718m`, `-0.059008점` |

alpha=0 base `0.583892`, uniform alpha=.425 champion `0.575233`, 두 ablation score를 RMSE 제곱 공간에서 결합하면:

| station | MSE 감소 기여 | 전체 개선 비율 |
|---|---:|---:|
| G-ORS | `0.001279485148` | `12.75%` |
| I-ORS | `0.004291256112` | `42.75%` |
| S-ORS | `0.004466122115` | `44.50%` |

세 station 모두 alpha=.425 보정에 양의 공식 기여가 있다. 특히 S-ORS가 가장 큰 기여를 보여, 로컬 station×lead 적합에서 S가 near-zero였던 결과를 공식 배치로 운송하면 안 된다는 강한 반례가 됐다. 현재 uniform alpha=.425 champion을 유지한다.

네 RMSE의 6자리 표시값을 각각 `±0.0000005` 범위로 두고 16개 모서리를 전수 계산해도 세 기여는 모두 양수이며 `S > I > G` 순서가 유지됐다. 이 강건성은 표시 반올림에만 관한 것이며 Private 운송을 보장하지 않는다.

## 다음 공식 probe의 가치가 높은 순서

### P1

S-ORS e150을 더 탐색할 이유는 낮다. 다음 날 3개 슬롯은 `I 제거`, `G-only`, `anchor+GI2 only`를 결과 전에 함께 동결하면 G/I의 조건부 기여와 interaction을 완성할 수 있다. 오늘 결과에 맞춘 threshold 재탐색은 하지 않는다.

### P2

bin18은 제거하고 bin17-only를 기준 후보로 둔다. 다음 질문은 bin17 correction 강도의 공식 이차 곡선이다. alpha=0과 1의 점만 있으므로, 서로 다른 세 강도를 결과 전에 공동 동결해 vertex를 추정해야 한다. 오늘 점수를 보고 한 건씩 적응적으로 정하면 안 된다.

### P3

station별 alpha=.425의 부호는 확인됐지만 최적 강도는 확인되지 않았다. 다음 날 3개 슬롯은 G/I/S 각각 하나의 추가 고정 강도점을 동시에 준비하면 station별 이차곡선의 세 번째 점을 얻는다. 이때도 오늘 결과를 본 뒤 제출 사이에 강도를 바꾸지 않는다.

## 해석 경계

1. Public aggregate는 Private 성능을 보장하지 않는다.
2. 플랫폼 표시가 6자리이므로 작은 차이와 분해 residual에는 반올림 오차가 있다.
3. 반복 Public query는 독립 holdout을 늘리지 않는다. 이번 7개는 모두 첫 점수 전에 동결해 결과 적응을 차단했다.
4. 오늘의 공식 정보는 다음 가설의 범위를 좁히지만, hidden truth나 row-level Public loss를 제공하지 않는다.
5. `모델 최종 제출하기`는 누르지 않았다. 향후 선택한 exact 답안을 재현하는 code/weights 패키지 QA가 별도로 필요하다.

## 재현·감사 근거

- 사전 결정: `reports/official_information_probe_cycle_20260830_v1/pre-submit-decision.md`
- 사전 독립 QA: `reports/official_information_probe_cycle_20260830_v1/pre-submit-independent-qa.json`
- P1 manifest/QA: `C:/Users/cedis/Downloads/해양 해커톤 제출용/20260830_P1_STATION_ABLATION_INFORMATION_PROBES_READY_V1/SET_MANIFEST.json`, `INDEPENDENT_QA.json`
- P2 manifest/QA: `reports/p2_rank1_bin_decomposition_probes_20260830_v1/prepared-probes.json`, `independent-qa.json`
- P3 manifest/QA: `submissions/p3_20260830_station_ablation_probes_v1/FREEZE_MANIFEST.json`, `QA_RECEIPT.json`
- P1 공식 결과 receipt: `reports/official_information_probe_cycle_20260830_v1/p1-official-result.json`
- P2 공식 결과 receipt: `reports/official_information_probe_cycle_20260830_v1/p2-official-result.json`
- P3 공식 결과 receipt: `reports/official_information_probe_cycle_20260830_v1/p3-official-result.json`
- 사이클 종합 독립 QA: `reports/official_information_probe_cycle_20260830_v1/independent-qa.json`
- 공식 제출 이력: [제출 관리](https://oceanaidata.org/app/submissions), 2026-08-30 22:56 KST 확인
- 공식 최종 표: [리더보드](https://oceanaidata.org/app/leaderboard), 2026-08-30 22:57 KST 확인
- 제출 규정: [참가자 전용 제출 안내](https://oceanaidata.org/app/notices), 2026-08-30 확인
