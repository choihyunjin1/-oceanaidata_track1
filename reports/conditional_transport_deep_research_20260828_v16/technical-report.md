# P1·P2·P3 조건부 운반성 Deep Research 실행 결론

## 결론

이번 사이클에서 공식 제출할 가치가 있는 새 후보는 없습니다. 세 문제를 병렬로 연구하고 각각 사전 고정된 단발 실험까지 수행했지만, P1·P2는 학습 전에 구조적 지지 조건을 통과하지 못했고 P3는 source 검증에서 기존 모델보다 악화했습니다. 공식 test/sample/submission 접근, CSV 생성, 업로드는 모두 0건입니다.

| 문제 | 새 구조 | 핵심 결과 | 판정 |
|---|---|---|---|
| P1 | 비동기 다층 latent-state GP subset scan | Q2/Q3/Q4 정상 calibration block `19/74/126`; 전체 조건 통과 cell `2/15`, `8/15`, `14/15` | `NO_GO_SUPPORT_EXACT_E150_NO_OP` |
| P2 | α50 supervised rank-1 correction용 train-only regime veto | 첫 outer fold의 두 training block을 LOBO와 자기참조 금지에 동시에 쓰면 reference row가 0 | `NO_GO_IMPLEMENTATION_PREFLIGHT` |
| P3 | Hs² 가중 방향성 에너지 기억 20 features | source RMSE `0.545996 → 0.546183m`, Δ `+0.000188m`; CI90 `[-0.000246,+0.000628]` | `NO_GO_SOURCE_GATE` |

## 문제별 해석

### P1

비동기 시계열을 직접 처리하는 multi-task GP 계열은 불규칙 표본의 공동 잠재 상태를 학습할 수 있다는 연구 근거가 있습니다. 그러나 이번 데이터에서는 모델 문제가 아니라 사전등록된 calibration 지지 조건이 먼저 깨졌습니다. Q2와 Q3은 최소 100개 정상 block을 확보하지 못했고, 양쪽 기간에서 peer coverage·matched row·posterior coverage를 모두 만족한 cell도 부족했습니다. 그래서 GP fit과 outer truth를 열지 않고 기존 e150 anchor를 한 건도 삭제하지 않은 exact no-op으로 봉인했습니다.

이 결과는 “GP가 나쁘다”가 아니라 “현재 분할과 엄격한 cell 단위 보정으로는 정직하게 평가할 표본이 없다”는 결론입니다. 같은 설정의 threshold 완화나 재실행은 금지합니다.

### P2

직전 supervised rank-1 correction은 pooled 개선 신호가 있었지만 Nov-Dec 운반 실패가 있었습니다. 이번에는 결과를 본 뒤 correction 크기를 줄이는 대신, outer-training 내부 증거만으로 적용 여부를 결정하는 regime veto를 설계했습니다.

그런데 첫 outer fold의 training block이 정확히 두 개뿐입니다. 한 block을 inner validation으로 숨기고, 다른 block이 자기 자신의 α50 reference에 label을 제공하지 못하게 하면 완전한 seasonal OAS reference row가 0이 됩니다. 이를 억지로 실행하려면 held label 재사용, 자기 label 재사용, 또는 두-source-block 기준 완화 중 하나가 필요하며 모두 사전 누수 방지 계약을 깨뜨립니다. 따라서 outer truth와 prediction commitment 전에 terminal NO-GO로 닫았습니다.

다음 P2 실험은 적어도 세 개의 독립 prefix block을 제공하거나, label-free reference를 쓰는 별도 구조여야 합니다. 같은 veto의 임계값만 바꾸는 것은 해법이 아닙니다.

### P3

파랑 스펙트럼은 평균 파향 하나만으로 설명되지 않으며, 방향 분포의 폭과 고차 특성이 추가 정보를 가질 수 있습니다. 이를 현재 확보된 Hs·평균파향으로 근사해 6/12/24/48h Hs² 가중 방향 집중도, 상대각, 회전율 10개와 mask 10개를 추가했습니다.

하지만 2014–2020 학습, 2021–2023 held source의 paired 검증에서 세 station이 모두 악화했고 특히 2022년과 18/24h 장기 lead가 나빠졌습니다. pooled ΔRMSE도 양수이며 bootstrap CI가 0을 가로질렀습니다. 계약대로 fresh 2024 shadow의 truth와 outer 181-case truth는 열지 않았습니다. 평균파향 기반 proxy는 실제 spectral spread/partition을 대체하지 못한다는 한계가 확인됐습니다.

## QA와 재현성

- 신규 단위 테스트: 13개 PASS.
- 신규 Python 코드 Ruff: PASS.
- 독립 QA: 18개 계약 검사 전부 PASS.
- P1 sealed prediction, P3 source model·prediction artifact, 모든 result hash를 재계산해 manifest와 일치함을 확인했습니다.
- 세 artifact 폴더에 CSV가 없고 공식 데이터 접근·업로드는 0건입니다.
- 결과 기반 재실행·임계값 변경·station 사후 제외는 하지 않았습니다.

## 다음 우선순위

1. **P2 우선:** 더 많은 prefix block을 만들 수 있는 시간 분할 또는 label-free reference 구조를 먼저 설계합니다. 현재 가장 강했던 correction 자체를 버리기보다, 운반 여부를 정직하게 판단할 수 있는 평가 구조가 필요합니다.
2. **P1:** cell별 100-block 기준을 억지로 낮추지 않습니다. station-level 부분 pooling이나 layer-group calibration처럼 표본을 합치는 새 모델은 가능하지만, 별도 계약과 보수적 uncertainty gate가 필요합니다.
3. **P3:** 현재 mean-direction proxy family는 닫습니다. 실제 directional spread·wave partition 같은 독립 source 변수가 확보될 때만 다시 엽니다.

## 주요 근거

- Futoma et al., 2017, [Learning to Detect Sepsis with a Multitask Gaussian Process RNN Classifier](https://proceedings.mlr.press/v70/futoma17a.html)
- Herlands et al., 2018, [Scalable Gaussian Processes for Characterizing Multidimensional Change Surfaces](https://proceedings.mlr.press/v84/herlands18a.html)
- Chernozhukov et al., 2018, [Exact and Robust Conformal Inference Methods for Predictive Machine Learning](https://proceedings.mlr.press/v75/chernozhukov18a.html)
- Li et al., 2024, [Regression with Rejection under Covariate Shift](https://proceedings.mlr.press/v238/li24g.html)
- ECMWF, 2023, [IFS CY48R1 Part VII: ECMWF Wave Model](https://www.ecmwf.int/en/elibrary/81373-ifs-documentation-cy48r1-part-vii-ecmwf-wave-model)
- ECMWF, 2026, [Wave measures and definitions](https://confluence.ecmwf.int/spaces/FUG/pages/673550584/Section%2B2A.3.1%2BWave%2Bmeasures%2Band%2Bdefinitions)
- Kuik, van Vledder, Holthuijsen, 1988, [A Method for the Routine Analysis of Pitch-and-Roll Buoy Wave Data](https://journals.ametsoc.org/abstract/journals/phoc/18/7/1520-0485_1988_018_1020_amftra_2_0_co_2.xml)

## 중단 기준

각 문제에서 새 구조 한 개를 사전등록하고 정확히 한 번 실행했으며, 모두 terminal gate에 도달했습니다. 같은 데이터와 같은 계약에서 추가 탐색하면 threshold 완화 또는 exposed surface 적응으로 바뀌므로 이 사이클은 여기서 종료합니다.
