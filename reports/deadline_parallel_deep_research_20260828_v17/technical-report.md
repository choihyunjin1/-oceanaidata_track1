# 마감 직전 P1·P2·P3 병렬 Deep Research 결론

## 결론

오늘 남은 제출 기회를 쓸 새 후보는 없습니다. 세 문제에 서로 다른 구조를 사전등록하고 한 번씩 실행했지만 P1은 pooling 가정, P2는 필수 효익, P3는 source 운반성 gate를 통과하지 못했습니다. 제출 횟수를 무리하게 소모하지 않았습니다.

현재 공식 현황은 분당독고다이 5위, `P1 28.901363 / P2 27.922187 / P3 24.066168`, 총 `80.889718`입니다. 확인 당시 오늘 남은 제출은 P1 3회, P2 1회, P3 2회였습니다.

| 문제 | 단발 후보 | 핵심 결과 | 제출 판단 |
|---|---|---|---|
| P1 | station-pooled hierarchical residual scan | 정상 block은 확보했지만 상위 tail의 단일-layer 비중 `45.5~52.9% > 40%` | 제출 금지 |
| P2 | three-way cross-fit regime veto | pooled `-0.002454°C`; Nov-Dec 회귀를 exact `0`으로 차단 | 보류 |
| P3 | Hs·period·Hmax joint MultiRMSE transfer | pooled `-0.000007m`, CI90가 0 포함; 2022·I/S·24h 악화 | 제출 금지 |

## 오늘 확보한 가장 중요한 발판

P2의 계절 운반 실패를 training-only 정보로 사전에 차단할 수 있다는 것을 처음으로 확인했습니다. 이전 correction은 Nov-Dec에서 `+0.008592°C` 악화했지만 새 veto는 해당 fold를 bit-exact no-op으로 만들었습니다. 동시에 Jul-Aug는 `-0.006201°C`, Sep-Oct는 `-0.002596°C` 개선했습니다.

다만 veto가 좋은 구간도 너무 많이 제거해 pooled 개선이 직전 `-0.004799°C`에서 `-0.002454°C`로 줄었습니다. 다음 실험의 질문은 더 이상 “회귀를 막을 수 있는가”가 아니라 “Nov-Dec no-op을 유지하면서 Sep-Oct correction coverage만 정직하게 되살릴 수 있는가”입니다.

## 문제별 판단

### P1

부분 pooling은 cell별 표본 부족을 완화했습니다. S-ORS Q2/Q3/Q4 정상 calibration cell-block이 `107/126/161`로 최소 100을 충족했고 7개 layer도 확보했습니다. 하지만 tail score가 특정 layer에 집중되어 station 공통 calibration의 교환가능성 가정이 깨졌습니다. outer truth는 열지 않았고 e150 anchor는 한 건도 삭제하지 않았습니다.

다음 P1은 station 전체 pooling이 아니라 layer family 또는 depth regime별 pooling이어야 합니다. 같은 `40%` gate 완화는 결과 기반 튜닝이므로 하지 않습니다.

### P2

세 연속 시간군을 held/correction-fit/reference-support 역할로 회전해 prediction과 reference를 분리했습니다. 결과는 pooled `-0.002453834°C`, bootstrap CI90 `[-0.004330819,-0.001569947]`, layer L2/L3/L4 모두 개선입니다. 그러나 사전 기준 pooled `≤-0.005°C`와 Sep-Oct `≤-0.003°C`를 통과하지 못했습니다.

공식 α50의 exact historical OOF가 없어서 이 수치는 계속 `INCUMBENT_PROXY_VALIDATION`입니다. 마지막 P2 슬롯을 쓰기에는 이득 크기와 comparator 신뢰도가 부족합니다.

### P3

ERA5 manifest에 실제 directional spread·spectral partition·Stokes drift·energy period가 없어, 존재하지 않는 독립 변수를 가장하지 않고 Hs·mean period·Hmax의 공동 상태 예측을 검증했습니다. 공동학습은 이론적으로 관련 task의 inductive bias를 제공할 수 있지만 이번 source에서는 거의 동률이었습니다.

source ΔRMSE는 `-0.00000709m`, CI90 `[-0.00042163,+0.00039804]`였고 2022년 `+0.00042534m`, I/S station과 24h가 악화했습니다. strict gate에 따라 fresh shadow truth는 열지 않았습니다.

## QA

- 문제별 one-shot 및 no-retry 계약 준수.
- 신규 cycle test와 Ruff PASS.
- 통합 독립 QA 18개 검사 PASS.
- P1/P3 result와 seal/model manifest, P2 prediction commitment와 independent QA 해시 확인.
- 공식 test/sample/submission 접근, CSV 생성, 업로드는 모두 0건.

## 다음 실행 우선순위

1. P2에서 Nov-Dec exact no-op은 고정하고, three-way OOF benefit을 연속 hard veto가 아닌 cross-fitted shrinkage 또는 monotone confidence weighting으로 변환합니다. 새 outer surface 없이는 다시 제출하지 않습니다.
2. P1은 layer를 수온약층 상·하부 또는 안정적 residual family로 사전 그룹화한 뒤 group-conditional calibration을 검증합니다.
3. P3는 현재 joint-state family를 닫습니다. 실제 spectral partition/source 변수가 추가되지 않으면 기존 champion의 long-lead 구조를 유지합니다.

## 주요 근거

- Chernozhukov, Wüthrich, Zhu, 2018, [Dependent-data conformal inference](https://proceedings.mlr.press/v75/chernozhukov18a.html)
- Martinez Gil et al., 2024, [Group-conditional calibration](https://proceedings.mlr.press/v244/martinez-gil24a.html)
- Chernozhukov et al., 2016, [Double/debiased machine learning and cross-fitting](https://arxiv.org/abs/1608.00060)
- Li et al., 2024, [Regression with rejection under covariate shift](https://proceedings.mlr.press/v238/li24g.html)
- CatBoost, [MultiRMSE documentation](https://catboost.ai/docs/en/concepts/loss-functions-multiregression)
- ECMWF, 2020, [IFS Wave Model](https://www.ecmwf.int/sites/default/files/elibrary/2020/81192-ifs-documentation-cy47r1-part-vii-ecmwf-wave-model_1.pdf)
- Caruana, 1997, [Multitask Learning](https://doi.org/10.1023/A:1007379606734)

## 중단 기준

세 후보가 terminal gate에 도달했고 동일 계약의 추가 실행은 결과 기반 완화가 됩니다. 오늘의 연구 목적은 남은 슬롯을 소모하는 것이 아니라 다음 공식 질의가 무엇을 판별할지 명확하게 만드는 것이므로 여기서 중단합니다.
