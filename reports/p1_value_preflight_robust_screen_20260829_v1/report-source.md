# P1 가치성 사전심사 및 환경강건 저충실도 검증

## 결론

가치 없는 실험을 완벽하게 사전에 알아내는 것은 불가능하지만, **정적 가치성 심사 → 10-epoch 저충실도 심사 → 30-epoch 확인 → 3-seed 본학습**의 네 단계로 대부분의 저가치 후보를 계산 전에 또는 수 분 안에 중단할 수 있다. 이번에 첫 두 단계를 코드로 고정하고 실제 후보 하나에 적용했다.

환경균형 replay 후보는 정적 심사에서 `10/10`, `PASS_TO_LOW_FIDELITY`를 받아 새로운 학습 목적 가설로서 시험할 가치는 있었다. 그러나 10-epoch 단일-seed historical shadow에서 Q3 F1은 `-0.0008115`, Q4는 `+0.0012711`, pooled는 `+0.0000426`으로 시기별 방향이 갈렸다. 새로 추가한 23행은 모두 오탐이었다. 따라서 `STOP_BEFORE_FULL_FIDELITY`가 맞으며, full GroupDRO나 장시간 3-seed 학습은 실행하지 않는다.

이번 결과는 “GroupDRO가 실패했다”는 결론이 아니다. station×layer 균형 replay라는 저비용 proxy가 안정적인 이득 신호를 만들지 못했다는 결론이다.

## 무엇을 사전에 거르는가

### 1. 정적 가치성 preflight

다음 조건 중 하나라도 어기면 GPU를 쓰기 전에 중단한다.

- 1차 지표가 공식 목적과 같은 binary row F1이 아니다.
- 확인 시기가 둘 미만이거나 G/I/S station slice가 빠졌다.
- 저충실도 단계와 명시적 중단 조건이 없다.
- 공식 test label 접근, submission 생성 또는 upload가 계약에 들어 있다.
- 변경 메커니즘이 비어 있거나 이미 닫힌 실험 가족과 사실상 중복된다.
- 근거 문헌과 반증 근거가 등록되지 않았다.

현재 registry는 partial pooling, cascade, frozen adapter, IORS residual, CP rescue, LightGBM long-event, RPCA, real-event donor, AnomalyBERT, target masked quantile, block inpaint, checkpoint consensus 등 12개 닫힌 축을 등록했다. 같은 intervention layer에서 태그 Jaccard 유사도가 임계값을 넘으면 자동 탈락한다.

### 2. 저충실도 multi-fidelity gate

정적 심사를 통과해도 바로 본학습하지 않는다. 이번 rung은 exact phase별 e150 checkpoint에서 시작해 균형 replay를 10 epoch만 수행했다. Q3와 Q4를 모두 예측·봉인한 뒤 truth를 열고, 다음 조건을 모두 만족해야만 승격한다.

1. Q3 delta F1 > 0
2. Q4 delta F1 > 0
3. pooled delta F1 > 0
4. 추가 예측이 한 station에 80% 이상 몰리지 않음
5. 공식 test 접근 0, submission 생성 0, upload 0

이번 후보는 1번을 실패했고, station별 추가 행 집계는 구버전 result schema에서 비어 있어 fail-closed 처리되었다. Q3 실패만으로도 본학습 중단은 결정적이다.

## 실제 결과

| 구간 | baseline F1 | candidate F1 | delta F1 | 추가 행 / true | 제거 행 / true |
|---|---:|---:|---:|---:|---:|
| Q3 | 0.9067084 | 0.9058969 | -0.0008115 | 4 / 0 | 13 / 9 |
| Q4 | 0.8872198 | 0.8884909 | +0.0012711 | 19 / 0 | 31 / 0 |
| pooled | 0.8987428 | 0.8987854 | +0.0000426 | 23 / 0 | 44 / 9 |

pooled의 미세한 양수만 보면 승격시킬 수 있지만, 그 해석은 잘못이다. Q3에서는 제거한 13행 중 9행이 실제 양성이어서 recall을 해쳤고, Q4에서는 제거한 31행이 모두 오탐이어서 precision이 좋아졌다. 즉 같은 조정이 시기별로 서로 다른 의미를 가졌고, 안정적인 환경강건 신호가 아니라 불안정한 proposal 축소로 보인다.

## 문헌이 지지하는 부분과 지지하지 않는 부분

GroupDRO는 사전에 정의된 그룹의 최악 성능을 낮추는 목적을 가진다. 따라서 station×layer 환경을 명시하고 worst-environment를 보려는 방향 자체는 타당하다. 다만 IRM 계열은 가정이 맞지 않거나 test 환경이 충분히 다르지 않으면 ERM을 이기지 못할 수 있으므로, “불변성”이라는 이름만으로 승격할 수 없다. [Awasthi et al., 2024](https://proceedings.mlr.press/v237/awasthi24a.html), [Rosenfeld et al., 2021](https://arxiv.org/abs/2010.05761)

Hyperband는 약한 설정을 조기에 중단하고 유망한 설정에만 자원을 더 배분한다. Multi-fidelity Bayesian optimization도 적은 epoch나 데이터 같은 싼 proxy를 이용해 비싼 평가를 줄인다. 이번 10→30→full 구조의 직접적 근거다. [Li et al., 2018](https://www.jmlr.org/papers/v18/16-558.html), [Wu et al., 2020](https://proceedings.mlr.press/v115/wu20a.html)

RAINCOAT는 시간·주파수 표현과 label shift를 함께 다루는 시계열 domain adaptation 구조를 제안한다. 환경균형 replay가 막힌 뒤의 다른 구조적 축으로는 의미가 있지만, P1에서 target covariate 사용과 leakage 경계를 먼저 계약해야 한다. [He et al., 2023](https://proceedings.mlr.press/v202/he23b.html)

## 새 승격 기준

| 단계 | 비용 | 통과 조건 | 실패 시 |
|---|---|---|---|
| 0. 정적 preflight | 초 단위 | 10개 계약·신규성 검사 전부 통과 | 가설 폐기 또는 문서만 보정 |
| 1. 10-epoch rung | 수 분 | Q3·Q4·pooled 모두 양수, concentration gate 통과 | 즉시 STOP |
| 2. 30-epoch rung | 수십 분 | 동일 방향 유지, 추가 precision > 0, learning curve 악화 없음 | full 금지 |
| 3. 3-seed full | 시간 단위 | seed 평균·worst slice·pooled 모두 비열화 없음 | 공식 후보 금지 |
| 4. 공식 probe | 희소 기회 | 로컬 근거와 정보가치가 충분하고 사용자 승인 | 점수·로컬 괴리만 기록 |

소수점 개선 자체를 무시하지는 않는다. 다만 작은 개선일수록 시기 일관성, 추가 행 precision, station 집중도와 독립적으로 결합해야 한다. 공식 점수와 로컬 F1의 단위가 다르므로 `로컬 +0.001 = 공식 +0.001점` 같은 환산은 금지한다.

## 한계와 오판 가능성

- Q3/Q4는 이미 여러 연구에서 노출된 retrospective window이므로 fresh generalization 주장이 아니다.
- 10 epoch proxy와 full ranking의 상관이 낮으면 late bloomer를 거짓 탈락시킬 수 있다.
- 현재 station addition concentration은 result schema 전환 전에 실행되어 기록되지 않았다. 이 항목은 낙관적으로 채우지 않고 unavailable로 실패 처리했다.
- 단일 seed는 분산을 추정하지 못한다. 단일 seed가 양수라는 사실은 승격의 필요조건일 뿐 충분조건이 아니다.
- 이번 실험은 GroupDRO 최적화가 아니라 균형 replay proxy다. full GroupDRO 전체 가설 공간을 닫지 않는다.

## 다음 연구 방향

동일한 환경균형 replay는 닫는다. 다음 후보는 기존 intervention layer와 겹치지 않는 **time-frequency target representation alignment** 또는 명시적 worst-environment loss 중 하나여야 한다. 어느 쪽이든 먼저 정적 registry를 통과하고, 10→30 epoch의 두 저충실도 rung을 통과해야 한다. target covariate를 쓰는 경우에는 official test label 0, sample/submission 0, time-causality와 transductive 허용 범위를 별도 preflight로 잠근다.

이번 사이클의 결론은 새 모델의 승격이 아니라 **저가치 실험을 빠르게 죽이는 운영 장치의 승격**이다.

## 재현 파일

- 제안 계약: `configs/experiments/p1_environment_balanced_replay_screen_20260829_v1.json`
- 닫힌 실험 registry: `configs/experiments/p1_experiment_value_registry_20260829_v1.json`
- 정적 preflight: `artifacts/p1_environment_balanced_replay_screen_20260829_v1/preflight.json`
- 저충실도 결과: `artifacts/p1_environment_balanced_replay_screen_20260829_v1/run/result.json`
- postrun gate: `artifacts/p1_environment_balanced_replay_screen_20260829_v1/postrun_gate.json`
- 실행 코드: `scripts/run_p1_environment_balanced_replay_screen_20260829_v1.py`
- 테스트: `tests/test_p1_experiment_value_preflight.py`, `tests/test_p1_ms_tcn_environment_balanced_replay.py`, `tests/test_p1_low_fidelity_gate.py`
