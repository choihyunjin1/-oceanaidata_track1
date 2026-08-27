# P1 고용량 장문맥 MS-TCN++/ASRF 연구 및 실행 보고서

**실험 ID:** `p1_incumbent_preserving_mstcn_asrf_v1`  
**상태:** 사전등록 및 구현 중 — 결과 섹션은 실행 영수증 생성 뒤 확정  
**목표:** 현 공식 합산점수 대비 최소 `+3.0`점이 가능한 구조적 후보를 찾되, 로컬 개선을 공식 개선으로 오인하지 않는다.

## 결론 요약

세 문제를 다시 비교한 결과, 첫 고용량 실험은 P1에 배정했다. 동결 Round-B의 로컬 OOF는 F1 `0.864670`, precision `0.951804`, recall `0.792152`이며 TP/FP/FN은 `12,718 / 644 / 3,337`이다. FN `3,337`행 중 `3,330`행이 19행 이상 연속 이벤트에 속한다. 이 값은 decoder가 회수할 수 있다는 oracle이 아니라 소속 기준 상한이다. 따라서 현재 병목은 임계값 주변의 전반적 불확실성보다 **긴 offset·drift·noise 구간의 미검출**이다.

새 모델은 incumbent를 대체하지 않는다. 실제 178채널 계약에서 `13,180,427` 파라미터인 offline MS-TCN++/ASRF가 긴 구간을 추가 제안하고 최종 후보는 `Round-B OR 새 구간`으로 만든다. 이 제약은 기존 양성의 `1→0` 변경을 원천 금지한다. 최대 `300 epoch`를 허용하되 마지막 epoch가 아니라 inner chronological validation에서 선택한 최선 checkpoint를 복원한다.

추가 예측 precision을 0.80으로 가정한 산술적 headroom은 다음과 같다. 이는 모델 성능 예측이 아니라 필요한 회수량을 해석하기 위한 계산이다.

| 회수 TP | 추가 FP | 로컬 F1 | ΔF1 |
|---:|---:|---:|---:|
| 500 | 125 | 0.879968 | +0.015298 |
| 846 | 211 | 0.890201 | +0.025531 |
| 1,500 | 375 | 0.908731 | +0.044061 |
| 2,000 | 500 | 0.922267 | +0.057597 |
| 2,500 | 625 | 0.935284 | +0.070614 |

로컬 Q2 화면은 `ΔF1 ≥ +0.015`, added precision `≥0.75`, long-event recall gain `≥+0.10`을 모두 요구한다. 전체 로컬 승격은 pooled `ΔF1 ≥ +0.0255`와 day-block bootstrap CI90 하한 `≥+0.012`를 요구한다. 이 기준을 통과해도 공식 `+3점`을 자동 주장하거나 제출 파일을 만들지 않는다.

## 왜 P1인가

### 공식 기준

2026-08-26 기록된 공식 최고치는 다음과 같다.

| 문제 | 공식 최고 지표 | 점수 |
|---|---:|---:|
| P1 | F1 `0.817873` | `28.492736` |
| P2 | RMSE `0.536536°C` | `26.601139` |
| P3 | RMSE `0.599072m` | `23.825229` |
| 합계 | — | `78.919104` |

현재 1위와의 격차는 `4.587329`점이며 사용자 기준은 다음 제출 후보가 적어도 `+3점` 개선 가능성을 가져야 한다는 것이다. 2026-08-26 P1 세 점수에서 관측된 변환 기울기는 F1 `+1.0`당 약 `26.578점`이다. 이를 같은 구간에 단순 선형 외삽하면 P1 단독 `+3점`에는 공식 F1 약 `+0.112876`, 즉 `0.930749`가 필요하다.

로컬 기준선과 공식 기준선이 다르므로 두 비교를 분리해야 한다. 동일한 `+0.112876` delta를 로컬 Round-B에 더하면 로컬 F1 `0.977546`가 필요하다. 반면 공식 목표 endpoint `0.930749`를 로컬에서도 찍는 경우의 로컬 delta는 `+0.066079`다. 후자는 단지 endpoint 정렬 시나리오이며 검증된 local→official transport가 아니다. 따라서 이번 Q2는 공식 승격이 아니라, 큰 회수율이 실제 존재하는지 확인하는 첫 화면이다.

### 문제별 연구 비교

| 문제 | 가장 강한 새 구조 | 현실적 로컬 개선 기대 | 비용/위험 | 우선순위 |
|---|---|---:|---|---:|
| P1 | incumbent-preserving MS-TCN++/ASRF long-event refiner | F1 `+0.015~+0.04` 탐색 | 3~6 GPU시간 Q2; 공식 전이 불확실 | **1** |
| P2 | full-context SSSD-S4 + vertical-set residual imputer | RMSE `-0.01~-0.02°C` 화면 | 6~18 GPU시간; 10,800-step 구현 위험 | 2 |
| P3 | grouped masked-patch mean+quantile transformer | RMSE `-0.005~-0.015m` | 12~30 GPU시간; 미래 forcing 부재 상한 | 3 |

P2와 P3 후보도 단순 재실행은 아니다. P2의 기존 `sssd_ssm_style`은 실제 S4가 아니라 512-step ModernTCN이었고, P3의 기존 Patch 모델은 약 9.3만 파라미터·사전학습 없음이었다. 그럼에도 현재 확인 가능한 회수 오류량과 점수 민감도는 P1이 가장 크다.

## 기존 P1 딥러닝 실패를 닫지 못한 이유

- 오래된 TCN은 창이 2,016행이어도 실제 dilation `1,2,4`의 유효 수용영역이 약 29행, 즉 4.8시간이었다.
- 최신 binary/masked TCN도 width `32/64`, 수용영역 `31`행, 학습량 `120` 또는 `30+90` optimizer step이었다.
- 오래된 Patch Transformer는 `d_model=64`, 2~3층이고 checkpoint 선택이 1~3 epoch에 끝났다.
- P1의 실제 offset/drift는 최대 86.5시간이다. 짧은 수용영역 모델의 실패는 14일 문맥의 다단계 경계 보정 모델을 반증하지 않는다.
- 반대로 TE-TAD-lite는 활성화된 proposal의 median IoU `0.850564`, normal-window FP `0`이었지만 target recall이 `14/17=0.823529`에 그쳤다. 따라서 이번 모델은 query 활성화 대신 dense row supervision과 multi-stage refinement를 사용한다.

## 선택 구조

### 입력과 분할

- 77개 등록 수치 feature와 각 missing flag
- train-prefix에서만 적합한 station/layer/depth-regime one-hot
- valid-row와 exact-cadence gap channel
- 2,048행 창, stride 512, center-weight overlap-add
- split을 먼저 수행한 뒤 각 split 안에서만 창을 만드는 구조와 centered feature 양쪽 support `168+168`시간, 추가 168시간 여유를 합친 21일 purge

Q2 화면의 inner-train 마지막 시각은 `2025-01-20 14:50 UTC`, inner validation은 `2025-02-10 15:00`부터 `2025-03-10 15:00 UTC` 미만이다. outer refit은 `2025-03-10 14:50 UTC`까지이며, Q2 frozen OOF는 `2025-03-31 15:00 UTC`부터 시작한다.

### 모델

- width 256
- prediction generator dual dilation `1..512`, 10층
- refinement stage 3개 × 10층
- start/end Gaussian boundary head, σ=3행
- spike/noise/flatline/offset/drift 5-type auxiliary head
- 13,180,427 parameters at the registered 178-channel input
- theoretical receptive field: generator 3,969행, final stage 10,107행

[MS-TCN](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html)은 dense temporal prediction을 다음 stage가 반복 보정하는 구조와 과분할을 억제하는 smoothing loss의 근거다. [MS-TCN++](https://arxiv.org/abs/2006.09220)와 [공식 구현](https://github.com/sj-li/MS-TCN2)은 prediction generator의 dual dilation과 refinement stage 분리를 뒷받침한다. [ASRF](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html)는 frame segmentation과 별도 boundary regression 결합의 근거다.

이 논문들은 영상 행동 분할에서 나온 결과다. 해양 센서 QC에서 같은 이득을 보장하지 않으며, 이번 실험은 구조 전이를 검증하는 것이다.

### 학습

- AdamW, LR `3e-4`, weight decay `1e-4`
- warm-up 10 epoch, cosine decay 최저 `3e-6`
- BF16, EMA `0.999`, gradient clip `1.0`
- 최대 300 epoch, patience 50
- BCE + soft Dice + boundary BCE + type BCE + truncated temporal smoothing
- stage weights `[0.25, 0.25, 0.5, 1.0]`
- inner-train positive weight는 음/양 행 비율을 `[1,20]`에 clip

RTX 5090의 label-free synthetic tensor 측정에서 실제 계약 폭과 같은 입력 `[64,2048,178]` 한 optimizer step은 약 `0.693초`, peak allocated memory는 `9.46GB`였다. 이 결과를 보기 전에 batch 64로 고정했으며 과학적 행이나 라벨은 사용하지 않았다.

## 데이터·진실 방화벽

1. config에 등록된 feature, key sidecar, label cache, OOF, Round-B anchor의 byte size와 SHA-256을 확인한다.
2. 학습 전에는 label cache에서 outer-training cutoff 이하 행만 투영한다.
3. Q2 feature와 key membership은 사용할 수 있지만 Q2 label은 열지 않는다.
4. fresh outer-refit 모델의 Q2 확률, key hash, model/config/code hash receipt를 atomic write한다.
5. receipt가 존재하고 재검증된 뒤에만 frozen OOF의 Q2 label을 연다.
6. 공식 test, sample submission, submission candidate는 읽거나 만들지 않는다.

## 승격 기준

### 구현 sanity

- deterministic 32 positive + 32 normal window overfit
- finite loss/gradient
- IoU≥0.70 event recall ≥0.95
- median matched IoU ≥0.80
- 예측이 있는 normal window ≤1

### Q2 연구 화면

- `ΔF1 ≥ +0.015`
- added precision `≥0.75`
- long-event recall gain `≥+0.10`
- long-event recall은 exact-cadence의 최대 양성 run 중 길이 19행 이상이며 offset·drift·noise를 하나 이상 포함한 run의 모든 행에 대한 row-weighted recall이다. truth run에는 상한을 두지 않고 anchor/candidate가 동일 행에서 비교된다.
- 동일 Q2 모집단의 candidate FP 행 / anchor FP 행 비율 `≤1.50`
- anchor positive 제거 `0`

### 전체 로컬 승격

- pooled `ΔF1 ≥ +0.0255`
- paired day-block bootstrap CI90 lower `≥+0.012`
- 3개 fold 중 2개 이상, 3개 station 중 2개 이상 개선
- Q3 ΔF1 `≥0`, G-ORS ΔF1 `≥-0.005`
- spike 및 flatline recall delta 각각 `≥0`

Q2 실패 시 Q3/Q4와 seed 확대를 실행하지 않는다. Q2가 통과하면 사전등록한 3 seed만 확인하며, 결과를 보고 구조나 threshold grid를 바꾸지 않는다.

## 실행 결과

> 실행 완료 뒤 `terminal_result.json`, learning curve, blind receipt, Q2 gate와 독립 QA 수치를 여기에 반영한다.

## 한계

- 역사적 로컬→공식 방향 일치는 완전하지 않다. 최신 기록에서도 P1 router는 방향은 일치했지만 공식 개선폭이 로컬의 약 10.84배였고, P3는 방향 자체가 뒤집혔다.
- Q2/Q3/Q4는 이전 연구에서 이미 반복 노출된 역사적 surface다. 엄격한 fresh holdout이 아니라 재사용 안정성 증거다.
- anchor OR 구조는 기존 FP 644행을 제거할 수 없다. 새 모델은 recall만 개선하고 추가 FP 위험을 관리한다.
- 논문 구조는 영상 분할에서 검증됐으며 센서 이상 검출로의 전이는 미확인이다.
- 공식 `+3점`은 로컬 F1 임계값만으로 보장할 수 없다.

## 다음 단계

1. 구현·누수·GPU smoke test를 통과한다.
2. sanity gate를 한 번 실행한다.
3. 통과한 경우에만 inner selection과 fresh Q2 refit을 실행한다.
4. Q2 gate가 통과한 경우에만 Q3/Q4와 3-seed 확인을 별도 실행한다.
5. 전체 로컬 승격과 공식 기대값 교정이 모두 충족될 때만 사용자에게 제출 승인을 요청한다.
