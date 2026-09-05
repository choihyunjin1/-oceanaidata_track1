# 다음 점수 향상 디자인 — 새 이름보다 유효한 변경 하나

## 판단과 현재 실행 범위

현재 우선순위는 **P2의 관측된 개선을 더 정직한 잔차 학습으로 확인**, **P1의 구간 내부와 양쪽 경계를 함께 표현**, **P3의 리드타임 표현 단일 교체**다. 점수 상승이 입증된 새 설계는 아니며 예상 공식 점수는 미산정이다. 외부 데이터/사전학습, 공식 입력, 업로드, Git 변경은 이번 작업에서 0이다.

기준 코어는 [재생성 원장](BASELINE_REGEN_CHECK.md)의 실제 코어이며 원문 로드맵의 강제 MS-TCN/v52/등가평균과 혼동하지 않는다. 현재 v5는 chronological P1/P3, 가을 primary P2, 평균 개선 후보 보존/위험 별도 표이며 0.8 자동 gate는 없다. 기존 성공·실패를 바꾸지 않는다.

신규 성능 학습 전 [평가 계약](../../configs/evaluation/ocean_forward_v5.json)의 source count audit와 실제 feature adapter의 누출·지원 검증을 완료하고 **B의 동일 평가키 OOF**를 먼저 만든다. 기존 v4의 3-fold 수치를 새 v5 성적으로 쓰지 않는다. CPU-only 요구를 따르는 새 B가 필요하면 GPU가 포함된 과거 코어와 같은 수치라고 가정하지 않는다.

실행 후 추가 근거: [source 지원 검사](../../reports/ocean_forward_support_20260906_v1/report-source.md)에서 P1 H1_2025의 51.48%가 미학습 정점/층이고 P2 B4/B8 outage 대상행이0임을 확인했다. P1은 정점 특이값보다 일반적인 구간 모양을 표현하는 bracket 가설의 동기가 되지만, 그 효과를 증명하지 않는다. P2 두 빈 outage는 평가불가로 유지한다. 기준선의 known/unseen slice와 빈 진단 처리를 완료하기 전 새로운 성능 비교표를 만들지 않는다.

## P1: 양쪽 경계가 닫히는 이상 구간 특징

가설: 행별 편차와 24~168h 먼 flank만으로는 수시간 offset/drift 구간의 시작/끝이 서로 맞물리는 증거를 표현하기 어렵다. 짧은 spike나 실제 수온 변화와 구분되는 **구간 양쪽 변화의 방향·복귀·내부 지속성**을 트리에 제공한다.

- 첫 비교는 clean O/B 중 B의 입력에 bracket feature bank 하나 추가. O와 모델 recipe·후처리 선택 절차는 그대로다. MS-TCN·새 decoder·T/S ratio와 동시에 넣지 않는다.
- 표현: 좌/우 경계 jump의 부호와 크기, jump cancellation, 양쪽 외부 중앙값 차이, 내부 대비 양쪽 편차, 실제 관측 시간 coverage. 모든 특징은 공개 관측만 사용한다. 구간 후보 선택에 outer label을 쓰지 않는다.
- 첫 창은 `[6h,24h,72h]`, 각 flank 1h를 설계값으로 고정하는 안. 최대 입력 의존성은 구현 전에 증명하고, 실제 시간 간격·segment 경계를 존중한다. 누락 행을 제거해 간격을 당기거나 label run 길이로 창을 고르지 않는다.
- 원본 80열/현재 특징 중 동등한 계산이 있으면 중복을 제거한다. T–S 19열 추가 실패와 먼 flank 실패를 단순 이름 변경해 재시도하지 않는다.
- 초기 비용 상한안: v5 세 forward fold에서 B inner/outer 각 1fit = 6 challenger fits. O/control은 동일 split/hash일 때만 재사용. baseline 생성 비용은 별도 계상한다. 정확한 inner 선택·창 계산 구현/봉인 후 실행한다.
- 판정: H1_2025 pooled F1, 전체 F1, 유형별 TP/FN·run 내부/경계 오차·FP를 보고한다. 유형별로 유리한 결과를 사후 조합하지 않는다. 최소 +3점이나 anchor 제거0 조건은 추가하지 않는다.
- 선행조건: P1 새 full O 학습 경로 차이를 해결하고 선택된 **동일한 전체 학습 명령**의 독립 2회 답안 일치/차이를 먼저 측정한다. 이것은 성능 튜닝이 아니다.

## P2: copula 잔차를 in-sample이 아닌 inner OOF로 학습

근거: 현재 full copula는 가을 RMSE 0.488284→0.472272℃로 개선했지만, 추가 7/14일 결측에서는 악화했다. 현재 잔차 학습은 C3가 자기 학습 행을 예측한 residual을 사용한다. outer 평가는 독립되어 있어 기존 이득이 자동 무효는 아니지만, 배포 시 보정해야 할 잔차 분포와 학습 잔차가 다를 수 있다.

- 첫 변경은 **잔차 학습 표본을 outer-train 내부 purged OOF로 교체하는 것 하나**다. 11개 물리 특징·copula 구조·강도1.0·C3 구조·증강은 고정한다.
- outer validation과 겹치거나 그 정답으로 학습한 base model의 OOF는 residual fit에 재사용하지 않는다. 각 outer-train 안에서 고정 2개 purged inner block의 3-seed 평균으로 residual을 생성한다. 누락된 inner 지원은 먼저 표에 기록한다.
- v5 8개 block 모두 실행 시 residual용 최대 8×2×3=48 추가 base fits, residual8fits. 기존 아웃오브폴드와 hash/split이 일치할 때만 줄일 수 있다. B/C3 outer 비용은 별도다. CPU-only 파일럿으로 전체 시간 측정 후 확정하며 아직 48fit을 시작하지 않는다.
- 조기 중단은 성능이 아닌 시간/기술 계약으로만 한다. 첫 fold 성적으로 inner 기간·보정강도를 재선택하지 않는다.
- 기존 고정 full copula와 새 OOF copula, no-op C3를 같은 행에서 비교한다. 평균 개선 후보는 유지하고 outage 악화를 별도 보고한다.
- **다음 독립 변경**으로만 public support count/수심 범위/missingness를 사용한 보정 gate를 검토한다. cross-fitting과 gate를 한 실험에 같이 넣지 않는다. T5 전문가 두 모델 선택 규칙의 재시도와도 구분한다.
- fallback은 보정 없는 재생성 C3다. 현 후보의 개선을 이 새 설계의 실측 성과로 옮겨 적지 않는다.

OOF 원칙의 1차 근거: [scikit-learn StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)는 base와 결합 학습에 같은 학습 예측을 쓸 때 과적합 위험을 경고한다. 이는 본 copula의 개선을 보장하는 근거가 아니라 잔차 학습/평가 구분의 근거다.

## P3: single CatBoost의 lead를 범주형에서 수치형으로

가설: 3/6/9/12/18/24시간을 단순 범주로 처리하는 대신 순서·거리를 가지는 수치로 처리하면, 같은 591개 파랑·기상 요약에서 시간에 따른 residual 변화를 더 효율적으로 학습할 수 있다. tree는 원래 범주형으로도 관계를 학습하므로 **이것이 현재 모델의 버그나 개선 보장은 아니다**.

- 변경은 single 모델의 `lead_h` dtype/cat_features 목록 하나뿐이다. multi 모델·target residual·591열·weights·CPU recipe·shrink·router 알고리즘은 고정한다. 새 single OOF가 나왔으므로 router는 그 arm의 이전 OOF로 같은 절차를 새 적합한다. 옛 router 계수를 복사하지 않는다.
- 새 onset/위상59열은 우선 추가하지 않는다. 기존 event_phase_probe는 0.802405→0.805033m로 악화한 기록이 있다. 그 기록의 분할/계보가 현 계약과 같다는 주장은 하지 않지만, 새 가설로 포장하지 않는다.
- clean lead-continuous 사후 residual 모델과 구분: 이번에는 과거 alpha/기존 답안에 ridge를 붙이는 것이 아니라 배포 train의 **base learner 입력형식**을 바꾼다. 과거 lead-continuous 예상 점수/계수는 사용하지 않는다.
- v5 5개 forward fold에서 single 후보5fits, control single5fits(필요 시), multi는 정확한 v5 control5fits 재사용. router fits 별도. CPU-only 시간 측정 전 무조건 30분 완료를 약속하지 않는다.
- 주평가 pooled RMSE, lead/station/episode 및 worst-quarter를 별도 보고한다. 이득난 lead만 골라 섞지 않는다. 평균 개선이면 새 full single + 같은 multi + 새 full router의 완성정책을 검증한다.
- 다음 분기: 실패하면 단순 seed 추가·바람 dropout·사건가중을 반복하지 않고, error slice에서 부족한 대상/구조를 확인한 후 별도 가설을 정한다.

자료형 구분의 근거: [CatBoost 공식 feature types](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic). 이 문서는 순서형 lead 변경의 점수 상승을 주장하지 않는다.

## 같은 표와 결합 검증

`변경 | 평균 Δ | CI90 | 개선 resample 비율 | worst-block Δ | slice 최대 악화 | fit/시간 | 후보/승격 판단`

F1 Δ>0, RMSE Δ<0가 평균 개선이다. bootstrap 비율은 공식 향상 확률이 아니다. 단독 이득을 단순 더하지 않고 누적본/각 변경 제거 fallback을 같은 평가키에서 비교한다. 최종 README의 문구는 "적합값은 이 학습 실행의 모델/metadata에서 생성; 리더보드 역산 계수·과거 답안 입력 0"으로 실제 경로와 일치시킨다. 독립 패키지 검사 전 제출 적격 확정이라고 쓰지 않는다.
