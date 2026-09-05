# P1 점수 개선 screen — 2026-09-05, 결과 확인 전 계약

목표는 기존 event-day LightGBM보다 좋아지는 것뿐 아니라 같은 내부 평가에서 XGB 및 결합 control보다 실제로 좋아지는지를 구분하는 것이다. 현재 공식 28.909341점의 예측을 재현했다고 주장하지 않는다.

## 고정 조건

- 입력은 환경변수 P1_DATA_DIR 아래 배포 train.csv 하나뿐이다. 공식 test/sample/submission/hidden 및 외부 관측 값은 읽지 않는다.
- Q2/Q3/Q4를 forward outer로 사용하며 이미 노출된 development replay라고 기록한다. outer 이전 21일 purge, 그 이전 60일 inner calibration, inner 이전에도 21일 purge를 둔다. train 말미에 잘린 양성 사건은 train 내부 label만 보고 후퇴한다. 평가 행은 label로 제거하지 않는다.
- 기존 `configs/p1_meaningful_learning_curve_generation_v1.json`의 event/day weighting 및 LightGBM 700 trees/63 leaves recipe를 재사용한다. 기존 p1.toml의 XGB CPU hist 700 trees도 재사용한다. seed20260813, CPU4threads/GPU0, RAM12GiB, wall cap5h다.
- LGBM control/flank 두 arm × inner/outer × 3fold =12 fits. 별도 XGB6 fits. calibration 검색5회/fold, 회당 사전고정 high threshold8개/low=0.5 high/closegap0/minrun12. classifier fit과 calibration 검색 수를 분리 기록한다.
- 공통 안전 수정: 특징은 각 split별로 생성한다. deployment nominal depth와 spike step scale은 해당 학습 split에서만 적합하여 평가에 전달한다. plateau evidence는168h로 cap한다. 원래 평가 전체 depth median을 그대로 재사용하지 않는다. 따라서 이것은 역사 제출 SHA의 exact clone이 아니다.
- 기존 station/layer router는 선택 계보가 확인되지 않아 이번 eligible control에서 제외한다. XGB-alone/B-alone/XGB∨B/XGB∨recalibrated-B 중 inner F1 최고를 control로 정한다. 후보는 flank-alone/XGB∨flank 중 inner F1 최고다. 삭제0 제약은 없다.

## 단일 표현 변경

기존 centered median(현재 근처 포함)과 달리 현재±24h를 제외한24~168h 좌/우 flank를 사용한다. temp, psal, same-time peer residual 각각에 left delta/right delta/flank change/min absolute delta/same sign/left support/right support7개, temp와 psal·peer의 sign disagreement와 availability4개로 총25열이다. 각 summary 최소 지원25%, exact10min segment 밖은 참조하지 않는다. long-window14일 기존 특징의 합성 support와21일 purge를 구별하며337h 전체를 미래 누출 시간이라고 부르지 않는다. split 격리는 plateau/hysteresis가 학습·평가 경계를 넘는 것을 추가 차단한다.

## 평가와 중단

intact full 평가와 고정 seed의 label-independent fragmentation(시작 확률0.0005,6/36/144행 결측)을 별도 평가한다. stress는 label에 무관하며 남은 동일 키에서 intact도 재채점한다. 원본 평가에서 점수 나쁜 행을 빼는 방식이 아니다.

TP/FP/FN pooled micro F1,48h 이상 사건 recall,normalFP, 추가/제거의TP/FP 효과, 확률0.05 미만 장기FN 및 일부 탐지 사건의 누락행을 기록한다. per-row OOF/model은 ignored artifacts에만 저장한다. root independent QA용 qa_oof.npz는 pickle 없이 key/fold/truth/reference/prediction을 제공한다.

pooled candidate가 inner-selected strongest control보다 좋을 때만 control/winner 추가2seeds 자원을 배정한다. 모두 나쁘면 broad HPO나 모델 전환 없이 현재 표현을 닫는다. 확률은 좋아졌지만 threshold만 불안정하면 inner calibration 교정1회가 다음 분기이며, 현재 control이 일부 잡은 사건 내부를 놓친 증거가 있을 때만 별도 decoder1개를 사전등록한다. 기술 실패는 성능 실패와 분리하며 자동 재시작하지 않는다.

## 현재 최고 제출본 재현 의존

O/XGB 학습은 `src/p1_qc/pipeline.py:train_full_model`, B/event-day 학습은 `scripts/run_p1_meaningful_learning_curve_generation_v1.py`에 있다. 현재 최종 패키지의 MS-only 학습과 router_anchor.csv/GI2 key patch 입력만으로는 O/B/router/GI2 전 과정 재학습을 입증하지 못한다. 기존 router의 station/layer 선택 출처와 GI2의 일반 `novel & predicted spike` 생성 원 모델을 확인·재생성해야 한다. 이번 screen에 기존 공식 예측 CSV, key patch, 외부 자료, MS 학습을 넣지 않는다.
