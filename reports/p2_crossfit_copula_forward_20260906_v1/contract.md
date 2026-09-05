# P2 CPU 8-fold C3 + inner OOF copula — 실행 전 계약

검증 질문은 "같은 11개 profile 특징과 copula 구조/강도1.0에서, in-sample 잔차 대신 inner purged OOF 잔차를 쓰면 B3 수온 복원 RMSE가 개선되는가"다. 이번 실행에는 공식 입력/CSV/fullfit/업로드/Git/외부자료/과거 모델·OOF 재사용이 없다. 이전 3-fold 내부 이득을 이번 8-fold 성적으로 재사용하지 않는다.

## 구속 계약

`configs/evaluation/ocean_forward_v5.json`의 P2 B1~B8 날짜 및 양측7일 purge를 그대로 사용한다. 모든 target temp+psal은 특징 전에 제외하고 label은 별도 배열이다. 공개 T5+S5 outage는 각 fold의 달력상 마지막17일에만 적용하며, 그 뒤 공개 profile 및 baseline/scale/tokens/copula 특징을 재계산한다. B3 natural RMSE가 primary, 전체 natural/자연 T5 결측/마지막17일 outage 및 whole-fold testmatched가 보조다. 임의 합성가중0, bootstrap p≥0.8 hardgate0, 같은 키/분모/순서/행삭제0이다.

B4/B8 마지막17일 eligible target는0행이므로 RMSE/CI는 NOT_ESTIMABLE이며0점차/PASS가 아니다. B1/B3의 actual-depth 지원 부족33/34행은 nominal 기반 C token 지원으로 보존된다. C는 missing actual depth 채널을 zero+missing flag로 표시하고 token 유효성은 finite temp/nominal로 정한다. 보정기의 실제 수심 보간은 기존 nominal fallback을 쓴다. 이 둘을 실제 수심 관측이 존재한다는 뜻으로 혼동하지 않는다.

## inner 결정 — 성능·목표값 사용0

각 outer validation±7일에 전혀 걸치지 않는 전체 KST 달력 월을 2024-05~2025-12에서 나열한다. 배포 source eligible 행이 있는 달만 남긴 후 날짜순 index `floor((N−1)/3)`와 `floor(2(N−1)/3)` 두 달을 선택한다. 결과/오차/계절별 개선 여부로 날짜를 이동하지 않는다. 각 inner 모델은 `outer_train AND outside(inner_month±7days)`에서 scratch3seed를 학습한다. outer label은 물론 그 label로 학습된 base model도 crossfit residual 생성에 사용하지 않는다. 두 inner month의 같은 행 C3 평균 예측으로만 잔차/CDF/covariance를 적합한다.

| outer | inner1 (OOF rows) | inner2 (OOF rows) |
|---|---|---|
| B1 | 2024-11 (12,960) | 2025-06 (12,950) |
| B2/B3 | 2024-12 (5,133) | 2025-06 (12,950) |
| B4 | 2024-08 (12,660) | 2025-06 (12,950) |
| B5 | 2024-08 (12,660) | 2024-12 (5,133) |
| B6 | 2024-08 (12,660) | 2024-11 (12,960) |
| B7/B8 | 2024-09 (12,934) | 2025-04 (3,978) |

전체 train/inner 키 hash, 지원 분모는 [feature-support.json](feature-support.json) 및 실행 전 봉인에 기록한다. 내부 두 달만 사용하는 calibration은 outer train 전체 OOF가 아니며, 성능 차이는 과적합 감소뿐 아니라 calibration 표본·계절 혼합 차이의 영향도 포함한다. 이는 알려진 한계로 보고한다.

## 고정 모델과 예산

- C: 기존 clean v23 blockmask `VerticalDeepSet` 3seed 평균, seeds20260901/02/03, 60epoch, batch4096, AdamW lr0.001/wd0.0001, normalized Huber+observed-temperature gradient penalty0.01, train-domain weights와 source-row mass-preserving augmentation. **CPU2threads, GPU0**만 변경한다. 과거 CUDA 결과와 동일 수치라고 가정하지 않는다.
- 8fold C3 baseline24fits 전부 끝나면 baseline-result/OOF/models SHA를 먼저 봉인한다. 이후 각 fold inner2×3seeds=48fits와 in-sample/crossfit copula 각1fit=16smallfits. 최대88fits; 신규 fullfits0. model/결과/lock은 새ID 하위이며 이전 것을 덮어쓰지 않는다.
- 최초 B1 seed20260901 1fit은 예정 baseline 그대로 resource pilot이며 재사용한다. validation 예측/점수 전에 `pilot_seconds / augmented_pilot_rows × all_planned_original_rows ×1.3×1.25 +180초`로 보수적 완료 시간을 추정한다. 90분 초과면 자원제한 terminal/NOT_ESTIMATED로 끝내고 root에 알린다. 무단 GPU 전환/epochs 축소/성능 기반 중단0. 실행 중 90분 deadline은 epoch/fit 경계에서 확인하며 한 epoch의 최대 지연 가능성이 있다.

## QA와 판정

합성8 tests는 target T/S 변조 불변, long-form T5/S5 선mask와 파생 특징 재생성 동치, 양측 double purge, nominal 지원, 빈 metric, CPU fit/저장/reload, 실제 native writer/해시/금지 파일 접근을 포함한다. 첫 합성 검사에서 사용하지 않는 diagnostic mean/std의 stale field를 발견해 신규 outage adapter에서 봉인 전 재계산했다. 기존 소스와 C가 실제 사용하는 입력·학습은 변경하지 않았다.

RMSE=sqrt(SSE/n), fold 평균이 아니다. CI90은 KST 7일 cluster(기준2024-01-01), seed20260906, paired2000 resamples다. 개선 resample 비율은 공식 개선 확률이 아니다. 각 평균 이득 후보를 보존하되 worst-fold/layer/자연결측/outage 악화를 함께 기록하며 자동 제출·full학습으로 이어가지 않는다. fallback은 같은 신규 CPU C3다. 종료 후 별도 PID 공개 context→새 outer 모델→세 arm의 전체166,268 historical keys/2surfaces replay와 독립 산술/해시/누출 QA를 한다.
