# P1 양측 지원 점검 — 새 학습 전 결과

## 결론

**같은 검증행을 유지하고 양측 배포 train 시기를 허용하면 H1_2025의 unseen station-layer 행은
107,125→0이다. 성능은 측정하지 않았다.** 세 반기 모두 검증 key/order SHA가 완료된 forward 실험과 같으며
지원 부족 행을 삭제하지 않았다. 이는 별도 retrospective 양측 평가의 지원 가능성이지 forward 성능의 개선 증거가 아니다.

[사전등록](preregistration.md), [canonical 집계 결과](result.json).
상태는 `SUPPORT_AUDIT_COMPLETE_NOT_EVALUATOR_EXECUTION_READY`이다.
26.938초, fit0, metric0, 관측 feature값 로드0, 공식/hidden/모델/예측/CSV/upload0.
읽은 열은 배포 train의 station/year/layer/time/label이며 label은 양성 run 경계와 지원 집계에만 사용했다.

| 검증 반기 | 검증행(변경0) | Forward 학습행 | 양측 학습 후보행 | Forward unseen행 | 양측 unseen행 |
|---|---:|---:|---:|---:|---:|
| H2_2024 | 172,638 | 86,987 | 565,539 | 0 | 0 |
| H1_2025 | 208,093 | 270,708 | 514,367 | 107,125 | 0 |
| H2_2025 | 287,862 | 444,081 | 444,081 | 0 | 0 |

H2_2025는 배포 train에 그 뒤의 시기가 없으므로 실제 학습행이 forward와 같다.
지원의 정의는 해당 station-layer에 학습행이 존재하는지이다. 동일한 계절/이상 유형/범위의 충분한 지원,
모델의 이득 또는 실제 배포에서 정답을 구할 수 있음을 뜻하지 않는다.

## 정확한 분할 및 확인

- 검증은 frozen v5의 동일한 3반기 달력 음성행 + 양성 run 시작 소유 반기의 전체 run이다.
- 학습 후보는 `time < start−21d OR time >= end+21d`. 양성 run이 제외 구간과 교차하면 전체 학습에서 빠진다.
- 반열림 제외 구간 밖이지만 whole-run 규칙으로 추가 제외한 행은 H2_2024 126, H1_2025 13, H2_2025 0이다.
- train/validation 겹침0, 양성 run 공유0, train의 partial positive run0을 실행 중 검사했다.
  v1 validation key SHA와 세 fold 모두 일치했다. 양쪽 purge 정확 경계와 fold를 가로지르는 run의 합성검사2 PASS, Ruff PASS.
- checker SHA `0c7f9718c87b1f78c95016d716d88da4fb4ce4011e13a028cdda64f01701556d`.
  결과 SHA `5d2ed8b088e569d6f74eddfacaa24bd91ec2a123d13ac7eced86288d4f898fca`.

## 다음 계약의 제안 — 실행하지 않음

1. **질문을 분리한다.** 현재 v1의 chronological primary H1 전체 F1은 그대로 보존한다.
   새 ID는 다른 배포 역사 시기의 label을 활용할 수 있는 retrospective 보간형 분류 비교다.
   실제 미래 시기에 대한 순방향 일반화 또는 fresh confirmation이라고 부르지 않는다.
2. 검증3fold, 양측21일purge, run 귀속/전체 배제, 전체 평가행과 known/unseen 부표를 그대로 사전 고정한다.
   단순히 지원행이0으로 줄었으므로 쉬운 사례만 남긴 평가지표를 만들지 않는다.
3. **Outer 제외를 먼저 적용한 뒤 nested inner**를 정의한다. 각 outer에서 허용된 학습 후보만으로
   inner 학습/검증을 만들고 inner 구간에도 양측purge 및 whole-run 제외를 적용한다.
   정확한 inner 날짜/집계/tie-break를 Root의 prospective 계약에서 고정한 뒤에만 학습한다.
   v1의 outer 성적을 보고 inner 날짜나 threshold를 골라서는 안 된다.
4. train/inner/outer partition을 원시 입력 단계에서 분리하고 통계·encoder·특징·decoder의 differential sentinel을
   새 양측 조건에서 검증한다. 시간 gap을 넘어 past/future를 이어붙이지 않는다.
   허용 partition 전체 run에 의존하는 hysteresis는 유한 purge만으로 안전하다고 주장하지 않는다.
5. 동일 키에서 새 clean O/B와 B+bracket 절차를 비교한다. v1 수치의 새 계약 소급 재채점,
   과거 공식 점수 이식 또는 후보 성능 자동 실행은 하지 않는다. 새fit/time예산과 terminal 분기를 먼저 봉인한다.

이 문서는 추가 학습 승인이 아니며, source 지원 점검 완료 후 자동 후속 실행은 없다.
