# P2 OAS α40 공식 probe readiness

## 결론

**OFFICIAL_PROBE_READY**입니다. 다만 이는 최종 champion 승격이 아니라, 사용자 명시 승인하에 정확히 한 번 공식 곡률을 확인할 가치가 있다는 판정입니다.

대상 파일은 `C:\Users\cedis\Downloads\해양 해커톤 제출용\20260828_P2_SEASONAL_OAS_TS40_PROJECTED_READY\P2_submission.csv`이며 SHA-256은 `6e28ddb8d78c0969e5104d7efbe28e1762f51e80d759fceb86cdef52baa29b96`입니다.

## 재현성과 파일 QA

- 26,061행, 열 순서 `station,layer,time,temp`
- layer 2/3/4 행 수 `8,713 / 8,712 / 8,636`
- test·sample key 순서 일치, key 결측·중복 없음
- temp 전부 finite, 범위 `16.302557–29.081289°C`
- canonical artifact와 byte-identical, 크기 1,284,060 bytes
- PAVA 재투영 최대차 `0`
- α10·α20은 저장본과 재생성본이 byte-identical이며 fit·projection receipt도 동일

## 공식 동일 계보 증거

α10은 공식 RMSE `0.507628`, 점수 `26.963865`; α20은 RMSE `0.483661`, 점수 `27.264587`입니다. 동일 계보에서 α20은 α10보다 RMSE `0.023967°C`, 점수 `0.300722점` 개선됐습니다.

정확한 α10·α20·α40 예측 벡터 기하를 사용하면 α40 공식 RMSE 범위는 `0.431042–0.465548°C`, 중심은 `0.448627°C`입니다. 상한에서도 α20보다 예상 점수가 `0.227268점` 높습니다. 이 계산은 공식 평가가 배포 `score.py`와 같은 26,061행 통합 RMSE이고 기록 점수가 6자리 반올림이라는 가정에 의존합니다.

## 로컬 반증

기존 exposed OOF 69,850행의 **unprojected** 동일 방향에서는 α20 RMSE가 `0.759201`, α40이 `0.775603`으로 α40이 `+0.016402°C` 나쁩니다. KST-day bootstrap α40−α20 90% 구간도 `[+0.007860,+0.024632]°C`입니다.

- 2024 Sep–Oct: `+0.031368°C`
- 2025 Jul–Aug: `+0.045791°C`
- 2025 Nov–Dec: `-0.115578°C`
- layer 2/3/4: 모두 악화

α40 projected local artifact가 동결돼 있지 않고 이번 재감사는 원본 public endpoint 값을 읽지 않았으므로, α40 local 수치는 unprojected 진단입니다. α20의 기존 projected RMSE는 `0.757714°C`입니다.

## 판정 이유와 정지 규칙

로컬 반복성 gate만 보면 HOLD입니다. 그러나 P2의 기존 transport 감사는 전역 보정계수를 금지하면서도 **고정 파일축의 공식 all-row 기하를 로컬 surrogate보다 우선**하도록 결론 내렸습니다. 이번 α40은 α20 대비 19,531행이 달라지는 물질적인 세 번째 점이며, 공식 α10→20 추세와 로컬 포화가 충돌하는 원인을 한 번에 구분합니다. 그래서 최종 승격이 아니라 정보가치가 높은 공식 probe로 READY입니다.

이 정확한 α40을 공식 검증한 뒤에는 결과를 기록하기 전 α60·α80 제출이나 같은 축의 추가 튜닝을 하지 않습니다. 이번 감사에서는 원본 관측값·공식 정답을 읽지 않았고, 재학습·새 CSV 생성·업로드도 수행하지 않았습니다.
