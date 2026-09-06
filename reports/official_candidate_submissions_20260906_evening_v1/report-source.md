# 2026-09-06 저녁 공식 채점

결론: 세 미제출 후보를 각각 한 번 제출해 채점 완료를 확인했다. P2 L120은 확인된 과거 Public 최고점도 갱신했다. P3 numeric은 clean 기준보다 개선됐지만 과거 전체 최고점은 아니다. P1 tree/MS 복원 후보는 개선되지 않아 기존 fallback을 보존한다.

| 문제·후보 | 제출 KST | 공식 지표 | 공식 점수 | 비교와 판단 |
|---|---|---:|---:|---|
| P1 b2f17 tree union OR MS | 17:55 | F1 0.773439 | 27.311774 | 오늘 06:39 후보 대비 F1 -0.012505, 점수 -0.332350. 미승격 |
| P2 fee6118 C3 L120 3-seed | 17:55 | RMSE 0.418892℃ | 28.077280 | clean 기준 대비 RMSE -0.036251℃, 점수 +0.454862. 과거 표시 최고 28.012945 대비 +0.064335 |
| P3 ff42a6 numeric lead | 17:56 | RMSE 0.604351m | 23.741446 | clean 기준 대비 RMSE -0.002832m, 점수 +0.044946. 전체 source-only cold 검증은 아직 별도 대기 |

비교값은 UI 표시 정밀도 기준이다. 내부검증과 Public 결과는 별개이며, Public 값을 학습·보정계수에 사용하지 않았다. P2는 내부 전체 8블록 평균 악화에도 Public은 개선됐다는 사실을 함께 보존한다. 이 사실만으로 모든 분포에서 개선됐다고 결론내리지 않는다.

## 증거와 파일 경계

- `preupload-inventory.json`: 제출 직전 세 exact CSV SHA 및 기존 QA receipt hash 대조. 이 도구 자체는 새 학습/새 schema QA를 수행하지 않는다.
- `result.json`: 정확한 파일 상대경로, 전체 SHA256, 행 수, UI 점수, 시각, 잔여 기회.
- [문제1](https://oceanaidata.org/app/problems/5), [문제2](https://oceanaidata.org/app/problems/6), [문제3](https://oceanaidata.org/app/problems/7)의 일반 `제출하고 채점` 성공 메시지와 [제출관리](https://oceanaidata.org/app/submissions)의 최신 세 기록을 대조했다.
- UI에 수치 접수 ID/서버 파일 SHA가 노출되지 않아 이를 추정하지 않았다. 로컬 SHA는 정확한 chooser 경로와 연결한 것이다.
- 각 1회 제출. 오늘 잔여: P1 0/3, P2 0/3, P3 1/3. P1/P2 UI는 09-07 00:00 KST 초기화를 표시했다.
- 최종 모델 제출/잠금, commit/push, hidden truth/외부 관측 접근 0. 기존 fallback과 현재 cold 학습을 변경하지 않았다.
- 오래된 P1 제출 감시 `p1-28-9`는 실제 채점 완료 후 PAUSED. 별도 P1 portable cold 학습 및 P3 budget90 대기는 중단하지 않았다.

리더보드는 이 시점 분당독고다이 8위/81.190220점을 표시했으나, 과거 P1/P3 최고 답안을 포함한 합계이므로 새 세 후보 또는 최종 재현 패키지의 합계로 해석하면 안 된다.
