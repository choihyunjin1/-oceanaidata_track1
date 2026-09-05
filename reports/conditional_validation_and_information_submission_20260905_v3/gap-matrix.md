# 이번 검증이 확인한 것과 남긴 것

| 문제/단계 | 기술 검증 | 성능 증거 | 아직 말할 수 없는 것 | 행동 |
|---|---|---|---|---|
| P1 frozen complete policy | 키·형식·hash·별도PID 전체CSV exact PASS | 실제 Public F1 0.767370, 27.150461점; 당일control 대비−0.620939점 | 수심만의 효과, Private 성능, clean-machine 완전재학습 | 교체하지 않음;1회 영수증 보존 |
| P2 첫0-fit 조건부 | 동일69,850키;142-check QA/11synthetic; root집계산식156개 PASS | 가을intact작은개선/가을outage개선; pooled·여름·겨울위험 | 모든scenario지원, 새독립holdout, 당일3-seedcontrol과동일성 | 원결과유지;별도3-seed단계 |
| P2 추가3-seed | 9fit완료/17synthetic/Ruff/845-check QA PASS;별도PID128학습행재생PASS | 가을intact+0.000783517℃/가을결측+0.030801703℃악화;정보조건false | 전체공식답안재현·공식성능·새독립holdout | 공식입력/CSV/추가제출0;첫seed개선미재현으로종료 |
| P3 | 기존 A/B근거 유지 | 새후보내부RMSE악화 | 학습하지않은비-no-op배포정책의공식성능 | 중복/미완성추가제출0 |

이 표는 평균개선과 안정성, 기술지원범위, 재현성을 같은 PASS/FAIL로 합치지 않는다. 정보확보용 제출도 데이터규정·계보·완성답안QA를 우회하지 않는다.
