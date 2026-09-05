# P2-A/B 근거와 남은 간극

| 질문 | 이번에 직접 확인한 근거 | 아직 확인하지 않은 것 | 다음 판단 |
|---|---|---|---|
| 기존 C를 정확히 재사용했는가? | 9개 저장모델 재추론이 기존 key/truth/fold/OOF와 exact 일치 | 공식 입력 재추론은 이 실행 범위 밖 | C 재학습 0 유지 |
| 절대 ℃ MSE가 더 좋은가? | M pooled 0.834330 < C 0.896731; 가을 0.516946 > C 0.465330 | 새 3-seed M·최종 공식 성능 | 가을 주평가 기준 승격하지 않음; 계열 전체 실패라고 기록하지 않음 |
| 균등 가중치가 더 좋은가? | R intact 가을 악화, 고정 outage 6031행 0.224904 < C 0.465796 | 여러 episode·seed·공식 결측 일치 | 결측 조건부 효과의 후속 가설만 보존 |
| 절대 프로파일 tree가 보완하는가? | 단독 주평가 0.710383; 고정 절반 0.518502, pooled 절반 0.823186 | 학습된 inner-OOF stacking·추가 seeds | B 정의된 분기 완료, 추가 fitting 금지 |
| 결과가 독립 확증인가? | 같은 69850행·같은 분모의 통제된 비교 | fresh holdout, 다년 autumn | historical development로 명시 |
| 점수가 얼마나 오르는가? | 기존 C 공식 0.455143℃ / 27.622418점 | 새 후보 공식 RMSE·점수 | 예상 점수 미산정; CSV/upload 0 |
| 비용을 줄였는가? | C9 재사용, A9+B3 실행, 추가6+6 미실행, wall합256.484초 | 미실행 fits의 실제 wall | fit 단위 절약만 확정, 시간 절약을 지어내지 않음 |
| 데이터 규정과 재현이 맞는가? | 배포 observations만, target-free features, hash/replay/27pytest/Ruff/A34+B30QA PASS | 최종 fulltrain·6시간 패키지 재현 | 이 실행은 제출 패키지 아님; fulltrain/official 단계는 별도 |

실측값과 출처는 [통합 report-source.md](report-source.md), [A result.json](result.json), [B result.json](../p2_physical_profile_tree_20260905_v2/result.json)에 연결했다. 과거 Public 기반 bin17 계보의 높은 숫자를 이번 clean 모델의 합법 기준선으로 사용하지 않았다.
