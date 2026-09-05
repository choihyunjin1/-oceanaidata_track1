# 남은 불확실성과 재진입 조건

| 항목 | 이번에 확인한 사실 | 아직 입증하지 못한 것 / 다음 조건 |
|---|---|---|
| P1 depth repair | 연도 의존 계약 제거, pooled F1 감소 | 계약 수정만으로 공식 향상; 과거 deep 부품은 동일 split/key/purge 재현부터 |
| P1 고정 decoder | A보다 일부 회복하나 control 미달 | 사후 계절별 ON/OFF 조합의 일반화; 이번에는 추가 탐색하지 않음 |
| P2 objective | 가을 intact control 승리, outage에서는 후보 이득 | 실제 배포 결측 분포에 대한 일반화; target-free 상태 rule·inner OOF·별도 episode 대조 필요 |
| P2 tree | pooled fixed50:50 개선과 primary 악화가 공존 | 전체 pooled로 primary를 사후 교체하지 않음; adaptive stack 학습용 inner OOF 부족 |
| P3 meta | bias 감소가 pooled RMSE 개선을 보장하지 않음 | 계절 편향 이동의 배포 일반화; 추가 자유도 자동 확장 금지 |
| P3 episode/seed | 두 새 완성 정책 모두 legacy no-op보다 악화 | 모든 seed/가중법이 무가치라는 일반명제는 아님; 이번 고정 후보만 완료 |
| 평가 | 동일 historical 행 paired 비교 | fresh holdout·확증적 유의성·정확한 예상 공식 점수 |
| 개발 환경 | 안내 7개 1,061→217줄, checker 22tests/Ruff PASS | 모든917개 scripts의 미사용 전수 증명·전체 프로젝트 QA·시간 절약률 |
| 검증 cache | 실제 실행 수/hash/source 변동/실패 처리 검사 | 임의 fixture·외부 상태 전체 의존성 검출; 해당 검사는 cache 사용 금지 |
| 재현 | 문제별 저장 모델 replay와 해시 근거 | 모든 모델 재학습의 bitwise 일치·신규 최종 ZIP 6시간 완전 인증 |

원본·관측 정답을 인터넷에서 찾거나 검증용으로 가져오는 것은 재진입 방법이 아니다.
규정, frozen artifact, 기존 dirty 변경을 유지한다. 커밋/푸시와 새 무제한 탐색은 별도 요청 범위다.
