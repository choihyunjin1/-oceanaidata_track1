# P1 SupCon/top-k 스크린 gap matrix

| 질문 | 이번 증거 | 판정 | 다음 연구 경계 |
|---|---|---|---|
| 실제 이벤트 support 부족이 주 병목인가 | 116~213 events, 5 types, 30~48 cells인데도 모든 창 큰 폭 악화 | 이 실행에서는 아님 | 합성/지원량 확대보다 proposal 품질을 직접 다룬다 |
| 이벤트 균형 SupCon이 incumbent를 넘는가 | Q2/Q3/Q4 ΔF1 = -0.1750/-0.1651/-0.1560 | exact recipe NO_GO | seed/epoch만 늘린 확인 금지 |
| 학습-only top-k F1 보정이 transport되는가 | 학습 최적 rate가 주로 2% cap, holdout precision 4.5~15.4% | NO_GO | 분포 이동 대응 위험 제약 또는 residual-only target 필요 |
| anchor 안전성은 지켜졌는가 | 제거 0행 | PASS | anchor-union 유지 |
| 한 station에만 변화를 몰았는가 | pooled 최대 share 0.5844 | PASS | 실패 원인은 집중도가 아니라 낮은 precision |
| 3-seed 확인 가치가 있는가 | 모든 과학 게이트 실패, 효과 크기도 큼 | 없음 | 별도 구조가 사전등록되기 전 실행 금지 |

