# P3 v2 가설-실측-다음 판단

| 가설/요구 | 실제 근거 | 판단/남은 한계 |
|---|---|---|
| A 직접 SSE 저자유도 결합 | simplex +0.0003626m, bias +0.0003242m | no-op 유지. 낮은 자유도만으로 계절 이동을 해결하지 못함 |
| 평균 bias를 제거하면 점수도 향상 | signed error 거의0으로 줄었으나 겨울 개선·2025H1 악화 | 평균편향0과 RMSE 개선은 다름. 사후 fold 조합 금지 |
| 두 seed 평균이 기존 clean보다 좋음 | control +0.0042333m; 세 fold·여섯 lead 악화 | 이번 고정 two-seed 완성정책 채택 안 함. seed 사후선택 안 함 |
| 긴 사건 반복 anchor 완화가 대표성 개선 | seed-matched control 대비 +0.0112194m | 이번 고정 sqrt 사건가중치 채택 안 함. exponent 재튜닝 안 함 |
| 메타 검증의 미래 누출 방지 | actual target-ready 시간, 정점78h/episode0, 49→128 과거-only 재적합 | hard integrity PASS. reused historical은 fresh evidence가 아님 |
| 완전 재학습·재로딩 가능 | historical18 CatBoost 신규학습, 신규18 saved models fresh-process 예측차이0 | 실험 재생 가능. GPU 재학습의 bitwise determinism을 보장한 것은 아님 |
| 최종 full4+router1 준비 | 전용 config/runner 및 synthetic tests5 PASS | 승자 없음으로 실행0, official/CSV/upload0. 새 제출물 아님 |
| 공식 점수 상승 | 이번 P3 A/B 공식 제출0 | 예상 공식 점수 미산정. 기존 clean 정책을 보존 |

새 가설이나 예산 확대 없이 승인된 P3-A/P3-B 두 분기를 모두 종료했다. 다음 연구는 다른 분기/새 계약으로 정의해야 하며 이 결과나 attempt lock을 변경하지 않는다.
