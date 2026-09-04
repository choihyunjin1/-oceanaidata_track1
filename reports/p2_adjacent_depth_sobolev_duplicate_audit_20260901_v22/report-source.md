# P2 v22 adjacent-depth Sobolev duplicate audit

## 결론

`p2_adjacent_depth_sobolev_duplicate_audit_20260901_v22`는 **0-fit 의미 중복으로 종료**한다. 새로 제안된 “인접 목표 수심의 유한차분을 값 손실과 함께 맞추는” 축은 이미 `p2_ts_continuous_depth_challenger_20260827_v1`에서 고정 가중치 `0.25`로 실제 실행됐다. v13 DeepSets에 같은 손실을 붙이는 것은 모델 외피만 바꾸는 것이며 새 과학 질문이 아니다.

## 정확한 중복 근거

- 기존 구현은 `src/p2_restore/ts_continuous_depth_challenger_20260827_v1.py:443-449`에서 예측값과 정답의 인접 수심 차분을 각각 계산한 뒤 그 차이의 제곱평균을 구성한다.
- 기존 runner는 `scripts/run_p2_ts_continuous_depth_challenger_20260827_v1.py:378`에서 `vertical_difference_weight=0.25`를 봉인했다.
- 기존 terminal 결과는 pooled temperature RMSE를 `1.0180704839→1.0152588181 C`로 낮췄지만, 2025 Jul-Aug block은 `+0.0045115872 C` 악화했고 최종 판정은 `INCONCLUSIVE`였다.
- 따라서 v22에서 가중치나 외부 architecture만 바꾸는 것은 이미 노출된 결과를 이용한 재탐색이며 허용하지 않는다.

## 운영 봉인

- observations 및 historical labels: 0행
- fit / prediction: 0 / 0
- attempt lock / candidate artifact: 0 / 0
- official / hidden / submission CSV / upload: 모두 0
- runner는 만들지 않았다. v22 ID는 재사용하거나 실행하지 않는다.

## 다음 질문

다음 후보는 출력의 수심 미분이 아닌 독립 축이어야 한다. 저장소 전체 exact/semantic audit를 통과한 뒤 단일 고정 설정만 v23으로 사전등록한다.
