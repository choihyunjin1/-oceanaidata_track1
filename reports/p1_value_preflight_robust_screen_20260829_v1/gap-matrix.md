# Evidence gap matrix

| 질문 | 현재 증거 | 빈틈 | 보수적 처리 | 다음 검증 |
|---|---|---|---|---|
| 정적 신규성 검사로 중복 연구를 막을 수 있는가 | 12개 closed family registry, Jaccard 기반 중복 검사 | 태그가 부실하면 우회 가능 | mechanism과 intervention layer를 모두 요구 | registry를 모든 terminal 실험 뒤 갱신 |
| 10 epoch 결과가 full 결과를 예측하는가 | Q3/Q4 방향 불일치를 빠르게 탐지 | fidelity rank correlation 미측정 | 양수여도 30 epoch rung 필수 | 10/30/full 쌍을 누적해 rank correlation 계산 |
| 환경균형이 worst-slice를 개선하는가 | Q3 전 station 악화, Q4 S-ORS만 개선 | full GroupDRO 미실행 | 균형 replay family만 닫음 | 새 objective는 별도 신규성·비용 심사 |
| pooled 미세 양수를 공식 가치로 볼 수 있는가 | pooled +0.0000426, additions 0/23 true | 공식 점수 변환관계 없음 | 공식 점수로 환산 금지 | 승인된 probe 뒤 local-official ledger 갱신 |
| 다음 time-frequency 축이 안전한가 | RAINCOAT 1차 문헌 | target covariate·transductive 허용 경계 | 실행 전 leakage preflight | offline/causal 및 test-covariate 계약 분리 |
