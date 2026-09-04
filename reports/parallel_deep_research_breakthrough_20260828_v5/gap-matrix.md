# V5 연구 gap matrix

| 문제 | v4에서 닫힌 실패 | 새 후보 | 구조적으로 다른 이유 | 가장 큰 반증 | 남은 검증 |
|---|---|---|---|---|---|
| P1 | 83개 proposal 위 ranker는 qualification 양성 1건으로 support 실패 | TS2Vec normal-prototype proposal generator | 양성 event 없이 무라벨 timestamp representation을 학습해 proposal bank 자체를 새로 만듦 | 원 TS2Vec anomaly protocol은 point anomaly·SPOT·point adjustment 중심 | 실제 long-event qualification에서 새 event 2건 이상, FP/day와 F1 동시 개선 |
| P2 | static CMFPCA와 public-only heave는 각각 동역학 부족·활성 부족 | BayOTIDE형 dynamic low-rank SSM | public T/S 관측으로 결측 기간 latent state를 계속 갱신하고 uncertainty 제공 | 논문은 random missing 비중이 크며 61일 target-channel blackout과 다름 | actual-depth regime, joint T/S block mask, strongest common OOF에서 sealed 검증 |
| P3 | ERA5 expert router는 1/3 outer support와 I4 magnitude/coverage 실패 | TimeXer형 direct 6-lead | expert 선택 대신 모든 anchor로 endogenous/exogenous representation을 직접 학습 | 3-station 소표본 Transformer 과적합, 미래 외생변수 부재 | 과거 외생만 사용한 3-fold×3-seed 직접 residual의 장리드 안정성 |

## 즉시 제출 가치와 새 구조 연구의 분리

- 기존 P2 seasonal OAS alpha=0.40 파일은 새 구조가 아니라 이미 공식 alpha=0.10→0.20 상승 방향을 잇는 **기술적 공식 probe**다.
- 새 구조 세 후보는 아직 prediction CSV가 아니며 공식 제출 승인이 없다.
- OAS40은 파일·lineage QA가 끝났지만, 업로드는 사용자가 정확한 파일과 SHA를 다시 승인할 때만 가능하다.
- 새 구조의 문헌상 개선률을 공식 점수로 환산하지 않는다.
