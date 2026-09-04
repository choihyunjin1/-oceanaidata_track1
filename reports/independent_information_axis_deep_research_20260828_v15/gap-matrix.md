# Evidence gap matrix — final

| 문제 | 이번에 검증한 독립축 | 확인된 증거 | 남은 공백 | 다음 표적 | 현재 제출 |
|---|---|---|---|---|---|
| P1 | temporally fused low-rank+sparse decomposition | layer별 coverage와 joint-time coverage가 다름; window 0, exact no-op | asynchronous layer에서 공통 상태를 어떻게 결합할지 | sensor별 event proposal 후 station-state 결합 | 금지 |
| P2 | public T/S spline → supervised rank-1 L2–L4 residual | pooled `-0.004799°C`, bootstrap CI 전부 음수, 2 folds·3 layers 개선, axis 독립 | Nov-Dec `+0.008592°C`; exact α50 OOF 부재 | train-only support/regime veto의 별도 사전등록 | 금지 |
| P3 | aligned wind power·phase-speed proxy·memory correlation | 18/24h `-0.011197m`, G/I 개선 | 17 cases뿐, S station 회귀, 38–40% feature 결측 | 새 independent shadow + pressure/spectral memory family | 금지 |
| 공통 | prediction commitment + no-retry gate | independent QA PASS, 공식/CSV/업로드 0 | exposed-fold selection bias | 다음 cycle 전에 새 contract와 stopping rule 고정 | 0건 |

## 닫힌 축

- P1: e125 checkpoint cell, 단순 threshold/router, synchronous multi-layer RPCA.
- P2: alpha 0–0.725 미세조정, 무감독 CMFPCA, 이번 fixed rank-1 correction의 결과 기반 shrink/재튜닝.
- P3: alpha 수축, ERA5-Hs², 단순 forcing analog, 이번 10+10 wind-wave memory family의 station 사후 선택.
