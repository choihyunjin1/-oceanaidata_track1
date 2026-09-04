# Claim-source ledger

| 주장 | 근거 | 출처 | 신뢰도 | 반증·한계 |
|---|---|---|---|---|
| F1은 비선형이며 calibrated probability의 최적 threshold는 최적 F1의 절반이다 | 정의·정리·Corollary 1 | Lipton et al. 2014, PMC4442797 | 높음 | P1 cell은 calibrated probability가 아니라 binary set |
| 반복 적응 제출은 leaderboard holdout 과적합 위험이 있다 | sequential adaptive leaderboard 분석 | Blum & Hardt 2015, PMLR 37 | 높음 | 본 대회 scorer가 Ladder 방식이라는 증거는 없음 |
| 유한 검증면을 반복 최적화하면 selection criterion 자체가 과적합될 수 있다 | CV criterion variance와 selection bias 실증 | Cawley & Talbot 2010, JMLR 11 | 높음 | 구체적인 이 대회 bias 크기는 알 수 없음 |
| affine prediction line의 squared loss는 이차식이다 | least-squares/convex quadratic 구조 | Boyd & Vandenberghe 2004 | 높음 | PAVA projection은 alpha에 대해 완전 affine가 아님 |
| forecast combination은 보완 오차가 있을 때 가치가 있다 | forecast combination 원리 | Bates & Granger 1969 | 높음 | P3 bases는 높은 상관과 공통 alpha 축을 보임 |
| source-target divergence가 크면 transfer 보장이 약해진다 | domain adaptation generalization bound | Ben-David et al. 2010 | 높음 | divergence만으로 개별 모델 실패를 확정할 수 없음 |
| P1 narrow cell은 두 historical drift 사건에서만 확인됐다 | frozen OOF disagreement audit | local artifacts listed in report | 중간 | 사건 단위 2개로 독립성이 약함 |
| P1 narrow cell은 full test로 운반되지 않았다 | 3-seed e150 exact replay 뒤 추가 1행·성분 길이 1 | p1 full-data result.json | 높음 | hidden label을 읽지 않은 support/coverage 판정이며 precision 자체를 측정한 것은 아님 |
| P2 alpha725 strong guarantee는 실패했다 | actual five-vector geometry upper 0.435846 > 0.421252 | p2_metric_geometry_after_alpha50.json | 높음 | conditional quadratic center는 여전히 개선을 예측 |
| P3 existing independent basis가 없다 | prediction-only orthogonalization + historical OOF bootstrap | p3_orthogonal_basis_audit.json | 높음 | future forcing 등 아직 없는 정보는 미검증 |
