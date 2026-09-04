# Claim–Source Ledger

| ID | 주장 | 근거 | 유형 | 상태 |
|---|---|---|---|---|
| C1 | 로컬 개선과 공식 개선의 부호가 자주 뒤집혔다 | `20260825_OFFICIAL_SCORE_RECONCILIATION.json`, `local_official_calibration.json` | 로컬 공식점수 원장 | 확인 |
| C2 | P2 OAS alpha 0.00/0.10/0.20/0.40 공식 RMSE는 0.535727/0.507628/0.483661/0.445147이다 | 공식 결과 원장과 `official_score_receipt.json` | 공식 관측 | 확인 |
| C3 | 저장된 alpha 0.10/0.20/0.40 예측은 재생성과 최대오차 3.56e-15 이내다 | `metric_geometry.json` | 실행 결과 | 확인 |
| C4 | alpha 0.50의 조건부 RMSE 상한은 0.439881이다 | `metric_geometry.json`, `metric_geometry.py` | 수학·실행 | 확인 |
| C5 | 반복 적응 제출은 leaderboard holdout을 과적합할 수 있다 | Blum & Hardt 2015; Dwork et al. 2015 | 1차 문헌 | 확인 |
| C6 | 예측 결합은 squared-error 하에서 개별 예측보다 나을 수 있다 | Bates & Granger 1969 | 1차 문헌 | 확인 |
| C7 | 공식 public/private 분할 계약은 공개 FAQ에서 확인되지 않았다 | 공식 홈페이지·FAQ 2026-08-28 확인 | 공식 출처 부재 | 미확정 |
| C8 | alpha 0.50이 최종 비공개 평가에서도 개선한다 | 직접 근거 없음 | 추론 | 주장하지 않음 |
| C9 | alpha 0.50 공식 Public RMSE는 0.431252이고 이전 최고보다 0.013895 ℃ 개선됐다 | OCN-02 채점완료 화면, `official_score_receipt.json` | 공식 관측 | 확인 |

## 문헌 링크

- https://proceedings.mlr.press/v37/blum15.html
- https://pubmed.ncbi.nlm.nih.gov/26250683/
- https://rss.onlinelibrary.wiley.com/doi/10.2307/2981683
- https://doi.org/10.1057/jors.1969.103
- https://oceanaidata.org/
- https://oceanaidata.org/api/faqs
