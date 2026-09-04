# Claim–source ledger

| Claim | Source | Type | Confidence / caveat |
|---|---|---|---|
| 현재 P2 public best는 26.611283점, RMSE 0.535727℃ | `reports/finite_horizon_submission_decision_20260827_v1/report-source.md` | 공식 결과 receipt | 높음 |
| 현재 P3 public best는 24.066167점, RMSE 0.583892m | 같은 receipt | 공식 결과 receipt | 높음 |
| P2 U 대비 OAS 10% 혼합 총 ΔRMSE −0.007782℃ | `artifacts/p2_oas_conditional_profile_20260827_v3/result.json` | 로컬 실행 결과 | 높음, 노출 블록 |
| 물리 투영 후 총 ΔRMSE −0.008060℃ | 같은 JSON | 로컬 실행 결과 | 높음, 노출 블록 |
| KST-day bootstrap 90% CI [−0.012785, −0.003176] | 같은 JSON | 로컬 통계 | 표본 변동만 반영; adaptive selection 미반영 |
| 2024 Sep–Oct 9개 주·3개 층 모두 α=.1 개선 | 같은 JSON | 로컬 분해 결과 | 공식 gap과 계절 유사, 그러나 fresh holdout 아님 |
| 2025 Jul–Aug는 α=.1 악화 | 같은 JSON | 반증 | regime risk의 핵심 증거 |
| 경계 정합 연주기 prior는 두 블록 모두 악화 | `artifacts/p2_boundary_registered_prior_20260827_v1/result.json` | 반증 실험 | 높음 |
| 부분 관측 다변량 프로파일을 교차 공분산으로 조건부 복원 가능 | https://arxiv.org/html/2608.05376v1 | 2026 arXiv 사전논문 | 방법 근거만 사용; 효과 크기 전이 금지 |
| ImputeFormer는 저랭크 projected attention과 block-missing 실험 제공 | https://arxiv.org/html/2312.01728v3 ; https://github.com/tongnie/ImputeFormer | 논문 + 공식 구현 | 교통 센서 도메인 불일치 |
| LSTI는 장·단기 imputer를 meta weighting으로 결합 | https://openreview.net/forum?id=9NVJ0ZgEfT | TMLR/OpenReview | 공개 평균 개선율 전이 금지 |
| CSDI는 조건부 diffusion imputation 기준 | https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html | NeurIPS 2021 | 로컬 유사 family는 이미 실패 |
| TSI-Bench는 다양한 모델·데이터·결측 패턴의 대규모 비교를 제공 | https://arxiv.org/abs/2406.12747 ; https://github.com/WenjieDu/Awesome_Imputation | 논문 + 공식 benchmark | 정확한 missingness 계약 필요성의 근거 |
| 반복 leaderboard 적응은 선택 편향 위험 | https://proceedings.mlr.press/v37/blum15.html ; https://www.jmlr.org/papers/v11/cawley10a.html | 동료검토 논문 | 직접적인 대회 점수 보정식은 아님 |
| 신뢰할 수 있는 참가자 공개 코드를 찾지 못함 | GitHub/web exact-name searches, 2026-08-27 | 검색 결과 | 부재의 증명이 아님 |

## Search log / stopping reason

- 공식 대회 홈페이지, 정확한 파일명·문제명·팀명, GitHub repository 검색을 수행했다.
- 시계열 imputation, long block missing, conditional functional profiles, diffusion, low-rank attention, ocean reconstruction을 우선 탐색했다.
- 기존 로컬 실패 계보와 대조해 즉시 재실행 가치가 낮은 family를 제거했다.
- 새로운 구조 후보 하나를 실제 로컬 반증 실험까지 완료했으며, 추가 문헌의 한계효용보다 배포 경로 재현의 정보가치가 커져 검색을 중단한다.
