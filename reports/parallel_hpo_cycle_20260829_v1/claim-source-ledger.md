# Claim-to-source ledger — 2026-08-29 parallel HPO cycle

접근일: 2026-08-29

| ID | 주장 | 근거 | 출처/저자 | 발행·갱신 | URL | 신뢰도·비고 |
|---|---|---|---|---|---|---|
| M1 | 무작위 탐색은 일부 hyperparameter만 중요할 때 격자보다 계산 예산을 효율적으로 쓸 수 있다. | 논문 abstract와 실험·이론 결론 | “Random Search for Hyper-Parameter Optimization”, Bergstra & Bengio, JMLR | 2012 | https://www.jmlr.org/papers/v13/bergstra12a.html | 높음. P1 성능 보장은 아니며 탐색 설계 근거만 제공. |
| M2 | `Sobol.random_base2(m)`는 `2^m`개 점을 생성하며 sequence balance property를 보존한다. | 공식 API 문서 | SciPy community | 접근 2026-08-29 | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.random_base2.html | 높음. P1의 32점 설계와 직접 대응. |
| M3 | 유한 validation criterion을 최적화하면 model-selection overfit과 후속 평가 selection bias가 생길 수 있다. | 논문 abstract와 일반화 범위 | “On Over-fitting in Model Selection…”, Cawley & Talbot, JMLR | 2010 | https://www.jmlr.org/papers/v11/cawley10a.html | 높음. P1/P2의 selection-confirmation 분리 근거. |
| M4 | CatBoost ordered boosting은 prediction shift와 특정 target leakage를 줄이기 위해 제안됐다. | 공식 학회 논문 abstract | “CatBoost: unbiased boosting with categorical features”, Prokhorenkova et al., NeurIPS | 2018 | https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html | 높음. P3 구조 선택 근거이며 성능 보장은 아님. |
| M5 | CatBoost의 `SymmetricTree`와 `Depthwise`는 서로 다른 grow policy다. | 공식 parameter-tuning 문서 | CatBoost | 접근 2026-08-29 | https://catboost.ai/docs/en/concepts/parameter-tuning | 높음. 문서는 각 policy를 설명하지만 비호환 오류의 직접 근거는 소스 M6. |
| M6 | CatBoost v1.2.10에서 non-symmetric grow policy는 `Plain` boosting만 허용하며 `Ordered + Depthwise`는 오류다. | `catboost_options.cpp` 705–707의 실행 제약과 오류 문자열 | CatBoost v1.2.10 source | v1.2.10 / 접근 2026-08-29 | https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/private/libs/options/catboost_options.cpp | 매우 높음. P3 terminal 오류와 문자열이 일치. |
| M7 | Successive Halving은 유망한 hyperparameter arm에 점진적으로 더 많은 자원을 배분한다. | 논문 abstract와 algorithm 설명 | “Non-stochastic Best Arm Identification and Hyperparameter Optimization”, Jamieson & Talwalkar, PMLR | 2016 | https://proceedings.mlr.press/v51/jamieson16.html | 높음. P3 rung 설계 근거. |
| L1 | P1은 36 fits 뒤 pooled ΔF1 `+0.000565637`로 `NO_GO_PRECONFIRM`이다. | aggregate, gate receipt, independent QA 28/28 | local sealed artifacts + `reports/p1_mstcn_sobol_hpo_20260829_v1/report-source.md` | 2026-08-29 | 저장소 로컬 경로 | 높음. official interface 0행. |
| L2 | P2는 84 PLS fits와 pooled ΔRMSE `-0.002041992 °C`를 얻었지만 fold·eligibility gate로 `NO_GO_CLOSE_FAMILY`다. | result, prediction commitment, independent QA | local sealed artifacts + `reports/p2_nested_pls_capacity_grid_20260829_v1/report-source.md` | 2026-08-29 | 저장소 로컬 경로 | 높음. official interface 0행. |
| L3 | P3는 74 successful fits 뒤 75번째 `challenger_37`에서 기술 실패했고 과학적 결론은 없다. | static preflight, attempt lock, frozen loop reconstruction, terminal exception, CatBoost source | `reports/p3_catboost_ordered_hpo_20260829_v1/failure-report.md` + M6 | 2026-08-29 | 저장소 로컬 경로 및 M6 URL | 높음. OS-level access audit는 없고 code-path/receipt 증거. |

## 반증·적용 한계

- M1의 random-search 결과는 Sobol sequence의 우월성 또는 P1 task의 개선을 직접 증명하지
  않는다. P1 설계는 M1과 M2를 함께 사용한 engineering 선택이다.
- M3은 nested/held-out 평가의 필요성을 지지하지만 현재 gate 수치 자체를 정당화하지
  않는다. gate는 실험 전 고정된 프로젝트 규칙이다.
- M4의 CatBoost 장점은 P3 v1의 grid correctness를 보장하지 않는다. M6이 오히려 v1의
  조합 오류를 직접 반증한다.
- P1/P2 local proxy와 공식 leaderboard 성능의 관계는 이번 사이클에서 측정하지 않았다.
  따라서 local 음성 결과를 공식 점수 하락이나 상승으로 환산하지 않는다.
