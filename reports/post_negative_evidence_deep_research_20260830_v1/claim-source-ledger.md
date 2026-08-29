# Claim–source ledger

| ID | 주장 | 근거 | 구분 | 신뢰도와 경계 |
|---|---|---|---|---|
| C01 | P3 v2는 모델 성능 실패가 아니라 confirmation schema 실패다. | 기준 커밋의 P3 failure report 및 negative evidence registry | 로컬 직접 증거 | 높음. 선택 구간 개선이 confirmation 일반화를 보장하지는 않음. |
| C02 | P3 후보 탐색을 다시 하지 않고 고정 winner만 확인하는 편이 selection bias와 계산 낭비를 줄인다. | [Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html), 로컬 138-fit 완료 증거 | 문헌 + 설계 추론 | 높음. 확인 결과는 아직 없음. |
| C03 | schema·feature contract는 모델 품질과 독립적으로 검증해야 할 ML system 축이다. | [ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/), [Data Validation for ML](https://proceedings.mlsys.org/paper_files/paper/2019/file/928f1160e52192e3e0017fb63ab65391-Paper.pdf) | 1차 문헌 | 높음. 구체적인 required columns는 로컬 pipeline에서 도출. |
| C04 | nonparanormal/Gaussian copula는 알려지지 않은 단조 marginal transform 아래의 의존구조와 예측을 모델링할 수 있다. | [Liu et al. 2009](https://www.jmlr.org/papers/v10/liu09a.html), [Dey & Zipunnikov 2018](https://www3.stat.sinica.edu.tw/sstest/j28n2/j28n219/j28n219.html) | 1차 문헌 | 높음. P2 성능 개선은 미입증. |
| C05 | P2 train-only support audit은 copula 실험의 실행 가능성만 지지하고 성능은 지지하지 않는다. | 기준 커밋의 P2 support audit | 로컬 직접 증거 | 높음. query/test support는 미검사·미승인. |
| C06 | 비선형 inverse marginal에서는 latent conditional mean의 단순 역변환이 일반적으로 원척도 conditional mean과 같지 않다. | 조건부 기대값과 비선형 변환의 수학적 성질 | 분석적 추론 | 높음. 그래서 deterministic quadrature를 사전등록. |
| C07 | supervised contrastive learning은 class-aware representation을 학습하며 time-series 변형도 존재한다. | [Khosla et al. 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html), [Deldari et al. 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25863), [TS2C 2023](https://proceedings.mlr.press/v225/noroozizadeh23a.html) | 1차 문헌 | 높음. 해양 anomaly 전이는 추론이며 low-fidelity screen이 필요. |
| C08 | F1은 비분해 가능하며 conditional-probability ranking 또는 calibrated linear-fractional surrogate로 직접 다룰 수 있다. | [Natarajan et al. 2016](https://proceedings.mlr.press/v48/natarajan16.html), [Bao & Sugiyama 2020](https://proceedings.mlr.press/v108/bao20a.html) | 1차 문헌 | 높음. 구현 안정성과 데이터 전이는 미입증. |
| C09 | random CV로 chronological validation을 대체하면 안 된다. | [Bergmeir et al. 2018](https://robjhyndman.com/publications/cv-time-series/) 및 로컬 nonstationary split 정책 | 문헌 + 보수적 설계 | 중간~높음. 논문은 잔차 조건에 따라 표준 k-fold의 유효성을 논함; 본 설계는 더 보수적인 chronological split을 유지. |
| C10 | 현재 추천 우선순위는 P3, P2, P1 순이다. | 로컬 evidence strength, 예상 fit 수, 실패 분류, 위 문헌 | 의사결정 추론 | 중간~높음. 공식 점수 상승량 예측이 아니라 증거/비용 기준 우선순위. |

## 금지된 과장

- 문헌 성능을 P1/P2/P3의 예상 공식 점수로 환산하지 않는다.
- 선택 구간의 P3 RMSE 개선을 confirmation 또는 공식 test 개선으로 부르지 않는다.
- P2 support audit을 모델 성능 PASS로 부르지 않는다.
- P1 관련 타 도메인 결과를 해양 anomaly 전이 증거로 부르지 않는다.
- 정확한 실패 조합을 전체 계열의 불가능으로 확대하지 않는다.
