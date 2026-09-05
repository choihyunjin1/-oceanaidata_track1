# P2 실제 수심 profile copula 잔차 — 내부 주평가 개선, 추가 결측 위험

결론: 적격 C3의 고정 full 잔차 보정은 가을 intact RMSE를 **0.488284→0.472272℃**로 낮췄다(Δ −0.016012℃, 26,273행). 전체 69,850행도 0.859250→0.853508℃로 개선했다. 그러나 추가 가을 7/14일 결측은 모두 악화했다. 반복 내부 평가에서 얻은 **정보가치 후보**이지 안정성 개선·공식 최고점·최종 패키지 적격 보장은 아니다. full 정책은 사전등록 순서에서 선택했으며 결측 결과를 보고 half로 바꾸지 않는다.

Canonical 수치: [result.json](result.json). [독립 QA](independent-qa.json) 367 checks PASS, 합성 pytest 10 PASS/Ruff PASS, [새 프로세스 replay](replay.json) 전체 69,850 historical correction exact PASS. 공식 입력/CSV/upload는 **이 연구 실행에서 모두 0**이다. 이후 기준 재생성·후보 materialization은 별도 ID와 권한으로 분리한다.

## 비교와 분모

| 동일 행 평가 | n | C3 RMSE ℃ | full RMSE ℃ | 고정 half RMSE ℃ |
|---|---:|---:|---:|---:|
| 가을 intact 주평가 | 26,273 | 0.488284 | 0.472272 | 0.476557 |
| 전체 intact | 69,850 | 0.859250 | 0.853508 | 0.855147 |
| 여름 intact | 26,693 | 1.286209 | 1.282702 | 1.283780 |
| 겨울 intact | 16,884 | 0.260733 | 0.255877 | 0.257602 |
| 자연 T5/S5 결측 | 17,165 | 0.277259 | 0.272372 | 0.273933 |
| 자연 T5/S5 모두 관측 | 52,685 | 0.976632 | 0.970385 | 0.972153 |
| 가을 자연 T5/S5 결측 | 212 | 0.549145 | 0.537412 | 0.534294 |
| 가을 자연 T5/S5 모두 관측 | 26,061 | 0.487758 | 0.471705 | 0.476058 |
| 기존 개발 가을 14일 결측 | 6,031 | 0.446773 | 0.413989 | 0.427682 |
| 추가 가을 7일 결측 | 3,018 | 0.848478 | 0.881242 | 0.859038 |
| 추가 가을 14일 결측 | 6,017 | 0.731300 | 0.754780 | 0.740367 |

RMSE는 각 scope의 sqrt(SSE/n)이며 fold RMSE 평균이 아니다. 추가 autumn_3d는 총 1,294행 중 4행이 공용 온도 2개 미만이 되어 **전체 시나리오 SUPPORT_BLOCKED/unscored**다. 1,290행으로 분모를 줄이지 않았고 실제 공식 입력 결함도 아니다. 겨울 추가 3개 outage는 T5/S5가 원래 결측이어서 조작이 no-op이다. 겨울 모델 간 차이는 intact 보정 차이지 새 결측 대응 효과가 아니다. 전체 9개 추가 episode와 1개 기존 개발 episode 수치는 result에 보존했다.

## 한 가지 변경 및 중복검사

- 기준: 9월 5일 배포 관측 scratch raw v23 blockmask의 3-seed 평균 C3. historical 3fold×3seed 모델 **9개 해시/예측 exact 재사용**. first-seed 성적과 혼동하지 않는다.
- 변경: C3 예측·실제/명목 목표 수심·실제−명목 공용 보간 차이·실제 수심 gradient·공용 T/S contrast 등 11개 조건부 특징으로 **truth−C3 잔차**의 Gaussian-rank 조건부 평균을 학습했다. 목표 T2/3/4 및 S2/3/4는 특징 생성 전에 제외한다.
- 각 outer train에만 empirical CDF와 Ledoit–Wolf covariance를 적합했다. 결측 특징의 공분산 적합은 latent 0 대입, 추론은 해당 조건을 주변화한다. 15-node Gauss–Hermite 적분으로 잔차 조건부 평균을 복원한다. 실제 수심 누락은 명목 수심 fallback, 중복 수심은 공개 온도 평균, 보간 바깥은 clamp/gradient 0으로 사전 정의했다.
- 옛 `p2_gaussian_copula_conditional_mean_20260830_v2`는 current_blend50/alpha50 계보 예측, 계절별 shrinkage 선택을 사용했으므로 **코드 결과·모델·계수를 재사용하지 않았다**. copula 아이디어만 참고했다. 옛 global standardized OAS와 달리 적격 C3 잔차/실제 수심 조건이다.
- 실패한 `p2_physical_profile_tree_20260905_v2` LightGBM 및 `p2_missingness_conditional_3seed_20260905_v3` 결측 OR routing을 재실행하지 않았다. 새 모델은 tree도 conditional R 전환도 아니다.

## 실행·증거·제약

- [config](../../configs/experiments/p2_profile_copula_residual_20260905_v4.json), [runner](../../scripts/run_p2_profile_copula_residual_20260905_v4.py), [preregistration seal](preregistration-seal.json).
- 현재 파일 봉인만 한 것이 아니라 기존 `p2_score_repair_20260905_v1/result.json.manifest`의 원 runner/config와 6개 dependency SHA도 현재와 전부 일치함을 read-only 대조했다. `alpha40`라는 역사적 모듈에서 import된 OAS 함수는 C3 scratch training 경로에서 호출하지 않는다. 별도 재생성 가드는 해당 함수를 명시적으로 차단한다.
- 3 deterministic joint residual fits; baseline의 seed 3개를 평균한 뒤 잔차 모델은 fold당 하나다. 같은 deterministic 모델을 3seed라고 세지 않았다. 승인 상한 6 historical fits 중 3 사용, 신규 backbone/full fits 0.
- train/validation 행수: 가을 133,948/26,273, 여름 136,551/26,693, 겨울 149,384/16,884. 동일 7일 purge, zero-hour feature dependency, key 삭제 0. 복원 문제의 train은 허용된 전체 배포 시계열에서 outer±purge를 제외하며 인과 예보 모델이 아니다.
- 총 실행 35.390초, covariance fit 합계 1.111초. CPU2/GPU0. 30분은 운영 계획이며 runner hard timeout은 아니다.
- 잔차 학습의 C3 train 예측은 **in-sample**이다. 따라서 보정 학습에는 과소추정된 baseline 학습 오차가 들어갈 수 있다. outer label은 fit에 들어가지 않지만, 이 반복 outer 평가가 새 확증 데이터로 바뀌지는 않는다.
- 모델 reload/69850행 saved-feature replay는 **빈 03_model에서 기준 모델을 다시 학습한 검사와 다르다**. 본 연구 결과의 기준 재생성 필드는 당시의 `NOT_RUN_ROOT_COORDINATES_SEPARATE_SCRATCH_TEST`로 보존한다. 이후 v4/v5의 자체 SHA 읽기 가드 기술적 실패를 보존하고, 새 [v6 빈 모델 재생성](../p2_clean_regeneration_20260905_v6/report-source.md)에서 scratch3fits→새 PID 공식 답안→세 번째 PID 전체 CSV replay와 기존 C SHA exact/27-check QA가 실제 PASS했다. 여기 기존 result를 사후 수정하지 않는다. [별도 full 후보 생성](../p2_profile_copula_deploy_20260905_v4/report-source.md)은 이 재생성 C3에서 1fit 보정과 전체 공식 답안 replay/28-check QA를 마쳤으며 업로드 전 상태다.
- 공식 예상 점수는 계산하지 않았다. 공식 반환 점수 역산·계수 fitting을 하지 않았고 내부 RMSE 개선이 공식 점수 개선을 보장하지 않는다.

방법 근거는 [scikit-learn LedoitWolf 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html) 및 [NumPy hermgauss 공식 문서](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite.hermgauss.html)다. 외부 관측·가중치·예보 입력은 사용하지 않았다.
