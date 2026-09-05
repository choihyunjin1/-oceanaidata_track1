# P2 v5 cross-fit copula: 기술 종료, 기준선 검증 완료

결론: 새 inner-OOF copula의 성능 우열은 **아직 판단할 수 없다**. 8개 구간 C3 기준선 24 fits는 완성·봉인·독립 검증됐지만, B2 cross-fit copula의 비마스킹행 불변성 검사에서 1행의 부동소수 반올림 차이 `3.552713678800501e-15`℃가 발생하여 실행이 정상적으로 차단됐다. 이는 **TERMINAL_TECHNICAL_FAILURE / NOT_ESTIMATED_INCOMPLETE**이며 과학적 NO_GO가 아니다. 기존 코드·설정·모델·attempt lock을 수정하거나 실행을 재시작하지 않았다. 후속 full fit·공식 입력·CSV·업로드는 0이다.

## 실제 완료 범위

| 항목 | 완료 / 계획 | 의미 |
|---|---:|---|
| Outer C3 | 24 / 24 fits | 8 folds × 3 seeds. 모든 후보 보정 전에 결과 봉인 |
| Inner C3 | 12 / 48 fits | B1/B2의 2 inner blocks × 3 seeds만 완료 |
| Copula fit/save | 4 / 16 fits | B1 2개, B2 2개. 마지막 것은 저장 후 불변성 assertion에서 종료 |
| Copula 성공 영수증 | 3개 | 영수증은 assertion 뒤 기록되므로 실제 fit 비용 4개와 구분 |
| 실제 fit 비용 합계 | 40 / 88 | 36 neural fits + 4 small covariance/CDF fits |
| 실행 시간 | 1,809.485초 | 약 30.16분, CPU 2 threads / GPU 0 |
| 새 프로세스 기준선 재현 | PASS | 166,268행 × 자연/추가마스킹 2면 전체, 8 folds 모두 exact |
| 독립 QA | 321 / 321 PASS | 기준선 산술·재현·학습 키·hash·purge·실패 비용 원장 |
| 합성 검사 / Ruff | 9 / 9 PASS / PASS | 공식 데이터 없이 target mask·outage·guard·CPU 경로 등 |
| 후보 비교 / CI90 | NOT_ESTIMATED_INCOMPLETE | 부분 구간만으로 후보 승격 또는 탈락 판단하지 않음 |

CPU 첫 예정 fit인 B1/20260901은 44.953초였으며 결과를 읽기 전 보수적으로 총 3,931.279초를 예상했다. 5,400초 계획 상한 이내이므로 해당 fit을 그대로 재사용했다. 실제 종료 원인은 시간 상한이 아니다. 상한 검사는 epoch/fit 경계의 운영 검사이며 OS 강제종료가 아니다. 원본 학습 시간과 종료 후 무학습 QA 시간(16.532초)은 구분한다.

## 완성된 C3 기준선: 동일 키·분모의 새 CPU 8-fold 결과

단위는 RMSE ℃. 자연 관측 입력과, 각 fold의 마지막 17일 공개 T5/S5를 함께 가린 test-matched 입력을 비교한다. `test-matched`는 배포 학습 자료를 인공 마스킹한 내부 평가이지 공식 정답 검사가 아니다.

| 평가 범위 | 행 수 | 자연 입력 C3 | 추가 마스킹 C3 |
|---|---:|---:|---:|
| 주평가 B3 | 26,273 | 0.517316228 | 0.580416943 |
| 전체 pooled | 166,268 | 1.233020028 | 1.296071339 |
| 자연 T5 결측 | 17,657 | 0.345083784 | 0.343638808 |
| 자연 T5 관측 | 148,611 | 1.298778863 | 1.365779488 |
| 추가 outage 구간 | 42,292 | 1.419900483 | 1.625740754 |
| B1 | 22,940 | 1.984428380 | 2.064342146 |
| B2 | 26,018 | 1.336328204 | 1.364392638 |
| B3 | 26,273 | 0.517316228 | 0.580416943 |
| B4 | 18,093 | 0.046053672 | 0.046053672 |
| B5 | 16,417 | 0.726273283 | 0.938872902 |
| B6 | 26,308 | 1.802253160 | 1.839248637 |
| B7 | 13,335 | 1.009578160 | 1.210349246 |
| B8 | 16,884 | 0.267988797 | 0.267988797 |

분모는 사전 정의한 finite target temp, finite nominal baseline, 공개 온도 2개 이상 지원을 충족하는 166,268개 target 키이다. 이 적격성은 숨겨진 QC를 읽은 것이 아니다. 평가 도중 행 삭제 0. 자연 T5 결측 분모는 T5 **또는** S5 결측 OR 분모와 다르며, 예전 3-fold 결과와도 비교 분모가 다르다. 새로운 CPU fold 모델이므로 과거 GPU 3-fold 수치나 공식 0.455143℃를 직접 빼서 개선으로 주장하지 않는다. 내부 평가를 공식 예상 점수로 환산할 검증된 대응식이 없어 예상 공식 점수는 산출하지 않았다.

## 기술 원인과 보존

봉인 runner 569행은 `natural`과 `outage`의 비마스킹행 답을 `np.array_equal`로 검사한다. B2 cross-fit에서:

- 비outage 19,402행의 C3 및 11개 물리 특징은 NaN 위치까지 exact 동일.
- Copula는 결측 패턴별 행렬 곱을 수행한다. 해당 패턴의 전체 batch 크기가 자연 25,482행, outage 18,931행으로 달라졌다.
- 교정값 최대 차이 `4.440892098500626e-16`, 최종 온도 최대 차이 `3.552713678800501e-15`℃; 최종값이 다른 행은 1개.
- 같은 shape의 동일 비outage 입력만 각각 재추론하면 exact 동일하다.
- source hash와 runner/config/dependency seal 모두 불변이다.

따라서 관측된 실패는 batch-shape에 따른 수치 반올림으로 엄격한 bit equality가 깨진 것이다. 입력 오염·규칙 변경·성능 실패의 증거가 아니다. [기술 진단](technical-diagnostic.json)은 **fit 0, 성능표 열람 0**으로 원인을 확인했다. 봉인 assertion의 허용치를 사후 수정하지 않았으며, 원래 실패 영수증도 PASS로 바꾸지 않았다. 기술 수정이 필요하다면 별도 승인·새 ID·실수 오차 한계와 동일 입력 계약을 사전 정의하고 이미 완료된 fit을 보존하는 절차가 필요하다. 현재는 실행하지 않는다.

## B4/B8의 평가 공백: 날짜 이동 없이 보존

원본 격자는 두 해 모두 12월 31일 23:50 KST까지 존재한다. 데이터셋 전체가 일찍 끝난 것이 아니다.

| Fold | 고정 outage 시작 | 마지막 유효·지원 target | 마지막 17일 target 격자 | 유효 target |
|---|---|---|---:|---:|
| B4 | 2024-12-15 00:00 KST | 2024-12-12 21:10 KST | 7,344 | 0 |
| B8 | 2025-12-15 00:00 KST | 2025-12-10 03:00 KST | 7,344 | 0 |

이 구간은 target 온도 자체가 전부 결측이어서 RMSE를 계산할 수 없다. **NOT_ESTIMABLE_NO_ROWS**이며 0오차·성공·겨울 강건성 통과로 표현하지 않는다. 위 표의 B4/B8 전체 fold 성적은 유효한 더 이른 날짜에서만 나온다. 이 마지막 17일 마스크는 평가 가능한 행에 실제로 적용되지 않으므로 자연/추가 입력 성적이 같은 것은 당연하다. 상세 근거: [원본 날짜 집계](evaluation-gap-source-metadata.json).

## 계약·누출·중복 가설 점검

고정 설계는 C3 / 기존 in-sample copula / inner purged OOF copula 3개 비교뿐이다. C3는 seeds 20260901/02/03, 60 epochs, 기존 v23 blockmask recipe이며, domain 가중치·원본별 augmentation 질량·normalized Huber·gradient penalty도 고정이다. Copula는 기존 11개 물리 특징, 강도 1.0이다. 추가 규칙·threshold·강도 탐색 0. 주평가 평균 개선 후보는 보존하고 위험을 별도로 기록하도록 계획했으며 bootstrap 0.8나 outage 비악화 조건을 소급 적용하지 않았다.

모든 target 2/3/4층의 temp와 psal은 feature 생성 전에 함께 마스킹하고 labels를 별도 분리한다. 공개 T5/S5 공동 outage도 파생 특징 생성 전에 적용한다. 새 adapter는 기존 refresh 함수가 갱신하지 않았던 사용되지 않는 public mean/std 진단열까지 재계산한다. 고정 C3 실제 입력과 copula 특징은 바꾸지 않았다.

각 outer train은 v5의 양방향 7일 purge이다. P2는 복원 문제이므로 이 분할은 causal past-only가 아니다. Inner 두 달은 source 지원만 보고 후보 월의 1/3, 2/3 위치를 선택해 봉인했다. Inner train은 outer train 안에서 다시 inner ±7일을 제거하며, outer label 또는 그 label로 학습한 모델은 cross-fit residual 적합에 사용하지 않는다. B1/B2에서 실제 완료된 inner 모델의 키를 독립 대조했다. 전체 8-fold 지원 계획과 실제 완료 범위는 혼동하지 않는다.

반복 가설 점검:

- `scripts/final_submission_20260905/P2/p2_pipeline.py:81` 이후 기존 C3 token에는 실제 수심 상대값, 명목 수심 상대값과 각각의 가용성 정보가 이미 있다. Token 유효성은 공개 온도와 **명목** 수심 지원에 의존하므로 실제 수심 결측 B1 33행 / B3 34행을 삭제하지 않는다.
- 같은 파일 131~143행 context는 target 명목 수심·layer·scale 및 DOY/hour/M2 sin/cos를 이미 포함한다. 단순히 “실제 수심 또는 시간 context를 추가”하는 제안은 새로운 가설이 아니다.
- `scripts/run_p2_score_repair_20260905_v1.py:154`의 기존 actual-depth ablation은 target actual-depth/지원 flag를 context에 더하는 경로였다.
- `scripts/run_p2_profile_copula_residual_20260905_v4.py:94`의 기존 11개 특징에는 실제 수심 profile interpolation, gradient, T/S 공개층 차이, heave가 이미 있다. 이번 변경의 핵심은 새 depth/time feature가 아니라 **보정 residual의 학습내 예측을 inner-OOF 예측으로 교체**하는 것이다.

다만 inner 두 달만 사용하므로 OOF 여부뿐 아니라 보정용 계절·지원 분포와 표본 수가 함께 바뀐다. 성공해도 cross-fitting 자체의 인과 효과만 분리 입증하는 실험은 아니다. 반복 역사 구간 평가이며 새 독립 확인 데이터도 아니다.

## 재현 준비도와 산출물

**기준 모델 재생성 검사: 이번 범위의 새 CPU historical C3 24개 학습 및 새 PID 전체 내부 추론은 PASS.** 이번 실행은 빈 `03_model`에서 공식 전체 학습·정답 CSV 생성까지 하는 검사와는 다르다. 기존 별도 clean regeneration v6 패키지의 결과를 이번 CPU 8-fold 실험의 재생성 결과로 혼용하지 않는다. 현재 실행은 공식 CSV를 생성하지 않았다.

- [원래 terminal failure](terminal-failure.json): 기술 종료, 36개 backbone/3개 post-invariant copula receipt.
- [봉인 계약](contract.md), [봉인 해시](preregistration-seal.json), [feature 지원](feature-support.json).
- [완성된 기준선](baseline-result.json), [독립 QA](failure-independent-qa.json), [검사 실행 기록](focused-validation.json).
- [CPU pilot](cpu-pilot.json), [자원 기록](resource-receipt.json).
- 코드: `scripts/run_p2_crossfit_copula_forward_20260906_v1.py`.
- 부분 종료 전용 QA: `scripts/qa_p2_crossfit_copula_forward_failure_20260906_v1.py`.
- 완전 완료용 QA/replay 경로는 후보 전체 결과가 없어 **미실행**이다. 준비된 `qa_p2_crossfit_copula_forward_20260906_v1.py`가 존재하는 것을 완료 증거로 보지 않는다.

로컬 원본/모델/배열은 `artifacts/p2_crossfit_copula_forward_20260906_v1/` 아래 보존한다. `baseline_oof.npz` schema는 `key, truth, fold, natural_C3, outage_C3, outage_mask`이며 key는 `station|layer|aware timestamp` 문자열, 166,268개 unique. `evaluation.npz`와 최종 후보 `terminal_result.json`은 미생성이다.

해시:

- source: `cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a`
- runner: `9761bfcad9d58ac5d7dbdd6ba95b42197b87aa97fedfe456e58dad3813c0f14f`
- config: `c861ca80ab34d30b8d8a56984a7d34d805db0475ae5fb7ebf27fe67d0774593c`
- baseline key: `122b40e82db95b313fdc59779a6b2d65427ccd37bee4bdb8a1120e26bd8b957f`
- baseline OOF: `5bcb877832d8bff03ffbf55987cf834b6bb585124b694ff7c89a6348f24b9295`

다음 판단: 검증된 기존 제출 패키지는 유지한다. 이 미완료 후보를 새 최고점 후보 또는 탈락으로 분류하지 않는다. 기술 종료를 기록하고 부모의 통합·패키지 작업으로 반환하며, 새 학습이나 업로드를 자동 진행하지 않는다.
