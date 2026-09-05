# P2 프로파일 보정 후보 — 로컬 답안·전체 재현 QA 완료, 업로드 전

결론: **빈 모델 폴더에서 다시 학습한 적격 C3에 고정 full 프로파일 copula 보정을 더한 P2 후보가 준비됐다.** 독립 QA 28 checks PASS, 합성 pytest 4 PASS/Ruff PASS, 별도 PID에서 공개 context부터 전체 26,061행을 다시 계산한 CSV SHA exact PASS다. 이는 생성·계보·재현 무결성의 통과이며 **공식 점수 개선 또는 최고점 보장**은 아니다. 내부 가을 intact는 개선했지만 추가 7·14일 결측은 악화했다. 정책을 사후 half로 바꾸지 않았고, 업로드/최종 모델 잠금/Git 작업은 하지 않았다.

## 파일 선택과 완료 증거

- **P2 / OCN-02 후보**: `artifacts/p2_profile_copula_deploy_20260905_v4/05_answer/submission_p2_profile_copula.csv`.
- 제출 schema: `station,layer,time,temp`; 26,061행, target layer 2/3/4, sample 순서 및 index 전체 키 일치, 중복/비유한값 0.
- 후보 SHA-256: `3dff2982f38ef2306248e76494c8d2c95b730244ce5730ba59335418dd3f557b`.
- `control_C3.csv`는 재계산한 기준 해시 대조용이다. `replay_p2_profile_copula.csv`와 `replay_C3.csv`는 재현 검산용으로 별도 후보가 아니다. 모델 파일을 리더보드 CSV 입력란에 올리지 않는다.
- 완료 근거: [training-result.json](training-result.json), [result.json](result.json), [replay.json](replay.json), [independent-qa.json](independent-qa.json), [실행 계약/명령](README.md).

## 학습과 재생성 계보

| 단계 | 신규 학습 | 검증/재사용 | 소요 시간 |
|---|---:|---|---:|
| 별도 C3 빈 `03_model` 재생성 v6 | 3 backbone fits, 각 60epoch | 배포 관측만 scratch 학습, 기존 model/answer/OOF 로드 0 | 69.094초 |
| C3 공식 답안/전체 replay | 0 | 서로 다른 PID, 기존 적격 C와 SHA exact, 27-check QA PASS | 추론 10.844초, replay 8.734초 |
| 본 full copula 학습 | 1 deterministic joint fit | 새로 재생성한 C3 3개를 동일 SHA로 새 `03_model`에 복사 | 11.188초, covariance fit 0.484초 포함 |
| 본 후보 공식 추론/전체 replay | 0 | 공개 context부터 재계산, 서로 다른 PID, 후보 및 C SHA exact | 각 11.391초 / 11.390초 |

기준 모델 재생성 검사: **PASS**. 실제 선행 검사는 [v6 보고서](../p2_clean_regeneration_20260905_v6/report-source.md)와 [v6 독립 QA](../p2_clean_regeneration_20260905_v6/independent-qa.json)에 있다. v4/v5는 자체 생성 모델의 SHA 읽기 가드 오류로 첫 seed 후 중단한 기술적 실패이며 모델/lock/실패 영수증을 보존했다. 재생성 관련 실제 학습 비용은 실패 첫 seed 2개 + 성공 v6 3개 = 5 backbone fits다. 본 후보 단계가 새 backbone 0fit이라고 이 비용을 없애지 않는다.

본 단계의 학습은 공개 배포 관측의 기존 적격 166,268행을 사용해 **재생성 C3 평균 예측의 `truth−C3` 잔차**에 한 번만 rank-CDF/Ledoit–Wolf 모델을 적합했다. 과거 답안값이나 historical OOF를 full 잔차 학습에 사용하지 않았다. CDF/공분산 모델도 학습 산출물로 `03_model/copula_full.npz`에 저장했다. correction 강도는 봉인된 1.0이며 추가 threshold/강도/routing 탐색 0이다.

본 단계의 training / inference / replay PID는 **23708 / 11772 / 37744**다. 128개 train 행의 모델 reload 검사는 학습 중 작은 검사이며, 별도로 전체 공식 26,061행의 공개 context→모델→답안 replay를 완료했다. 두 검사를 혼동하지 않는다.

## 내부 효과와 위험 — 같은 행, 같은 3-seed 평균 기준

| 내부 평가 | n | 기준 C3 RMSE ℃ | full 후보 RMSE ℃ | Δ 후보−기준 ℃ |
|---|---:|---:|---:|---:|
| 가을 intact 주평가 | 26,273 | 0.488284 | 0.472272 | −0.016012 |
| 전체 intact | 69,850 | 0.859250 | 0.853508 | −0.005742 |
| 자연 T5/S5 결측 | 17,165 | 0.277259 | 0.272372 | −0.004886 |
| 추가 가을 7일 결측 | 3,018 | 0.848478 | 0.881242 | +0.032764 |
| 추가 가을 14일 결측 | 6,017 | 0.731300 | 0.754780 | +0.023479 |

전체 원수치/다른 계절/고정 half 비교/분모 및 367-check 독립 산술 QA는 [historical 연구 보고서](../p2_profile_copula_residual_20260905_v4/report-source.md)에 있다. 추가 가을 3일은 1,294행 중 4행의 공용 온도 지원 부족 때문에 전체 시나리오 `SUPPORT_BLOCKED`/unscored를 유지한다. 1,290행으로 점수를 잘라내지 않았고 이는 실제 공식 입력 결함이라는 뜻이 아니다. 본 공식 26,061행은 모두 지원 조건과 finite 검사를 통과했다. 겨울 추가 outage는 원래 T5/S5 결측인 no-op이라는 한계도 유지한다.

잔차 모델의 baseline train 예측은 in-sample이다. outer label은 fitting에서 제외됐지만 반복 historical 평가이므로 새 독립 확증 또는 공식 점수로 치환하지 않는다. 공식 예상 점수는 `null`이며 기존 Public 반환값의 역산·계수 맞춤은 없다. 따라서 **위험을 기록한 정보가치 후보**로만 전달한다.

## 데이터 경계와 해시

- 학습: 배포 `observations.csv`만 읽기. official index/sample/CSV 생성/old OOF/old answer값/업로드 0.
- 추론 및 replay: 각 프로세스에서 official index/sample의 station/layer/time 키를 각 26,061행 읽음. sample 목표값·hidden truth·과거 답안값·과거 OOF 0, 업로드 0.
- 공개 원본 SHA 전후 불변: `cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a`.
- 다시 추론한 C3 SHA: `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071`; 기존 답안은 값이 아닌 알려진 해시로만 대조했다.
- 저장 copula SHA: `fd1e889eacdcfc07cdd540f52f157f36987a147ff2ae71894cb0722eb3f4ca1f`.
- [사전 봉인](preregistration-seal.json)에 runner/config/dependencies 및 재생성 C 모델 3개의 해시를 보존했다. sealed runner/config 수정 0.
- 후보는 기준과 26,061행 모두 달라 중복 C 제출 파일이 아니다. 이것만으로 성능 개선을 증명하지 않는다.

현재 checkout에서의 재학습·답안 생성·재현 검증이며 **portable 최종 코드/환경 ZIP, 다른 머신·OS 수준 인터넷 차단·운영진 하드웨어 6시간 재현까지 완료했다는 뜻은 아니다.** 데이터/모델/답안/attempt locks는 Git에서 제외한다. 다음 판단은 root가 이 위험과 미채점 상태를 함께 검토해 별도 제출 여부를 결정하는 것이며, 본 실행에서 추가 모델 학습이나 업로드를 자동 수행하지 않는다.
