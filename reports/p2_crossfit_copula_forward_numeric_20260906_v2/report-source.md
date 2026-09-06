# P2 결론 — in-sample profile-copula만 평균 개선 후보로 유지

고정된 8구간 내부 비교를 완료했다. C3에 **outer-training 내부 잔차로만 학습한 strength 1.0 copula**를 추가하면 가을 주평가 RMSE가 **0.517316→0.499286℃**로 개선된다. 독립 검산 1,117/1,117, 별도 프로세스 전체 행 replay 32/32가 통과했다. 반대로 두 고정 inner 월의 OOF 잔차로 적합한 crossfit은 **1.893761℃**로 크게 악화했으므로 full 학습·배포 대상에서 제외한다. 모든 cross-fitting 방법의 실패를 입증한 것은 아니다.

기존 기술 실패를 과학적 실패로 바꾸지 않았다. 원 attempt는 그대로 보존하고 새 ID에서 **40 fit 재사용 + 48 fit 추가 = 총 88 fit**을 완료했다. 연구 자체는 공식 입력·CSV·업로드 모두 0이다. 개선 arm의 source-only full 학습/답안은 별도 [materialization 기록](../p2_crossfit_copula_materialization_20260906_v3/report-source.md)으로 분리한다. 해당 링크의 단계 상태를 확인해야 하며 이 연구 PASS만으로 공식 제출 완료를 뜻하지 않는다.

## 평가와 실제 수치

각 표의 RMSE는 동일 행의 `sqrt(SSE/n)`이며 fold RMSE의 산술평균이 아니다. 기준/후보 모두 세 seed 성분평균 후 평가했다. 과거 CUDA 답안의 공식 점수를 이번 CPU C3에 붙이지 않는다.

| 평가면 | n | C3 ℃ | in-sample ℃ | crossfit ℃ |
|---|---:|---:|---:|---:|
| B3 가을 자연 입력 — 주평가 | 26,273 | 0.517316 | 0.499286 | 1.893761 |
| 전체 8구간 자연 입력 | 166,268 | 1.233020 | 1.227144 | 1.864060 |
| 자연 T5 결측 | 17,657 | 0.345084 | 0.352721 | 0.515937 |
| 추가 마지막 17일 결측의 실제 평가 행 | 42,292 | 1.625741 | 1.623824 | 1.958891 |
| B3 추가 결측 행 | 7,322 | 0.514968 | 0.497314 | 1.647139 |

주평가 in-sample ΔRMSE는 −0.018030184℃(−3.49%). 7일 paired block bootstrap 2,000회/seed 20260906의 기술적 90% 구간은 **[−0.029492503, −0.004649777]℃**, 10 block이다. 전체 pooled Δ는 −0.005875837℃, CI90 [−0.009513546, −0.002310437]. 추가 결측 Δ는 −0.001916400℃이지만 CI90 [−0.007212602, +0.004528294]로 0을 포함한다. bootstrap 개선 비율은 공식 개선 확률이 아니며 0.8 hard gate로 쓰지 않았다. 내부→공식 점수 변환은 분포·CPU/CUDA·동일 SHA 여부가 달라 확정할 수 없다.

| 자연 입력 구간 | n | C3 ℃ | in-sample ℃ | crossfit ℃ |
|---|---:|---:|---:|---:|
| B1 | 22,940 | 1.984428 | 1.979809 | 2.170842 |
| B2 | 26,018 | 1.336328 | 1.317950 | 3.040240 |
| B3 | 26,273 | 0.517316 | 0.499286 | 1.893761 |
| B4 | 18,093 | 0.046054 | 0.134964 | 0.482433 |
| B5 | 16,417 | 0.726273 | 0.718948 | 0.966289 |
| B6 | 26,308 | 1.802253 | 1.794943 | 1.923969 |
| B7 | 13,335 | 1.009578 | 1.019629 | 1.041969 |
| B8 | 16,884 | 0.267989 | 0.275598 | 0.456482 |

위험은 별도로 남긴다. in-sample은 B4/B7/B8 및 자연 T5 결측에서 악화한다. 추가 결측 B2/B6/B7도 악화하며 B7은 1.384607→1.401023℃다. 반면 B3의 세 층은 모두 평균 개선이다(layer2 0.204184→0.195616, layer3 0.379761→0.339695, layer4 0.786828→0.772224℃). 이 위험을 본 뒤 correction 강도를 낮추거나 행별 router로 바꾸지 않았다.

## 고정 계약·누출 방지·지원 공백

- `ocean_forward_v5`의 정확한 8구간/7일 outer purge와 각 outer-training 안의 사전 선택 2개 calendar-month/7일 inner purge를 유지했다. inner 날짜는 source 지원만으로 골랐고 outer 결과로 옮기지 않았다. 24개 C3 outer fit 결과가 후보보다 먼저 봉인됐다.
- C3 blockmask/normalized-Huber/domain weighting/gradient penalty, seed 20260901/02/03, 60 epoch, 11개 physical profile 특징, copula strength 1.0은 동일하다. augmentation의 원본 행별 총량을 보존했다.
- 목표 layer2/3/4의 temp와 psal을 함께 선마스크한 뒤 공개층 특징을 만들었다. 추가 결측은 T5/S5를 함께 가린 후 파생값을 재계산했다. 외부 관측·pretrained·Public 역산·옛 답안/계수 계보 0.
- in-sample 보정은 outer-training의 모델과 그 training residual을 사용한다. outer-label을 보정 학습에 넣지 않으므로 outer 평가와는 분리되지만, 잔차의 낙관적/분포 편향 가능성은 남는다. crossfit 보정에는 outer model이나 outer label이 들어가지 않는다. 고정 2개월만의 OOF는 전체 outer-training OOF가 아니며 계절 대표성이 희박하다. 이것이 큰 악화의 원인인지 별도 개입으로 입증한 것은 아니다.
- B1/B3 actual-depth 지원이 부족한 33/34행은 기존 nominal fallback으로 보존했고 평가 행 삭제 0이다. 시간문맥 특징 추가나 실제 수심 token 전환은 이번 변경에 섞지 않았다.
- B4/B8 마지막 17일은 평가 가능한 target 0행이다. 원 관측 grid 끝을 이동하지 않았다. 해당 연도의 유효 목표 관측이 각각 12월12일/12월10일에 끝나며 12월15일 이후 7,344개 grid target 행은 가려져 있다. **NOT_ESTIMABLE_NO_ROWS**이지 RMSE 0이나 겨울 강건성 PASS가 아니다.

## 기술정정·비용·검증

원 v1은 B2 crossfit의 outage 밖 최종값 1행에서 3.5527e−15℃ 차이를 bit-exact 단언한 것이 기술 실패 원인이었다. 입력/키/NaN/baseline/physical features의 정확 동일성은 그대로 검사한다. 다른 missingness-pattern batch 크기 사이의 **최종 출력 비교만 rtol 0, atol 1e−12℃**를 허용했으며 예측값을 고치거나 복사하지 않았다. 같은 입력 형태의 별도 프로세스 replay는 여전히 bit exact이다. 1e−9 출력 오류와 1-bit 입력/키 변경은 합성 테스트에서 거부한다.

0-fit provenance 검사 **205/205**로 원 source·code·dependency·36 backbone·4 copula·calibration·baseline 계보를 재계산했다. 4개 copula는 실제 학습/저장이 끝났지만 원 성공 후 receipt는 3개뿐이었다. 네 번째는 새 fit으로 세지 않고 정확 모델/데이터로 복구했으며 개별 runtime은 미상(null)으로 남겼다. 원 artifact 전체 hash inventory는 전후 동일하다.

- 원 수행 1,809.485초 + 이번 추가 수행 1,557.250초 = **3,366.735초(56.11분)**. 원 90분 연구 실행 상한 안이다.
- 이번 신규 36 backbone + 12 copula; 재사용 36 backbone + 4 copula. 합계 72 backbone + 16 copula = 88. 신규 full training은 이 연구에서 0이다.
- 0-fit provenance 135.406초와 개발/테스트/후속 QA·replay 시간은 연구 fit 실행 시간과 분리한다. 전체 개발 시간을 최종 6시간 재현 증명으로 부르지 않는다.
- synthetic 9 PASS/Ruff PASS. 담당 독립 QA **1,117/1,117 PASS**: SSE/n/RMSE·CI90·학습 키·mass·마스크·CDF/covariance/shrinkage·hash·원 artifact 불변·예산 검사.
- root의 별도 산술 검산 **73/73 PASS**, 모델 runner를 import하지 않은 원 evaluation 배열 재계산.
- 학습 PID 37864와 다른 PID 41680에서 자연/추가결측 전체 166,268키 × 3개 arm의 저장 모델 replay, 32/32 PASS, 최대 오차 0. 128행 probe나 공식 CSV 전체 재현으로 혼동하지 않는다.
- 기준 모델 재생성 검사: 기존 적격 CUDA C3의 빈 폴더 3-fit→답안→replay는 [portable baseline 보고서](../portable_cleanroom_20260906_v1/P2/report-source.md)의 PASS를 유지한다. 이번 CPU 후보의 full 재생성은 아래 별도 단계다.

## 결과 원장과 다음 판단

- [result.json](result.json): 전체 n/SSE/RMSE/bias/CI/fit receipt/hash의 canonical 수치.
- [independent-qa.json](independent-qa.json), [replay.json](replay.json), [zero-fit-reuse.json](zero-fit-reuse.json), [focused-validation.json](focused-validation.json), [contract.md](contract.md).
- [root independent arithmetic](../remaining_work_completion_20260906_v1/p2-root-arithmetic-qa.json).
- result SHA256 `cd9b6c35285b9cc096b1e1d3fa62bab429388276dbcb6ce492ddd1a287a16266`.
- replay SHA256 `98e93a735740178be9238d31d0198baf80507dd95a4df020ce537e8fcc6fcba1`.

후속 승인된 **in-sample 후보만** 새 빈 폴더에서 C3 full3 + covariance/CDF1 = 4회 학습해 source-only 배포 답안까지 준비했다. 실패한 crossfit의 full/inner6 학습은 실행하지 않았다. CPU C3 control과 후보 파일의 schema/key/order/finite/SHA·별도 PID·ZIP 추출 replay는 [별도 materialization 보고서](../p2_crossfit_copula_materialization_20260906_v3/report-source.md)에 연결한다. root의 [공식 비교](../p2_copula_official_submission_20260906_v1/report-source.md) 후 **기존 CUDA fallback 유지**로 판단했으며 점수 기반 재튜닝은 없다. zt_real/time-context 및 추가 튜닝은 이번 범위 밖이다.
