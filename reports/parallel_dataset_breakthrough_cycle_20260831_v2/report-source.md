# 2026-08-31 병렬 데이터셋 돌파구 연구

## 기술 요약

결론은 **P1을 다시 연다. 단, 기존 신경망 재튜닝이 아니라 `clean-state cross-layer CAPA segment likelihood` 한 축만 Stage-1 후보로 올린다.** 2026-08-31 02:54 KST 공식 리더보드 읽기 전용 재확인에서 우리 P1은 28.909341점, 문제 최고는 32.110453점으로 **3.201112점 차이**다. 같은 시점 P2·P3 차이는 각각 0.739438점, 0.580444점이어서 P1 우선순위는 유지가 아니라 강화됐다. 문제별 배점과 원지표의 정확한 변환은 주최 측 공식식으로 단정하지 않는다.

P1의 재개 이유는 명확하다. 강건한 계절 clean state, 인접 층 관계, 8–96시간 다중척도 offset/drift 표현을 쓰는 v6 계보는 성능 실패가 아니라 실행 격리 QA 실패로 `fit=0`이었다. 다만 옛 pointwise logistic+고정 run filter를 그대로 실행하지 않고, point anomaly와 collective segment를 함께 비교하는 penalized segment likelihood decoder로 바꾼다. Stage-0 12개 검사는 모두 통과했지만, 새 decoder의 실행 계약과 독립 QA가 아직 없으므로 현재 상태는 `READY_TO_PREREGISTER_RESEARCH_ONLY_NOT_READY_TO_FIT`이다.

P2는 bin17 champion을 유지하고 mask-matched multivariate DINEOF residual을 Stage-0 후보로만 남긴다. P3는 KMA α=.425 champion을 유지하며 case를 operational forecast cycle에 연결할 issue-time manifest가 없어 학습 0회로 정지한다.

## P1 격차는 용량보다 long-event decoder와 분포 이동 문제다

P1 공식 목적함수는 event F1이 아니라 모든 행의 binary F1이다. 따라서 event IoU, station, layer, month, anomaly type은 원인 진단에는 유용하지만 승격 목적함수를 대신할 수 없다. add-only 후보라면 같은 평가면에서 추가 행 precision이 incumbent F1의 절반보다 커야 F1이 개선된다는 정확한 sanity condition을 적용한다.

로컬 aggregate는 구조적 headroom을 보인다.

- offset recall 0.649154, drift recall 0.646150이다.
- I-ORS layer 1 F1 0.432777, I-ORS layer 5 F1 0.695513, S-ORS layer 2 F1 0.506373이다.
- 세 hard cell의 공식 test row-share proxy 합은 24.7907%다.
- June F1은 0.502853이다.
- 반면 Q2/Q3/Q4 OOF F1은 0.774627/0.901888/0.906618로 크게 갈린다.

이 패턴은 더 큰 범용 신경망보다 offset/drift의 지속 구간을 높은 precision으로 찾는 decoder가 먼저라는 뜻이다. 과거 SupCon, Group-DRO, trial18, hierarchical add-only는 표현을 바꿨지만 long-event proposal precision의 시간 수송 문제를 해결하지 못했다. 또 test mass의 43.6108%는 같은 계절 Q2 station-layer-month 지원이 없고, 월 비중 total variation은 0.683706이다. 그러므로 exposed Q2–Q4의 작은 이득만 보고 정식 승격하면 다시 과적합될 가능성이 높다.

## P1 Stage-0가 미실행 계보를 복구했다

새 preflight는 aggregate JSON만 8개 읽었고 raw training row, 공식 test/sample/submission/hidden row, model fit, prediction, CSV, upload는 모두 0이다. 확인된 사실은 다음과 같다.

1. v6r2는 기술 QA P0 2개/P1 3개로 tombstone됐고 actual run과 model fit은 0이다.
2. v6r4도 QA receipt, authorization, attempt lock, actual run이 모두 없고 fit 0이다.
3. 기존 표현은 robust seasonal baseline, adjacent-layer graph, 8/16/32/64/96시간 trajectory geometry다.
4. 새 decoder는 pointwise logistic+fixed run filter가 아니라 point/collective penalized segment likelihood다.
5. 따라서 이 후보는 완전히 새로운 표현은 아니지만 exact 재실행도 아니다. 판정은 `REOPEN_UNEXECUTED_FAMILY`다.

옛 v6r4의 `full +0.02`, 모든 late fraction 양수, 3/3 inner nondegrade, worst-of-16 nonnegative 같은 hard gate는 쓰지 않는다. 새 one-shot의 primary는 pooled row micro-F1 delta 하나다. paired whole-event/KST-day uncertainty를 함께 보고하되, I1/I5/S2·June·type·thermocline false positive는 diagnostic으로만 남긴다.

## 데이터 품질과 모델 위험

P1 train의 intended grain은 station-year-layer-time 한 행이다. 기존 audit에서 776,706행 모두 key unique이고 exact duplicate 0, key null 0이다. label/anomaly_type 정합성도 깨지지 않았다. positive는 32,126행, contiguous event는 263개뿐이다. 데이터는 깨끗하지만 독립 사건 수가 작아 model capacity 확대보다 event-level uncertainty와 분포 이동이 더 큰 위험이다.

이상치 제거는 적용하지 않는다. P1의 label=1 자체가 탐지 대상이므로 robust fitting에서 영향력을 제한할 수는 있어도 해당 행을 학습자료에서 삭제하면 문제 정의를 훼손한다. KORS 연구가 gradient check가 thermocline의 정상 변동을 오탐할 수 있다고 보고했으므로, 새 decoder는 thermocline false positive를 핵심 kill diagnostic으로 둔다.

## P2는 fresh 61-day label surface 전까지 Stage-0만 허용한다

우선 후보는 `mask-matched multivariate DINEOF residual`이다. target layer 2·3·4의 temp/psal을 61일 동시에 가리는 mask와 똑같은 inner CV로 rank 1 또는 2만 선택하고, bin17 anchor의 residual만 ±0.2°C 범위에서 보정한다. fixed rank-5 standalone DINEOF, variance-selected CMFPCA, bin17/18 PLS, Gaussian copula와 입력·목표가 다르다.

차선 후보는 public T/S와 fixed depth basis를 쓰는 rank 1/2 inductive depth-factor completion이다. 두 후보 모두 unsupported/OOD에서 exact bin17 no-op이다. 하지만 현재 모든 historical label block이 반복 노출됐으므로 fit을 돌려도 fresh evidence가 아니다. organizer-provided untouched same-season 61-day block 또는 별도 blind evaluator가 생길 때만 최대 33 fits one-shot을 허용한다.

## P3는 issue-time manifest가 없어서 학습하면 안 된다

P3에서 검증할 가치가 있는 새 정보원은 마지막으로 발행 완료된 GFS/GEFS cycle의 regional fetch forcing과 두 operational wave centre의 disagreement다. 그러나 현재 data contract에는 anonymous case를 `UTC issue_time, station 좌표, model cycle, publication cutoff`에 연결하는 manifest가 없다. ERA5는 reanalysis이며 deployment-time operational forecast가 아니므로 이를 익명 평가기간에 정렬하는 방식은 허용하지 않는다.

따라서 현재 판정은 `STOP_NO_DATA / TRAIN_0`이다. KMA α=.425 champion을 유지한다. signed issue-time manifest, as-of archive, 최소 8개 fresh synoptic episode block이 모두 생길 때만 regional-fetch residual Stage-0를 다시 연다. 최소 12개 block과 station별 3개 block이 있어야 confirmation 가능 상태로 올린다. 8/12는 문헌의 보편 상수가 아니라 이번 실험의 보수적 preregistration 값이다.

## 다음 실행 순서

1. P1 CAPA decoder의 exact proposal fingerprint, segment penalty, overlap resolution, incumbent union rule을 하나의 새 config로 동결한다.
2. synthetic fixture와 aggregate-only contract test로 기존 semimarkov/long-event decoder와 semantic duplicate가 아님을 검증한다.
3. 독립 QA가 통과하면 exposed historical folds에서 one-shot falsification을 수행한다. 양수여도 `RESEARCH_ONLY`이며 결과 기반 재시도는 0회다.
4. pooled delta가 0 이하이거나 same-surface add-only precision이 incumbent F1/2 이하이면 exact 후보를 닫는다.
5. P2는 fresh label block, P3는 signed issue-time manifest가 생길 때까지 fit 0을 유지한다.

## 남은 질문

- 2026-08-31 02:54 KST 재확인에서 P1 격차는 3.201112점이었다. 이후 실시간 리더보드 변동은 별도 시간표시 스냅샷으로만 갱신한다.
- P1의 새 segment decoder를 fresh label 없이 공식 제출로 검증할지, research-only local falsification 후 기다릴지는 제출 기회와 9월 7일 최종 모델 일정에 맞춰 별도로 결정해야 한다.
- P2의 untouched 61-day surface와 P3 issue-time manifest를 주최 측이 제공할 수 있는지 확인이 필요하다.
