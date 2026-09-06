# P3 numeric training schedule — 실행 전 계약

numeric-lead+hmax 유지 정책에서 학습률/트리 횟수만 작은 범위로 바꾼다. 모델 구조, depth, regularization, seed, case weights, 591특징, 시간 분리, router, long persistence0.2는 바꾸지 않는다. 각 recipe의 iterations×learning_rate는 기준과 같지만, 이것이 학습 결과의 등가성/안전을 보장한다는 뜻은 아니다.

| recipe | single iterations/lr | multi iterations/lr |
|---|---|---|
| baseline | 700 / .035 | 1,200 / .03 |
| compact | 525 / (.035/.75) | 900 / .04 |
| gentle | 1,050 / (.035/1.5) | 1,800 / .02 |

## 선택과 외부 평가

가장 이른 Q2_2024 outer training mask 안의 2024-03-01≤anchor<03-25가 고정 held-inner다. Inner train은 3월1일−78시간 이전이며 inner held-out과 같은 station/episode를 제외한다. Parent outer train과 교집합하므로 Q2 holdout episode도 제외한다. 실제 metadata에서 모든 inner target이 최초 outer context 이전에 준비되는지 검사한다.

Q1에는 earlier OOF router가 없으므로 세 recipe의 inner 비교에는 동일 equal single/multi + 고정 long0.2 정책을 적용한다. 6개 inner backbone을 끝내고 pooled6lead SSE가 가장 작은 recipe 하나를 선택한다. 정확 동률은 baseline→compact→gentle 순이다. 모든 inner 결과를 본 뒤 선택 receipt를 봉인하고, 그 후에만 선택된 recipe로 다섯 outer fold를 학습한다. Inner 선택용 warmup 정책과 outer의 prior-OOF router 완성 정책 차이를 숨기지 않는다.

Primary는 수정 없는 v5 5분기 103,602행 전체의 unweighted pooled RMSE이다. 선택된 recipe의 각 outer router는 해당 recipe에서 생성한 completed-previous-fold OOF 중 78h/episode mask를 통과한 것만 사용한다. 비교 기준은 이미 검증한 numeric baseline OOF의 exact SHA 재사용이다. 해당 데이터·source·split·입력·키·149QA·별도 PID replay 해시를 전부 연결한다. Outer 결과로 recipe를 다시 고르거나 비율을 조정하지 않는다.

기준이 inner에서 선택되면 기존 numeric baseline을 그대로 보존하고 추가 outer 학습은 하지 않는다. Compact/gentle이 선택되면 5×2 backbones + 4 routers를 수행한다. 평균 개선은 보존하고 quarter/lead/정점/wind missingness 위험은 별도 표로 기록한다. 안정성 0.8 hard gate나 결과 기반 추가 후보는 없다. 이미 노출됐던 분기를 포함하므로 virgin evaluation이라고 주장하지 않는다.

## 예산과 실행 경계

- 최대 신규 역사 backbone16 + router4 = **20 fits**. Baseline 선택 시 inner6 fits만 수행.
- CPU2, GPU0 multi만 독점. root GPU 해제 신호 전 execute 금지.
- 처음 예정된 baseline inner2 fits의 실측을 재사용하여 결과 점수를 계산하기 전에 최악 recipe 전체 예산을 예측한다. 60분 예산 초과 전망이면 RESOURCE_STOP, 재시작/축소 자동실행0.
- 최악 약35~45분 예상. 실행 프로세스 3,600초 watchdog, CPUfit callback 및 GPUfit 전후 검사. 학습 중 프로세스/진행률만 확인하고 성능 조기 공개0.
- 공식 입력/hidden/외부/CSV/upload/full-fit/Git/최종 모델 lock0. 이전 frozen 코드/attempt 변경0.
- 끝나면 saved model 전체 예측 새PID exact replay, 독립 pooled SSE/RMSE/cluster CI/키/hash/시간/source/fit QA와 focused pytest/Ruff 근거를 남긴다.

## 중복과 재사용 감사

`p3_meaningful_learning_curve_v1`은 학습 데이터 prefix와 구조/단일·다중 head 비교다. 현 paired learning-rate/tree schedule 실험과 같지 않다. `p3_catboost_valid_hpo_20260829_v2`는 categorical single, 다른 sparse windows, 300/900/2500 halving/구조·정규화 변경이며 KMA downstream을 고정한 구형 계보다. 따라서 이 실험의 예측·선택 결과·모델은 재사용하지 않는다. 설정 이력만 중복 점검에 사용했다.

재사용하는 값은 새 clean numeric `p3_numeric_lead_forward_gpu_20260906_v2`의 후보 OOF와 source-only prepare cache다. `ff42a6...7960` local full 후보는 별도 보존한다. 이번 실험은 기존 numeric 또는 새 승자의 whole-cold package 검증을 대체하지 않는다. 기존 `P3_forward_candidate_cold_v1`은 numeric591 지원 및 prepare/train/qa/infer/verify-answer 경로가 있으나 새 경로에서 source prepare+12backbone+5router를 실제 완료해야 하는 gap이 남는다. 그 추가17 fits는 이번 budget에 포함하지 않으며 root의 별도 GPU 큐를 따른다.
