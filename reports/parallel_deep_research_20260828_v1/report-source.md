# P1·P2·P3 병렬 딥리서치 의사결정 보고서

부제: 제한된 공식 제출 기회를 Public 최고점과 구조적 돌파구에 함께 배분하는 실행 설계

작성 기준: 2026-08-28 KST
팀: 분당독고다이
상태: P1·P2·P3 독립 연구 완료 · 코드 및 공식 점수 이력 교차검증 완료 · 신규 제출/업로드 없음

## 결론부터

**현재 확보된 어느 단일 후보도 공식 총점 +3을 보장하지 않는다.** 그러나 세 문제를 같은 방식으로 다루지 않고 서로 다른 시간축으로 운용하면 +3에 접근할 수 있는 실험 포트폴리오는 분명하다.

1. **P2는 즉시 실행 가능한 exploit 축**이다. 공식 OAS 10%→20%가 같은 계보에서 단조 개선했고, 다음 정보가치가 가장 높은 한 번은 `alpha=0.40`이다. 중심 추정은 현 최고보다 약 `+0.44점`, 느슨한 보수구간 상단도 약 `+0.19점` 개선을 시사한다. 이는 보장이 아니라 공식곡선 연장의 사전등록 가설이다.
2. **P1은 가장 큰 구조적 상한을 가진 축**이다. e150의 저확률 장구간 누락은 threshold 조절로 복구되지 않는다. anchor를 보존하고 change-point proposal을 F1-aware interval selection으로 추가하는 설계가 우선이다. local oracle 상한은 크지만 실제 과거기반 scorer는 아직 검증되지 않았다.
3. **P3는 기존 alpha 축을 종료하고 새 backbone으로 이동해야 한다.** `24.066168` 부근의 미세조정은 수렴했다. ERA5 데이터는 완성됐지만 과거 one-shot은 의존성 환경 오류로 fit 0회였다. 새 experiment ID에서 Chronos-2 zero-shot/LoRA와 ERA5 masked-pretrain PatchTST를 비교해야 한다.

총점 +3은 한 문제에서 억지로 만드는 목표가 아니다. 현실적인 분해는 **P2의 저비용 수확 + P1의 장구간 recall 구조 개선 + P3의 새로운 전이학습 축**을 합산하는 것이다. 각 공식 제출은 `최고점 갱신`, `기전 판별`, `다음 의사결정 변경` 중 최소 하나를 충족해야 한다.

## 현재 기준선과 목표의 크기

| 문제 | 현 Public 최고 | 지표 | 33점 환산식/근사 | 단독 +3에 필요한 변화 | 판단 |
|---|---:|---|---|---:|---|
| P1 | 28.901363 | F1 0.833248 | `6.753689 + 26.580043×F1` | F1 약 +0.1129 | 단일 미세조정으로는 비현실적 |
| P2 | 27.264587 | RMSE 0.483661 | `33.333329 − 12.547518×RMSE` | RMSE 약 −0.2391 | 단독 +3은 상위 알려진 점수도 넘어야 함 |
| P3 | 24.066168 | RMSE 0.583892 | `33.333960 − 15.872414×RMSE` | RMSE 약 −0.1890 | 기존 alpha 축으로 불가능 |
| 합계 | 80.232118 | 합산 점수 | 문제별 33.33점 | 총 +3 | 포트폴리오 목표로 해석 |

위 수치는 저장소에 보존된 공식 원장 기준이다. Private 지표는 대회 종료 후 공개되며, Public 개선은 최종 일반화를 보장하지 않는다.

## 공통 승격 기준: 로컬 점수 하나로 자르지 않는다

과거 이력은 로컬 변화가 공식 변화로 일정 비율 변환되지 않음을 보여준다. P1 Router는 로컬 F1 개선보다 공식 개선이 약 10.8배 컸고, P2 OAS 10→20%는 로컬 incremental gain보다 공식 RMSE 개선이 약 9.2배 컸다. P3에서는 부호 역전도 있었다. 따라서 전 문제 공통 scalar transport는 금지한다.

| 축 | 필수 질문 | 통과 기준 | 실패 시 행동 |
|---|---|---|---|
| 계보 무결성 | 같은 anchor·행·후처리·CSV인가 | 기존 제출 재생성 및 hash/row QA PASS | 공식 제출 금지 |
| 누수 방지 | outer truth가 모델/threshold에 노출됐는가 | blocked/LOBO/cross-fit, blind seal | 실험 폐기 또는 탐색으로 강등 |
| best-state 보존 | 마지막 epoch가 아닌 최적 상태인가 | inner metric best checkpoint와 restore 증거 | 재평가 후에만 승격 |
| 오류공간 독립성 | incumbent가 못 잡는 오류를 겨냥하는가 | segment/layer/lead별 residual 근거 | 단순 변형으로 분류 |
| 공식 정보가치 | 결과가 다음 결정을 바꾸는가 | 양·중립·음성 결과별 후속 규칙 사전등록 | 제출 슬롯 보류 |
| Public 과적합 방어 | Public 최고점과 과학적 주장이 분리됐는가 | champion과 private-ready 후보를 별도 기록 | 우승 주장 금지 |

**두 종류의 승격을 분리한다.** `Public champion update`는 제한된 대회 최적화 목적의 공식 곡선 활용을 허용한다. `Scientific/private-ready`는 다중 시간창·station/layer/lead 안정성, bootstrap 불확실성, 누수 없는 외부검증을 추가로 요구한다. 한 후보가 전자를 통과해도 후자를 통과했다고 쓰지 않는다.

## P1 — anchor를 보존한 장구간 누락 복구

### 확인된 병목

- 현재 공식 최고는 e150 full-station Router union이며 F1 `0.833248`, 점수 `28.901363`이다.
- e150은 단순 마지막 epoch 선택이 아니라 e120/125/130/145/150 중 Q3+Q4 pooled 최고였다. checkpoint union·majority·intersection은 더 나빴다.
- local pooled F1은 anchor `0.902917` → candidate `0.906804`, `+0.003887`; 90% bootstrap CI는 `[-0.01315, +0.02114]`였다. 공식 F1 개선은 `+0.015375`로 더 컸다.
- 잔여 오류는 48시간 초과 장구간 recall `0.7090`, offset recall `0.7327`, drift recall `0.7245`에 집중된다. 특히 I-ORS/L1, G-ORS/L1, S-ORS/L2가 크다.
- e150 확률이 `1e-5~5e-5`인 누락은 threshold만 내려 복구할 수 없다. 새 proposal scorer가 필요하다.

### 1순위: F1-aware long-event change-point rescue

anchor e150의 양성 구간은 그대로 보존한다. PELT/CAPA/CPOP 계열 change-point로 과거신호 기반 후보 interval을 만들고, cross-fit score가 있는 구간만 추가한다. 삭제는 첫 버전에서 금지하고 G/L1, I/L1, S/L2에 한정한다.

| 항목 | 설계 |
|---|---|
| 입력 | past-only 286개 특징 또는 P1 causal feature subset, anchor probability, change-point 통계 |
| 학습 | outer fold마다 proposal 생성과 score fit을 분리; target 구간 미노출 |
| 목적 | 후보 interval의 `ΔTP`, `ΔFP`, 예상 pooled F1 변화 |
| 선택 | 겹침 제약이 있는 interval scheduling; station×level cap |
| 안전장치 | anchor 양성 삭제 금지, 변경행 manifest, zero-add no-op arm |
| 승격 | pooled와 3개 핵심 cell 개선, day/station bootstrap, FP budget 준수 |
| 계산 | 구현 3–6시간, CPU 검증 1–3시간 |

local zero-FP oracle은 959행을 더해 pooled F1을 `0.906804→0.955818`까지 올릴 수 있었지만 이는 진짜 모델 성능이 아니다. 의미는 “오류공간의 상한이 남아 있다”는 것뿐이다. past-only scorer가 이 상한의 얼마를 회수하는지는 아직 미검증이다.

정확한 구현 표면: `src/p1_qc/long_event_segment_proposal_rescore.py`, 신규 runner/config, anchor `artifacts/p1_mstcn_e150_full_deployment_20260827_v1`.

### 후속 구조 후보

| 후보 | 상태 | 장점 | 핵심 위험 |
|---|---|---|---|
| TE-TAD식 direct interval set predictor | GO-MEDIUM | frame classifier와 다른 오류공간, boundary/actionness 직접 학습 | 기존 tinygrad sanity recall 0.8235<0.90, PyTorch 재구현 필요 |
| LTContext dense segmentation | GO-EXPLORATORY | 장거리 temporal context를 직접 모델링 | 라이선스와 재현비용 확인 필요 |
| checkpoint/SWA 평균 | HOLD | 저비용 | 이미 checkpoint set operation이 열세, 독립성이 낮음 |

### P1 공식 슬롯 규칙

첫 공식 검증은 `anchor + long-event rescue` 단일 후보가 local blind gate를 통과했을 때만 쓴다. 두 번째는 같은 proposal에 threshold만 바꾸지 말고 cell support를 분리하는 기전 probe로 쓴다. 세 번째는 구조적으로 독립적인 direct interval predictor 또는 no-op 안전후보에 배분한다.

## P2 — 가장 빠른 수확과 가장 명확한 구조 결함

### 확인된 공식곡선

| 후보 | Public RMSE | 공식 점수 | 이전 대비 |
|---|---:|---:|---:|
| 기존 U | 0.535727 | 26.611283 | 기준 |
| OAS 10% | 0.507628 | 26.963865 | +0.352582점 |
| OAS 20% | 0.483661 | 27.264587 | +0.300722점 |

같은 OAS 계보인 10→20%가 단조 개선했다. 전체 26,061행 예측벡터와 두 공식 RMSE를 이용한 기하 재구성에서 `alpha=0.40`의 중심 RMSE는 약 `0.4486`, 느슨한 보수구간은 `[0.4276, 0.4687]`이다. 현 최고 `0.483661`보다 구간 상단도 낮지만, PAVA 및 계보 오차 때문에 공식 결과를 보장하지는 않는다.

### 1순위: OAS alpha=0.40

| 항목 | 사전등록 |
|---|---|
| 후보 | 기존 U 60% + 계절 국소 OAS T/S 조건부 프로파일 40% |
| 계보 | 기존 `build_p2_seasonal_oas_submission_20260827.py`를 새 deploy tag/config로 재사용 |
| 선행 QA | alpha10/20 재생성 hash·행·key·PAVA·범위 일치 |
| 기대 | 중심 약 +0.44점, 느슨한 하한 약 +0.19점 |
| 중단 | 계보 불일치, alpha20 비재현, changed-row가 예상과 불일치 |
| 계산 | CPU 10–20분, 재학습 없음 |

`alpha=0.40`은 새 구조가 아니라 공식에서 검증된 방향의 exploit이다. Public 최고점 갱신 후보로는 강하지만 scientific/private-ready 주장에는 부족하다.

### 구조적 결함: layer 번호가 물리 수심이 아니다

현재 OAS는 `(1,5,6,7)` layer 번호를 같은 공분산 변수로 pivot하고 layer 8을 제외한다. 그러나 중앙 수심은 2024 layer7 약 `49.076m`, 2025 layer7 약 `39.471m`, 2025 layer8 약 `49.388m`이다. 즉 `temp_7/psal_7`은 서로 다른 물리 수심을 묶고, 2025의 진짜 약 49m 채널을 버린다. 완전 profile만 남기는 `.dropna()`는 부분관측 정보도 버린다.

### 구조 1순위: 물리 수심 정렬 conditional MFPCA/PPCA

- T/S를 물리 `depth` 또는 `nominal_depth` 위 B-spline 함수로 표현한다.
- 2024 layer7과 2025 layer8을 같은 약 49m support로 정렬한다.
- 2025 layer7의 약 39m는 별도 선택 support로 둔다.
- 부분관측 profile에서 conditional PC score를 추정하고 measurement-error diagonal을 포함한다.
- 단일 MFPCA/PPCA부터 시작하고 잔차가 명확할 때만 K=2–3 mixture로 확장한다.

검증은 target layer 2–4 동시 masking, 월 블록, 2024→2025 및 2025→2024 year-out, KST day bootstrap으로 고정한다. 기존 layer-index OAS와 동일 mask에서 직접 ablation한다. 신규 표면은 `src/p2_restore/depth_registered_mfpca.py`, `configs/experiments/p2_depth_registered_mfpca_v1.json`, `tests/test_p2_depth_registered_mfpca.py`다.

### P2 공식 슬롯 규칙

1. `alpha=0.40`: 최고점과 α-반응곡선을 동시에 갱신하는 exploit.
2. 구조 모델이 준비되면 depth-registered MFPCA; 준비되지 않으면 `alpha=.40 + layer3만 .60` 기전 probe.
3. 두 번째 probe와 직교하도록 `layer4만 .60` 또는 구조 모델과 alpha40의 cross-fit blend.

PAVA가 층 간 결합을 만들 수 있으므로 layer probe는 바뀐 행뿐 아니라 다른 layer까지 변경되는지 반드시 QA한다.

## P3 — 수렴한 alpha축을 버리고 전이학습 backbone 교체

### 현재 상태

- Public 최고는 `alpha=-10.217432`, RMSE `0.583892`, 점수 `24.066168`이다. `-10.235445`와 사실상 같고 `-12`는 악화했다.
- ERA5 raw/derived는 각각 `363/363`, `.partial=0`, combined `262,917행`, past-only `286특징`, preflight PASS다.
- 과거 고정 one-shot은 CatBoost가 없는 다운로드 전용 interpreter에서 실행되어 첫 import에서 끝났다. 모델 fit·예측·gate는 0회였다. 기존 lock은 수정하지 않고 새 experiment ID를 사용해야 한다.
- KMA source-vs-local domain AUC `0.996779`, effective sample ratio `0.0008406`이므로 raw source pooling은 위험하다.
- 기존 GRU/TCN은 inner-best가 2–4 epochs였고 고정 30 epochs가 더 나빴다. 수백 epoch 자체는 전략이 아니다. best checkpoint와 수렴곡선이 전략이다.

### 1순위: Chronos-2 multivariate past-covariate transfer

| 단계 | 비교축 | 목적 |
|---|---|---|
| A | zero-shot Chronos-2 | 학습 없이 구조적 적합성 확인 |
| B | local-only LoRA | 작은 local 분포 적응 효과 |
| C | ERA5→local LoRA | 외부 forcing representation의 추가가치 |
| D | 고정 incumbent blend | calibration과 안전한 no-op arm |

공식 `Chronos2Pipeline.fit`에는 `finetune_mode="lora"`, `lora_config`, validation 입력, best-model reload가 있다. 그러나 `peft`가 없으면 경고 후 full finetune으로 바뀔 수 있고, 설치된 PyPI가 main보다 뒤처질 수 있다. 따라서 official commit SHA pin, `inspect.signature`, `peft` import, 1-step GPU smoke를 lock 전에 통과해야 한다. `eval_loss` checkpoint와 P3 RMSE 순서가 다를 수 있으므로 validation RMSE로 별도 best-state를 기록한다.

신규 표면: `requirements-p3-tsfm.txt`, `src/p3_wave/chronos2_transfer.py`, 신규 runner/config/test. 예상 시간은 zero-shot 수분–1시간, LoRA 1–3시간이다.

### 2순위: ERA5 masked-pretrain PatchTST

ERA5 2014–2020 train, 2021/2022/2023 source validation을 고정하고 masked reconstruction으로 representation을 학습한다. local은 사전등록된 3개 historical window와 78h embargo를 사용한다. dense energy trajectory 또는 physics auxiliary head는 본체 후보의 ablation으로만 붙인다. 예상 GPU 시간 2–6시간이다.

### P3 공식 슬롯 규칙

첫 슬롯은 Chronos-2 zero-shot 또는 LoRA 중 blind local gate와 lineage QA를 통과한 하나에만 쓴다. 두 번째는 ERA5→local과 local-only의 차이를 식별하는 기전 probe로 쓴다. 세 번째는 PatchTST 또는 incumbent blend로 배분한다. 공식 변화가 `+0.10점`을 넘으면 같은 구조를 계속하고, `±0.05점`은 중립으로 기록하며, `−0.30점` 이하는 해당 구조를 중단하는 운영 규칙을 사전등록한다. 이 숫자는 대회 운영용 기준이지 통계적 보편법칙이 아니다.

## 제한된 제출 기회의 최적 배분

하루 문제당 3회라는 예산은 각 문제에서 같은 강도의 세 변형을 내는 용도가 아니다.

| 슬롯 | 목적 | 필요한 후보 성격 | 결과가 바꾸는 결정 |
|---|---|---|---|
| Exploit | Public 최고점 갱신 | 기존 공식곡선의 가장 유망한 연장 | champion 업데이트 여부 |
| Mechanism | 구조 가설 식별 | orthogonal error-space 또는 layer/cell/lead 분해 | 다음 모델 계열 선택 |
| Guard | 손실 제한·곡선 bracket | no-op arm, 안전 blend, 반대편 probe | 계속/중단/구간 축소 |

### 권장 실행 순서

1. **P2 alpha=.40를 먼저 재현·QA**한다. 20분 내외로 다음 공식슬롯의 가장 높은 기대가치를 확보한다.
2. **P1 long-event rescue를 구현**한다. anchor 삭제 금지와 zero-add arm으로 최대 손실을 제한한다.
3. **P3 Chronos-2 환경 smoke와 zero-shot을 실행**한다. 설치/버전/LoRA 모드가 확인되기 전 장시간 학습을 금지한다.
4. 병렬로 **P2 depth-registered MFPCA**를 만든다. OAS40 이후의 구조적 후속이다.
5. 공식 결과가 돌아올 때마다 문제별 evidence ledger만 갱신한다. 전 문제 공통 보정계수를 재학습하지 않는다.

### 예상 시간과 컴퓨터 자원

| 작업 | 벽시계 예상 | 주 자원 | 제출 준비 시점 |
|---|---:|---|---|
| P2 alpha=.40 재현·QA | 10–20분 | CPU | 즉시 |
| P1 change-point rescue v1 | 4–9시간 | CPU 중심 | blind gate 후 |
| P3 Chronos-2 smoke/zero-shot/LoRA | 1–4시간 | RTX 5090 | 버전 preflight 후 |
| P2 depth MFPCA/PPCA | 1–3시간 | CPU, <4GB RAM 예상 | year-out ablation 후 |
| P3 PatchTST masked pretrain | 2–6시간 | GPU | Chronos 결과와 비교 후 |

## 무엇을 지금 제출하면 안 되는가

- P1 zero-FP oracle: 상한 분석이지 past-only 모델이 아니다.
- P1 단순 threshold 하향: 누락 확률이 너무 낮아 FP만 늘릴 가능성이 크다.
- P2 미제출 deep stack 단독: OOF `0.745814`, LOBO `0.775660`으로 현 OAS 계보보다 우선순위가 낮다.
- P2 alpha를 0.40 밖으로 무제한 외삽: 보수 상한이 다시 나빠진다.
- P3 기존 alpha 미세조정: `+0.000001점`에서 축 수렴이 확인됐다.
- P3 raw ERA5 pooling: source/local domain 분리가 거의 완벽해 음성전이가 예상된다.
- best checkpoint 없이 수백 epoch 마지막 모델: 계산량은 크지만 승격 근거가 아니다.

## 연구의 한계와 정직한 해석

1. 공식 Public 결과는 제한된 hidden subset이며 repeated probing에 적응할 수 있다. Public champion은 Private 성능 보증이 아니다.
2. P2 alpha=.40의 수치는 관측된 두 점과 예측벡터 기하에서 온 extrapolation이다. 보수구간도 lineage·postprocess model error를 완전히 포함하지 않는다.
3. P1 oracle은 현실적인 성능 추정이 아니라 오류공간의 존재 증거다.
4. P3 Chronos-2는 최신 공식 구현이 존재하지만 P3 데이터에서 zero-shot/LoRA 성능은 미관측이다.
5. 외부 논문의 개선률은 데이터셋과 채점법이 다르므로 이 대회의 점수 상승률로 옮기지 않는다.

## 1차 출처와 코드 근거

### 대회·검증 방법론

- 대회 공식 사이트: [Ocean AI Data Challenge](https://oceanaidata.org/). Public 리더보드와 대회 진행 정보의 1차 출처이며 세부 규정은 로그인 영역에 있다.
- Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 2010: [JMLR](https://www.jmlr.org/papers/v11/cawley10a.html). 반복 모델선택과 성능추정 편향을 구분하는 근거.
- Blum & Hardt, *The Ladder: A Reliable Leaderboard for Machine Learning Competitions*, ICML 2015: [PMLR](https://proceedings.mlr.press/v37/blum15.html). leaderboard feedback 과적합을 통제해야 하는 근거.

### P1

- Killick et al., PELT, JASA 2012: [DOI](https://doi.org/10.1080/01621459.2012.737745).
- Fisch et al., CAPA, *Statistical Analysis and Data Mining* 2022: [DOI](https://doi.org/10.1002/sam.11586).
- Maidstone et al., CPOP, *Journal of Statistical Software* 2024: [JSS](https://www.jstatsoft.org/article/view/v109i07).
- Kim et al., TE-TAD, CVPR 2024: [CVF paper](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html), [official code](https://github.com/Dotori-HJ/TE-TAD).
- Liu et al., TadTR: [arXiv](https://arxiv.org/abs/2106.10271), [official code](https://github.com/xlliu7/TadTR).
- Bahrami et al., LTContext, ICCV 2023: [official code](https://github.com/ltcontext/ltcontext).

### P2

- Yao, Müller & Wang, *Functional Data Analysis for Sparse Longitudinal Data*, JASA 2005: [author PDF](https://utstat.utoronto.ca/fyao/2005-jasa.pdf). 부분관측에서 conditional trajectory 추정의 기초.
- *Conditional multivariate functional PCA for the reconstruction of temperature and salinity profiles partially sampled by deep-diving marine mammals*, arXiv 2026: [arXiv:2608.05376](https://arxiv.org/abs/2608.05376). 물리 수심상의 bivariate T/S conditional MFPCA 설계 근거. v1 preprint이며 보고 효과를 P2에 직접 이전하지 않는다.
- Tipping & Bishop, *Mixtures of Probabilistic Principal Component Analysers*, Neural Computation 1999: [author PDF](https://www.miketipping.com/papers/met-mppca.pdf).
- Nie et al., ImputeFormer, KDD 2024: [official code](https://github.com/tongnie/ImputeFormer), [DOI](https://doi.org/10.1145/3637528.3671751).
- LSTI, TMLR 2025: [OpenReview](https://openreview.net/pdf?id=9NVJ0ZgEfT), [official code](https://github.com/iLearn-Lab/TMLR25-LSTI).

### P3

- ERA5 official dataset: [Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels), [ECMWF ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5).
- Ansari et al., Chronos-2: [arXiv:2510.15821](https://arxiv.org/abs/2510.15821), [official repository](https://github.com/amazon-science/chronos-forecasting). Multivariate·covariate zero-shot forecasting과 공식 `Chronos2Pipeline` 구현 근거.
- Nie et al., PatchTST: [arXiv:2211.14730](https://arxiv.org/abs/2211.14730), [official repository](https://github.com/yuqinie98/PatchTST).
- Du et al., AdaRNN: [author paper](https://jd92.wang/assets/files/cikm21-adarnn.pdf).
- Sun & Saenko, Deep CORAL: [arXiv:1607.01719](https://arxiv.org/abs/1607.01719).
- Ganin et al., Domain-Adversarial Training: [JMLR](https://www.jmlr.org/papers/v17/15-239.html).

### 내부 재현 근거

- `reports/HACKATHON_HANDOFF_2026-08-28.md`
- `reports/next_day_breakthrough_deep_research_20260827_v1/local_official_calibration.json`
- `artifacts/p1_mstcn_checkpoint_diagnostic_20260827_v2/fixed_epoch_150_metrics.json`
- `scripts/build_p2_seasonal_oas_submission_20260827.py`
- `scripts/qa_p2_seasonal_oas_alpha20_20260827.py`
- `reports/p3_era5_context_transfer_v1/p3_era5_context_transfer_report_ko.md`

## 최종 의사결정

**다음 실진행은 P2 alpha=.40 재현·QA, P1 long-event proposal rescoring, P3 Chronos-2 environment smoke의 세 갈래가 맞다.** 이 중 P2만 즉시 공식 제출 후보이며, P1·P3는 새 blind gate를 통과하기 전에는 제출 후보가 아니다. P2 depth-registered MFPCA는 alpha=.40 이후 가장 먼저 개발할 구조 모델이다.

이 보고서는 연구와 다음 실행 순서를 고정했으며 어떤 CSV도 생성·제출·업로드하지 않았다.
