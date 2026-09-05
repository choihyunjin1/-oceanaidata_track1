# Ocean AI Data Track 1 — 정찰 결과와 점수 개선 계획 (2026-09-05, 최종안)

## Context

사용자는 해양 해커톤(Ocean AI Data Track 1: P1 수온 이상탐지 / P2 중간층 수온 복원 / P3 유의파고 예측)에 참가 중이며, 여러 AI와 풀어온 결과를 GitHub
`choihyunjin1/-oceanaidata_track1`(로컬 원본 `C:\Users\cedis\PycharmProjects\PythonProject`, HEAD 535f94a 동일)에 정리해 두었다. 요청: (1) 저장소를 정찰 자산으로 면밀히 읽고
취약점·이상한 점·한계를 기록·보관, (2) 그 기록으로 **공식 점수가 더 잘 나오는 코드**를 작성하고 **최종 재현 패키지까지** 문제없이 완성. 마감 **2026-09-07(모델 제출)**, 세 문제 병렬.
답안 업로드·최종 모델 제출은 사용자가 직접 수행하고, 규정 위반(KMA/ERA5) 제출 8건은 새 clean 후보 채점 후 사용자가 삭제(목록은 내가 준비).

환경: `.venv-p1`(Python 3.12, torch 2.13 cu130, lightgbm 4.7, xgboost 3.4, catboost 1.2.10), RTX 5090 32GB, Ryzen 7800X3D 8코어, 64GB RAM, 디스크 700GB 여유.
원본 데이터(읽기 전용): P1 `C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly`, P2 `...\p2\데이터셋_P2\P2_profile_restore`, P3 `...\p3\데이터셋_P3\P3_wave_forecast`.

정찰 산출물(계획 디렉터리에 임시 보관, 실행 단계 0에서 저장소로 복사):
`C:\Users\cedis\.claude\plans\mellow-wishing-storm-agent-{a04ef53bc24dcb675 (P1), a0668d64e21437bd0 (P2), a49742ee4ce52d031 (P3), ac2f059db52e41f9a (제출이력·채점식)}.md`
설계 패널 결과: 워크플로 `wf_6d6c1cfd-1fd` journal(설계자 6 + 심판 3; 일부는 재실행 중).

## 1. 규정·일정 (공지 원문 직접 확인)

- **09-02 공지**: 리더보드 점수를 역산해 계수·임계값·파라미터를 정하는 것 금지. 모든 파라미터는 배포 데이터로 적합. 이런 상수는 **재현 검증 1항(과도한 상수 리터럴)·4항(학습 산출물 제거 후 예측 재생성)** 에서 확인되며 **검증 통과 팀만 본선 진출**. Public≈30%, **최종 순위는 Private**. 점수를 참고 지표로 보는 것은 허용.
- **08-31/09-01 공지**: 외부 데이터·사전학습 가중치 금지(합성 전용 TabPFN만 4조건 예외). 상위팀은 인터넷 차단 환경에서 재현 검증, 6시간 이내.
- **08-12 공지**: 09-07까지 모델(코드+가중치) 제출. **모델 제출 즉시 최종 확정, 이후 답안 업로드 불가**. 운영진은 모델이 업로드한 답안을 재현하는지 검증. 답안 업로드 문제당 하루 3회.
- 리더보드(09-05): 우리 9위 81.13(표시, P3는 규정 위반 계보 포함; clean 80.99). 1위 84.97(P1 32.26≈F1 0.96). 한계효용 P1 +0.01F1=+0.266점, P2 −0.01℃=+0.125점, P3 −0.01m=+0.159점.

## 2. 정찰 결론 — 현재 세 후보는 모두 재현 검증 탈락 위험 (교체 필수)

| 문제 | 현재 최고(Public) | 문제점 |
|---|---|---|
| P1 | F1 0.8335 | 양성 6,396행 중 **94.8%가 동결 CSV `router_anchor.csv`**(08-26 제출물; XGBoost O + LightGBM B + 정점·층 라우팅)에서 오고 학습 코드·가중치가 패키지에 없음. +2행 수동 패치(공개 점수로 선택). MS-TCN e150은 333행 add-only, bf16/cuDNN 비결정 |
| P2 | RMSE 0.424 | **80%가 동결 CSV `bin17_anchor.csv`**(공개 점수로 층별 α·OAS 혼합비·rank-1 강도·bin 등 ≈6개 스칼라 적합), 생성 코드 없음. 20%만 재현 가능한 DeepSets(5,889 파라미터) |
| P3 | RMSE 0.5839 | `O + α(A−O)`, **α=−10.217은 공개 점수 5개로 적합한 상수**(효과: (O−persistence) 1.65배 증폭). 정직한 O는 0.607. 로컬 OOF는 반대 부호 → Private 하방 −1.4점 위험. 09-02 공지 직접 위반 |

### 문제별 검증된 사실 (직접 계산 + 설계자 정찰)
- **P1**: 주입 이상은 자연 변동보다 훨씬 큼(자연 연속 동일값 최대 4행 vs flatline ≥12행; noise 차분 std 2.96℃ vs 자연 ≤0.22; spike |Δ|≥2℃; offset |잔차| 중앙 2.7℃; drift 종료 복귀점프 큼). 주입은 **temp 전용**(psal/depth 불변 → psal은 "깨끗한 쌍둥이"). 양성 구성 noise 29/drift 27/offset 23/flatline 19/spike 0.3%. 손실 본체는 offset/drift **장기 이벤트 내부 행 미탐**(recall 0.65; FN의 92.7%). test는 2026-01~06(5~6월 55%), 세그먼트 단절이 train보다 훨씬 심함(장기창 결측 18~32%); 기존 fold는 4~12월만 → 1~6월 검증 공백. 5~9월 S-ORS 중층 자연 |dT| p99 4~5℃ → FP 위험.
- **P3**: test 사례는 폭풍 **onset** — "hs≥1.5 최초 교차 + 78h 간격" greedy가 test 분포를 재현(정시 :00 격자 anchor 기준이 가장 정합). 기존 학습은 hs≥1.5 전 시각(24,360)·GPU 비결정·persistence 수축·라우터. 설계자 실측: 선택정합 재가중은 이득 없음(잡음 이내), 풍부한 특징도 onset에서는 이득 ≈0 → **앙상블 다양성·결정론**에 투자. 정직 모델 Public 기대 0.59~0.60.
- **P2**: 2024-09/10 가상 가림 — 선형보간 0.971, T1 복사 0.593, 실제수심 보간 0.880, LightGBM(y−T1 목표, T5 마스크 증강) 0.563; 2025-11/12(완전 혼합)는 T1 복사 0.068 vs GBM 0.28~0.37(성층 과대예측). **test의 29.3%(10/14~10/31)는 T5가 17일 연속 결측인데 강성층(T1−T6 7.8℃)** → 점수를 결정하는 최난 구간. hidden 행 실제 depth 99.3% 존재(미사용).
- **공통**: 로컬→공식 방향 일치율이 낮았던 원인 = 노출 검증면 재사용, test 분포와 다른 fold, LB 프로빙, 비현실 comparator(P2). Public은 30% 표본이라 P3 0.001~0.01m 차이는 식별 불가.

## 3. 접근 원칙

1. 기존 챔피언 파일·패키지(`artifacts/official_final_submission_20260905/`)는 보존, 덮어쓰지 않음. 새 코드는 `src/ocean_v2/{common,p1,p2,p3}/` + `configs/ocean_v2/*.json` + `scripts/ocean_v2/*.py` + `tests/ocean_v2/`.
2. **LB로 파라미터를 정하지 않음.** 보정·혼합·임계값·분위 상수는 모두 `train`이 CV OOF에서 계산해 `derived_constants.json`으로 저장, `predict`는 읽기만(재현 검증 1·4항 대응). 코드 리터럴은 설정 JSON의 구조·물리 상수만.
3. 결정론: CPU 트리 모델(`deterministic=True`, 고정 seed·thread), 딥모델은 CPU/deterministic 설정, CSV 반올림(P2 5자리, P3 4자리). GPU 비결정 학습 배제.
4. 배포 데이터만, 재현 ≤6h(문제당 목표 ≤2h), 클린룸 재생성 검증.
5. 후보 사다리: **안전 기준선(Day 1 확정, fallback)** → 사전등록 CV 게이트를 통과한 단계만 누적. Public 업로드는 sanity check(기대 범위 이탈 시 버그 점검)로만.

## 4. 문제별 설계 (설계 패널 종합)

### P3 — `src/ocean_v2/p3/` (재현 ≈1.5~3h CPU)
- **데이터/anchor**: `p3_wave.data`(로더·10분 격자·`build_anchor_table` 20분 stride=24,360 anchor) 재사용. anchor 메타: `is_hourly`, `first_cross`, `onset3`, `hs_min_3/6/12h`, `rising`, `episode_id`(`revin_patch.assign_storm_episodes_from_wave`). 운영진 선택 복제(정시 anchor greedy 78h → ≈272 사례)는 검증 전용.
- **특징**: `p3_wave.features.summarize_context` + `compact_feature_columns`(591) + onset/위상 블록 ≈25(1.5 통과 후 경과, hs 최소/최대·피크 경과, 가속, wave_age proxy, 풍향 회전, 기압 최소 경과 등). 절대시각·월 없음, 사례 내부만.
- **CV**: 3개월 6블록(2024-01~2025-06), purge ±78h + episode 공유 제거 + footprint 무겹침 assert. 평가 = 블록 내 정시 anchor 전부, 지표 = 선택정합 밀도가중 SM-RMSE(1차) + greedy-272 RMSE(2차) + 블록/정점/리드/persistence 비교 + episode bootstrap CI90.
- **모델 B0(안전)**: LightGBM pooled(cat station·lead) + CatBoost CPU pooled(기존 검증된 하이퍼) + XGBoost per-lead, 잔차 목표 `target−hs0`, 균일 가중, seed 평균(CV 3, 최종 5) → 세 멤버 등가평균 → clip[0,30] → 4자리 반올림. persistence 수축·라우터·α 없음.
- **사다리**: L2 wave-only 멤버(2024 I/S 기상 전무 헷지) + log-ratio 멤버 → L3 리드별 NNLS(등가와 50% 수축) → L1 표본가중 3안 → L4 승법 보정 a_L(OOF WLS, CI90이 1 배제 & 0.7~1.3일 때만; 가법 b_L=0 고정) → L5 소규모 HPO. 게이트: ΔSM<0 ∧ CI90 상한<+0.001 ∧ worst-block Δ≤+0.005 ∧ greedy Δ≤+0.005 ∧ 정점 Δ≤+0.01.
- 기존 O(0.607)를 같은 검증면에서 재평가해 비교 기록.

### P1 — `src/ocean_v2/p1/` (재현 ≈1.5h, 중첩 확인 포함 ≈3h, CPU)
- **재사용**: `p1_qc.data`(로드·세그먼트), `p1_qc.features.build_features(offline)` 80열 중 depth 계열 4열 제외(G-ORS 2026 depth 전결측), `p1_qc.postprocess`(gap close/short-run 제거; 히스테리시스는 벡터화 재작성), `p1_qc.metrics/validation/submission`.
- **신규 특징**: (B) 시간 기반 간격 허용 창(24h/72h/168h/336h median·MAD·robust z, 좌/우 비대칭 잔차, `twosided_min`, 국소 dT 스케일) — test 장기창 결측 18~32%→0%; (C) psal 쌍둥이(psal 잔차·T–S 회귀 잔차·dT/dS 비); (D) 단계/브래킷 변화점(step_k, 좌/우 최근 큰 단계, `bracket_closure`, `bracket_level_z`, drift ramp 역외삽 잔차; 임계는 fold train 자연행 99.9 분위로 학습 시 계산·저장); (E) 노이즈(zigzag, 국소/계절 std 비); (G) 단절·관측밀도; (I) Stage-2 문맥(Stage-1 OOF 확률의 시간창 집계, 좌/우 고확률 거리, 브래킷 결합).
- **CV**: 반기 4블록(2024H1·2024H2·2025H1·2025H2), 양측 purge 15일, 양성 run 시작 블록 귀속, 인코더·분위 상수 fold train 전용. 1차 지표 `f1_season`(1~6월 블록 풀링), worst-block, 유형별 recall, 계열별 예측률, 정상일 FP, 일-블록 bootstrap CI, **단절 스트레스 표면**(test gap 분포로 인위 단절 후 재계산).
- **모델 C0(안전)**: Stage-1 LightGBM×3 seed + XGBoost×2 seed(기존 O 레시피) 평균, 하드 규칙 "동일값 연속 ≥12 → 1"(자연 FP≈0 검증), 벡터화 히스테리시스·gap 브리징·min run·singleton 정책을 OOF 격자에서 선택(중첩-정직 추정 병기). Stage-2 LightGBM×3(문맥 스태킹)은 C4 단계.
- **사다리**: C1 시간창 → C2 psal → C3 브래킷/노이즈(가장 큰 레버, offset/drift 내부) → C4 Stage-2 → C5 단절 증강 학습 → C6 singleton 정책. 게이트: Δf1_season>0 ∧ Δworst≥−0.005 ∧ CI90 하한>−0.002 ∧ Δstress≥0. 기대 정직 CV 0.86~0.90, Public/Private 0.82~0.87.
- MS-TCN은 사용하지 않음(비결정·자체 gate 실패·GPU). (심판 결과에 따라 CPU 결정론 재학습 옵션만 검토.)

### P2 — `src/ocean_v2/p2/` (재현 ≈1~1.5h CPU)
- **데이터**: 10분 UTC 완전 격자(105,264 시각) wide 패널, **수심 슬롯**(4/19/30/39/49 m; 연도별 layer↔수심 불일치 해소), 목표 수심 = 실제 depth(결측 시 nominal), 조직 baseline 정확 재현(clamp 보간; 기존 외삽 버그 제거), hidden 26,352행 NaN assert, `year`/`elapsed_days` 특징 금지.
- **증강(학습행, 고정 RNG)**: T5(+S5) **연속 블록** 아웃티지(1~20일, ≈30% 시각), psal_1 연속 블록, 단기 무작위 마스킹 — 시간문맥 특징도 마스킹 후 재계산.
- **특징 ≈80**: 슬롯별 T/S/z/presence, 목표 `zt_real/zt_dev`, 기준(`base_nom/base_real/T1/deepT/frac_z/scale/grad`), 층간 차분, **혼합층 깊이 교차점 h05/h10/h20**과 `zt−h`, 시간문맥(±1/6/12/24/72h 차분, 이동 std·range·3일 추세; 복원 문제라 미래 허용), doy/hour/M2·S2·K1·O1, 가용성, `T19_last/next·staleness`.
- **목표 3종**: `resid_lin`(y−base), `resid_T1`(y−T1), `frac`((y−T1)/(deepT−T1), clip) — 각각 LightGBM(결정론) 다중 seed, 디코드 후 결합. **T5-부재 전문가**(without_t5 특징만) 라우팅은 결측 여부(결정론 규칙).
- **CV**: 두 달 블록 6~8개(2024-05~2025-12), purge 7일, **테스트정합 T5 마스크**(각 블록 마지막 17일 연속 제거) 변형을 주 지표로, 아웃티지 아날로그 fold(2024-10-08~31 T5 마스크 + 9월 psal_1 마스크). 지표: pooled/층별/T5 유무별/14일 bin별 RMSE·편향, 헤드라인 Composite = sqrt(0.71·pooled² + 0.29·outage²)(0.29=7,621/26,061 자료 유래). 비교군: nominal 보간·실제수심 보간·T1 복사·DeepSets 이식본.
- **모델 R0(안전)**: 3목표 LGBM × 5 seed 등가평균 + T5 전문가 + envelope[min,max](T1, deepT) 클립 + 3층 PAVA(T5 결측 시 T6 endpoint).
- **사다리**: R1 아웃티지 특화(마스크 길이 확장·staleness·T1 수축 s_L OOF) → R3 10 seed + NNLS 스택(50% 수축) → R4 소격자 HPO → L3 잠재 MLD 모델(학습행 8층 프로파일로 라벨, 내부 OOF 스태킹) → 아날로그 연도 특징(r_clim) → R6 DeepSets(CPU 결정론, 실제수심) / CatBoost 다양성 → R7 1h 평활. 게이트: Composite 개선 ∧ B2(2024-09/10)·B8(2025-11/12)·M1 어느 것도 +0.01℃ 이상 악화 없음; |Δ|<0.003이면 단순한 쪽.

### 공통 `src/ocean_v2/common/`
경로 해석(`P?_DATA_DIR`/CLI), 입력 SHA manifest, 결정론 seed/thread 설정, 러닝타임 영수증, submission validator(배포 `score.py` 스키마 동등 + 키 순서), 상수 리터럴 감사 스크립트(predict 경로의 숫자 리터럴 목록 출력).

## 5. 실행 단계 (지금 09-05 ~ 09-07 12:00 완료 목표)

### 단계 0 — 정찰 기록 보관 (30분, 메인 루프)
`reports/claude_recon_20260905/{P1,P2,P3,SUBMISSIONS_AND_SCORING}_recon.md` 복사 + `00_SUMMARY.md`(취약점·한계 레지스트리 통합, 공지 원문 요약, 리더보드·제출관리 스냅샷, 삭제 대상 제출 목록 초안, 설계 패널 요약).

### 단계 1 — 골격 + 안전 기준선 (Workflow: 문제별 구현 에이전트 3 병렬 + 적대적 검토 2) — 09-05 오후~저녁
- 각 구현 에이전트: 패키지 골격 → 단위 테스트(누출·purge·결정론 2회 실행 SHA 동일·hidden NaN) → CV 실행(백그라운드 프로세스 + 로그, 에이전트 중단에도 학습은 지속) → 안전 기준선 전체 학습 → `submissions/claude_v2/<P>/<candidate>/{csv, sha256, cv_report.json, validator.json}`.
- 검토 에이전트: 누출 경로, LB 상수, 재생성 가능성, 런타임, test 분포 정합을 적대적으로 점검 → 수정 지시.
- CPU 경합: P1(8 thread)·P2(4)·P3(8) 동시 실행 시 벽시계 2~3배 → 순차 우선순위 P3 → P2 → P1(P1 특징 캐시 먼저).
- 산출: 후보 CSV 3개(사용자 업로드 #1 sanity: P3 기대 0.56~0.63, P2 0.43~0.47, P1 0.79~0.82).

### 단계 2 — 개선 사다리 (Workflow 문제별, 사전등록 게이트) — 09-06
- 각 문제 사다리 실행 → 게이트 통과 단계 누적 → 후보 CSV(업로드 #2, #3). 어떤 Public 결과도 파라미터에 되먹임하지 않음(로그에 사전 지정 후보로 기록).

### 단계 3 — 최종 재현 패키지 + 클린룸 검증 — 09-06 저녁 ~ 09-07 오전
- `artifacts/official_final_submission_v2_20260907/{P1,P2,P3}/`: 기존 원자적 폴더 계약(01_data organizer_dataset 하드링크·INPUT_MANIFEST / 02_train TRAIN.ipynb+train_model.py / 03_model weights+MODEL_MANIFEST+derived_constants.json / 04_predict PREDICT.ipynb+predict_submission.py / 05_answer CSV+receipt / 06_submission FORM.json+FORMAT.md / 07_source ocean_v2+의존 모듈 allowlist / RUN_TRAINING.ps1·RUN_INFERENCE.ps1(`--clean-room`) / contract.json / README(환경·소요시간·seed·thread·상수 정책·허용오차)). 기존 빌더 `scripts/build_official_final_submission_20260905.py`의 골격을 ocean_v2용 새 빌더로 개조(기존 파일 수정 없이 `scripts/build_official_final_submission_v2_20260907.py`).
- 클린룸: 새 임시 폴더에 패키지만 복사 → 03_model 비움 → RUN_TRAINING → RUN_INFERENCE → 업로드한 답안 SHA 대조(트리 모델 byte-exact, 딥모델 허용오차) → 총 소요시간 기록 ≤6h. 업로드 ZIP ≤50MB 분할, 상수 리터럴 감사 통과, 외부자료·사전학습 0 확인, pytest.
- 사용자 전달물: 문제별 업로드 파일 집합·폼 값·최종 답안 SHA·삭제 대상 제출 목록. 순서: 답안 업로드 완료 확인 → (삭제) → 모델 최종 제출(사용자 클릭).

### Git
코드·설정·테스트·작은 보고서만 커밋 대상(사용자 요청 시). `artifacts/`, `submissions/` 비추적 유지.

## 6. 검증 방법
- 문제별 `cv_report.json/md`(블록·정점/층/유형·persistence/baseline 비교·CI), 배포 `score.py` 스키마 통과, 행 수·키 순서·유한값·범위 검사, 결정론(2회 실행 SHA 동일), 누출 테스트(hidden NaN, 목표층 미포함, purge 무겹침, P3 미래행 미사용), 상수 리터럴 감사, 클린룸 재생성 ≤6h.
- Public 업로드는 사용자가 수행하며 결과는 기대 범위 확인용으로만 기록.

## 7. 첫 6시간 작업 순서 (실행 즉시)
1. 단계 0 정찰 기록 복사·요약 작성(메인 루프, 30분). 설계 패널 심판 결과(`wf_6d6c1cfd-1fd` journal, 재실행 중)를 도착 즉시 읽어 각 문제 구현 에이전트의 검토 지시에 반영.
2. `src/ocean_v2/common` + 세 문제 골격·테스트 생성(구현 에이전트 3 병렬, 각각 자기 문제 폴더만 수정).
3. P3 B0 CV·전체학습(≈1.5h) → 후보 CSV → 사용자 업로드 #1(P3). 이어 P2 R0(≈40분) → 업로드 #1(P2). P1은 특징 캐시 생성 후 C0(≈1.5h) → 업로드 #1(P1).
4. 검토 에이전트가 각 안전 기준선의 누출·상수·재생성·런타임을 점검, 문제 시 수정 후 재실행.
5. 밤: 사다리 실험 백그라운드 실행(로그 기반), 다음 날 아침 게이트 판정.

## 8. 사용자 결정 반영
- 마감 09-07, 세 문제 병렬, 최종 패키지까지 제작, 업로드·최종 제출은 사용자, 규정 위반 제출 삭제는 새 후보 채점 후 사용자(목록 제공).
