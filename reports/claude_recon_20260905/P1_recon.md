# P1 (수온 센서 이상 탐지, row-F1) 정찰 보고서 — 2026-09-05

> 읽기 전용 정찰. 저장소·데이터 파일은 수정하지 않았고 raw 관측값은 한 행도 인용하지 않았다(집계만).
> 대상 후보: `P1_1_E150_PLUS_GI_SPIKE2` (Public F1 0.833548, 양성 6,396행, SHA 57844ef2…).
> (plan mode 제약으로 이 파일에 작성함. `reports/claude_recon_20260905/P1_recon.md`로 그대로 복사하면 된다.)

---

## 1) 현재 최고 파이프라인 정확한 재구성

### 1.1 최종 CSV의 실제 구성 (`scripts/final_submission_20260905/P1/predict_submission.py`)

```
final = anchor_preserving_union( router_anchor.csv , decode( mean_3seed(MS-TCN e150) ) ) + GI spike 2행 패치
```

| 구성요소 | 양성 행 | 출처 | 근거 |
|---|---:|---|---|
| ① router anchor (동결 CSV) | 6,061 | `03_model/decision_artifacts/router_anchor.csv` = 2026-08-26 공식 제출 `P1_1_EXPLOIT_DISAGREEMENT_ROUTER` (SHA 1b04e81c…) | `predict_submission.py:69-76`, `contract.json` `router_anchor_sha256` |
| ② MS-TCN e150 add-only 제안 | +333 (G 15 / I 80 / S 238, 35개 연속 구간) | 3-seed 체크포인트 실제 추론 → decoder → anchor와 OR | `predict_submission.py:32-76` |
| ③ GI spike 2행 패치 | +2 | `gi_spike2_patch.json` (S-ORS L5 2026-06-21 05:10, S-ORS L6 2026-06-15 18:50) | `predict_submission.py:79-92` |
| 합계 | 6,396 | SHA 57844ef2… 와 byte-exact 검증 | `predict_submission.py:100-102` |

**핵심 사실: 양성의 94.8%(6,061/6,396)는 MS-TCN이 아니라 2026-08-13~25에 만든 트리 모델(XGBoost "O", LightGBM "B")의 동결 CSV에서 온다.** MS-TCN은 장기 사건 333행만 얹는 add-only 보조 모델이다.

### 1.2 router anchor의 계보 (패키지 `07_source`에 학습 코드가 **없음**)

- 규칙(`scripts/run_p1_incumbent_preserving_mstcn_asrf_v2.py:873-896` `_current_router_bits`):
  `router = B; O-only 행 중 (G-ORS L1) ∪ (I-ORS L2) → 1 로 복원(217행: G 81, I 136); B-only 행 중 (S-ORS L1/L5/L6) ∪ (I-ORS L4) → 0 으로 제거(12행)`.
- **O** = offline XGBoost (`artifacts/runs/20260813T153038+0900_cv_378a4e89`): 80특징 offline 모드, `max_depth 7, lr 0.04, 700 iter, subsample/colsample 0.85, min_child_weight 20`; 후처리 hysteresis high 0.2 / low 0.1, close_gap 0, 최소 run 12행, singleton spike 보존(`selection.json`, `src/p1_qc/postprocess.py:276-350`). OOF F1 0.860371, Public 0.790709(양성 6,504).
- **B** = event-day balanced LightGBM 3-seed (Round B 2026-08-25, SHA decedb8a…): OOF 0.864670, Public 0.793710(양성 5,856). 레시피 요약은 `configs/experiments/p1_matched_budget_local_compare_20260825_v1.json` families·`artifacts/p1_matched_budget_local_compare_20260825_v1/report_ko.md`.
- 정점·층 라우팅 규칙은 **이미 노출된 421,032행 OOF(2025 Q2–Q4)에서 셀별 F1로 선택**했다(“로컬 시간검증에서 재현된 정점-층만”, round C `P1_제출정보.txt`).
- 집계 검증(내가 CSV 라벨만 비교): O-only 824 / B-only 176 / 공통 5,680; router = B + 217 − 12; champion = router + 335(333+2), 제거 0.

### 1.3 MS-TCN e150 (`07_source/scripts/run_p1_incumbent_preserving_mstcn_asrf_v2.py`, `src/p1_qc/ms_tcn_asrf*.py`)

**입력 특징 (165채널)**
- 캐시 `artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet` (80열, `src/p1_qc/features.py build_features`, mode=offline, group=(station, layer), rolling 3/6/12/24/48/72/168h + 7d/14d, 동일 시각 타층 peer 특징). test 캐시는 test.csv 단독으로 계산(`test_features.json source_sha256` = test.csv).
- `_feature_dependency_audit` (`run_p1…:796-870`)가 전구간 의존 4열(`nominal_depth_m, plateau_full_length, plateau_count, depth_regime`)을 제외 → 수치 74열.
- `RobustRowEncoder` (`ms_tcn_asrf_data.py:327-500`): train 전체로 median/IQR 스케일(±20 clip), NaN→0 + 결측 플래그 74, `row_valid`, `gap`(station/year/layer 단절 시작행), one-hot station 3 + layer 8 + depth token 4(현재행 depth_raw를 train 전용 33/67 분위 9.0 m / 28.74 m로 shallow/mid/deep/missing) = 74+74+2+3+8+4 = 165.
- 세그먼트: `SegmentLayout.from_aligned` (`ms_tcn_asrf_data.py:105-181`) — (station, year, layer) 내 정확히 10분 간격 run. 창 2048행·stride 512·오른쪽 0-패딩 + valid mask. 학습 창 = 양성 포함 창 전부 + 음성 창 2배(해시 결정적) → 1,707창; 추론은 전체 창(1,101) 중심가중 overlap-add(`stitch_center_weighted`).

**모델** (`ms_tcn_asrf.py`): MS-TCN++ 형 — 1×1 stem → dual-dilated residual 10층(dilation 1..512 오름/내림 쌍, kernel 3, **대칭 padding = 중심(양방향) 합성곱, 비인과**) → row head; refinement 3 stage(각 10층 dilated residual); boundary head(start/end 2) + type head(5)는 generator 특징에 연결. width 512, dropout 0.15, 52,568,587 params. 이론 수용장 generator 3,969행, 최종 10,107행(창 2048 전체 커버).

**손실** (`ms_tcn_asrf.py:443-594`): stage 가중 (0.25,0.25,0.5,1.0) × [BCE(pos_weight = clip(neg/pos,1,20) = **20.0**, 실제 비 23.2) + soft-dice] + 0.15×truncated log-prob smoothing(τ=4) + 0.2×boundary BCE(σ=3행 가우시안 타깃) + 0.2×type BCE(양성 행에서만, multi-hot).

**최적화**: AdamW lr 3e-4, wd 1e-4, warmup 10 ep, cosine→3e-6 **300 epoch 지평**(`_schedule_geometry` `run_p1…:1514-1534`), bf16 autocast, clip 1.0, batch 64, 27 step/epoch. **150 epoch에서 중단 → 종료 시 LR 1.6e-4(어닐링 안 됨)**, seed 당 4,050 step, RTX 5090 약 36.5분. seeds 20260827/20260839/20260863. 학습 데이터 = train.csv **전체 776,706행(제외 없음)**, encoder도 전체 train으로 fit (`p1_pipeline.py:108-114`).

**디코더** (`run_p1…:1348-1450`): `p_dec = p_row·(0.75+0.25·max(P_noise,P_offset,P_drift))`; 세그먼트별 hysteresis high 0.8 / low 0.4; 각 후보 구간의 start/end를 boundary 확률 argmax로 ±12행 스냅; 길이 ≥19행만 채택(상한 없음); `anchor_preserving_union` = max(anchor, proposal) → anchor 양성 제거 불가.

**e150/0.8 레시피의 선택 경위(중요)**: 사전등록 v2 프로토콜은 Q2에서 width 512·epoch **125**·thr **0.9**를 골랐고(882격자 중 고립 peak, ΔF1 +0.098), Q3/Q4 fresh refit 확인에서 **pooled −0.00514, Q4 −0.0315, CI90 [−0.028, +0.014] → NO_GO** (`reports/p1_incumbent_preserving_mstcn_asrf_v2/report-source.md`). 그 뒤 별도 `p1_mstcn_e150_full_deployment_20260827_v1`이 epoch 150·thr 0.8을 “완료된 역사적 진단으로 동결”이라고만 적고(`configs/experiments/p1_mstcn_e150_full_deployment_20260827_v1.json`, 근거 수치 미기재) 전체 train으로 3-seed 학습해 제출 → Public +0.015. 즉 **현 챔피언의 add-on은 자체 확인 gate에 실패한 계열**이며 선택 근거가 문서화돼 있지 않다.

**GI spike 2행의 정체**: “GI” = round E/F의 `P1_3_EXPLOIT_GI_NO_REMOVALS`(B + G/I 복원, 12행 제거 없음). `scripts/build_deadline_probe_set_20260828.py:117-131`은 (e150 union=0 ∧ GI=1)인 6행 = **router가 제거한 B-only 12행 중 e150이 되살리지 못한 6행**을 뽑고, 그중 B CSV의 `anomaly_type=="spike"`인 2행(S-ORS L5/L6, 6월)만 추가한 것이 챔피언이다. 즉 2행은 LightGBM B의 singleton spike 규칙 산출물이다.

### 1.4 노트북·실행 경로
`TRAIN.ipynb`는 기본적으로 인증된 체크포인트 해시만 검증, toggle 시 `train_model.py`로 별도 폴더 재학습. `PREDICT.ipynb` → `run_submission.materialize` → `predict_submission.predict`. `train_model.py:44`는 CUDA 필수.

---

## 2) 검증 설계와 약점

**설계** (`src/p1_qc/config.py:60-86`, `splits.py:99-208`): 고정 3 outer fold — 2025_q2(검증 04-01~07-01, train ≤03-24), q3(07-01~10-01), q4(10-01~12-11); purge 7일; 양성 run은 시작 시점 fold에 통째 배정. MS-TCN v2는 purge 21일(train max 03-10/06-09/09-09), split 후 windowing, Q2 선택 / Q3+Q4 확인, 21일 block bootstrap 10,000회. router 선택·수십 개 ablation은 모두 같은 421,032행 OOF(`artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet`) 위에서 했다.

**약점**
1. **계절 불일치**: 검증은 전부 2025-04~12, test는 2026-01~06. 2025 Q1은 항상 train, 2024 Q1은 S-ORS 단일 정점. Jan–Mar는 한 번도 holdout이 아니었다(train 월별 양성률은 0.6%~8.6%로 크게 변동: §6).
2. **선택면 노출**: 동일 OOF에서 router 셀 규칙, threshold, 882격자, Sobol 32점, 60여 사이클을 골랐다. 자기감사(`reports/research_process_self_audit_20260831_v1`)도 “선택 양수→확인/공식 음수 역전”을 3문제 공통으로 인정.
3. **로컬↔Public 괴리**: OOF 0.86–0.87 vs Public 0.79–0.83; 부호 역전 다수(router 로컬 +0.002→공식 +0.024; e150 로컬 NO_GO→공식 +0.015; label-shift EM 로컬 +0.0018→공식 −0.035).
4. **O/B OOF의 특징 지원 누출**: purge 7일(168h) < centered 특징 미래지원 168h + 과거지원 169h = 337h(`run_p1…:1084-1116`이 요구). O/B와 router 규칙의 0.86 F1은 경계 누출이 있는 fold에서 측정됐다(MS-TCN v2만 21일). O/B는 `plateau_full_length/nominal_depth_m/depth_regime`(전구간 통계, `features.py:203-209`, `data.py:255-267`)도 사용.
5. **threshold**: O는 fold별 서로 다른 threshold(0.15/0.2/0.15, close_gap 0/0/6, min run 12/12/6)를 fold 내부에서 골랐다(`p1_matched_budget…json` `historical_fold_schedule`).
6. **유형별 recall**은 2026-08-13 XGB OOF에서만 측정(spike 0.82, noise 0.94, flatline 1.00, offset 0.65, drift 0.65; FN의 92.7%가 offset/drift, 77%가 48h 이상 사건) — `reports/P1_FAILURE_RECON_2026-08-13.md`. anchor FN의 98.6%는 이미 부분 탐지된 장기사건 **내부**(`p1_anchor_false_negative_oracle_audit_v25`). 최종 챔피언(router+e150)의 유형별 recall은 어디에도 없다.
7. **test 구조 shift 미반영**(§6): test 세그먼트 중앙값 31행(train 70), 2048행 미만 세그먼트에 test 행의 88%(train 31%); 장기창 특징 결측 2–4배; G-ORS depth 전결측; I-ORS L3 부재; I-ORS L5 test 19,456행(train 10,190행).
8. **Public은 test의 약 28–31%만 본다(추정, §3-14)** → “S 238행 중립” 같은 결론은 Public 부분집합 진술일 뿐이다.

---

## 3) 취약점·버그·이상한 점·한계

| # | 심각도 | 내용 | 위치 |
|---|---|---|---|
| 1 | **높음** | router anchor는 공식 제출 CSV 그 자체(SHA 1b04e81c)를 예측 입력으로 쓰는데 `contract.json`은 `frozen_candidate_csv_used_as_prediction_input: false`. O/B 학습 코드·가중치가 `07_source`에 없음 → “모델 추론으로 재현” 주장은 MS-TCN 5%에만 해당. 재현성·규정 서술 리스크 | `predict_submission.py:69-76`, `artifacts/official_final_submission_20260905/P1/contract.json` |
| 2 | **높음** | O/B/router OOF의 fold 경계 특징 누출(purge 7일 < 337h) + 전구간 특징 사용 → 로컬 0.86은 낙관 | `config.py:85`, `splits.py:53`, `features.py:203-209` |
| 3 | **높음** | train/test 세그먼트 구조 shift: 장기창 특징 NaN(14d) train 10.4% → test 31.7%(S-ORS 42.7%, G-ORS 50%); 168h 5.8%→17.9%; peer 3.4%→10.6%. 모델은 “장기창 있음” 분포에서 학습, test는 “없음”이 다수 | `features.py:213-263, 291-335`(min_periods 25%), §6 집계 |
| 4 | **높음** | G-ORS 2026 depth 전결측: train G-ORS는 depth token shallow 90%/mid 10%, missing 0% → test는 전행 `missing`+`depth_raw_missing=1`(미학습 조합). O/B에선 `depth_regime="G-ORS\|unknown\|l1"` 미관측 범주. G-ORS 예측률 3.6%, 1월 1/1,449, 4월 3/3,841 | `ms_tcn_asrf_data.py:311-324`, `data.py:261-267` |
| 5 | 중간 | e150 = 300-epoch cosine의 중간(LR 1.6e-4) 체크포인트. seed 20260827 history: epoch 140 loss 0.0092 → 150 loss 0.0713(loss spike 위에서 저장). EMA/SWA 없음 | `run_p1…:1514-1534`, `03_model/training_provenance/*_history.json` |
| 6 | 중간 | 디코더 최소 19행 → 19행 미만 세그먼트(test 323개, 2,484행)와 spike(1행)·짧은 flatline(12행)은 MS-TCN이 구조적으로 못 더한다. spike/flatline/짧은 사건은 전적으로 2026-08-13 트리 모델 몫 | `run_p1…:1399`, `predict_submission.py:65` |
| 7 | 중간 | singleton spike 규칙: 챔피언 singleton 양성 83행, train 유병(101/776,706)로 기대되는 test spike는 ~22행 → ~60행 FP 의심. O의 spike 단독 precision 0.18 | `postprocess.py:169-207`, `selection.json spike_standalone` |
| 8 | 중간 | `keys["time"].astype(str) <= cutoff_kst` ISO 문자열 비교로 prefix 절단 — 모든 시각이 동일 `+09:00` 포맷일 때만 정확 | `run_p1…:992, 3465` |
| 9 | 중간 | pos_weight 20 clip이 실제 비 23.2에서 작동(경미). 캐시 `has_gap_before`는 (station, layer)로 연도 경계를 잇지만 창은 (station, year, layer)로 자름(문서화됨, S-ORS 2024→2025 경계) | `run_p1…:1453-1460`, `features.py:117`, config `cached_temporal_grouping` |
| 10 | 중간 | 비결정성: bf16+cuDNN, `torch.use_deterministic_algorithms` 미설정 → `train_model.py` 재학습 결과는 candidate SHA와 다르며 `predict_submission.py:100-102`가 SHA drift로 **반드시 실패**(인증 가중치로만 통과). 재학습 경로가 자기 검증을 통과할 수 없는 구조 | `train_model.py:57-58`, `predict_submission.py:100` |
| 11 | 낮음 | depth 분위(9.0/28.74 m)를 전 정점 pooled로 잡아 G-ORS(얕음)는 사실상 station 복제 토큰 | `ms_tcn_asrf_data.py:395-400` |
| 12 | 낮음 | `SegmentLayout.from_aligned`가 키별 O(n) 리스트 컴프리헨션(O(n·k)) — 성능만 | `ms_tcn_asrf_data.py:148-153` |
| 13 | 중간 | 경계 스냅 후 길이 재검사: 19행 이상 후보가 스냅으로 19 미만이 되어 버려지거나 그 반대 가능(스냅 실패 시만 원구간 fallback) | `run_p1…:1389-1400` |
| 14 | **높음** | **Public 점수는 test 부분집합**. 공식 receipt 산술: (+2 GI행 → +0.000300)은 D=2TP+FP+FN≈3,888에서 TP 1행; (−15 G행 → −0.004519)은 D≈3,887에서 15행 전부 TP; (−238 S행 → 0.000000, S5 추가 3행 → 0.000000)은 6자리에서 TP/FP 정확 상쇄가 불가능(7a≈5b, a=25,b=35도 +3e-6)하므로 **S 행들이 Public 밖**; (ALL6−SPIKE2 = −0.000215)은 4행 중 정확히 1 FP(I-ORS L4 05-27); (−80 I행 → −0.010753)은 ≈36–54 TP; veto(−325, 8 유지)의 −0.012909는 G15+I(80−8)로 정합; O∪B(+824 O-only, −0.011404)는 ≈78 TP/174 FP ⇒ 약 31%만 Public. 결론: **Public ≈ test 행의 28–31%, D_pub≈3,880(TP≈1,617, FP+FN≈646), 행 1개 = ±0.0002~0.0003 F1**. 공개 행은 월·정점 정렬이 아님(G-ORS 06-28/29 공개, S-ORS L6 06-27 62행 비공개, S-ORS L5/L6 6월 중순 spike 1행 공개) → 일/사건 블록 표집 추정 | `reports/p1_official_component_score_ledger_20260901_v1/score-ledger.json`, §4 표, 내 집계 |
| 15 | 중간 | router 제거 12행(S-ORS L1/L5/L6, I-ORS L4)은 Public에서 사실상 미검증(“GI 무제거” 0.817968 vs router 0.817873 → 제거 효과 ≈ −0.0001, 즉 중립/약간 해로움) | round F 결과 |
| 16 | 중간 | 계열별 예측률 편차(§6): S-ORS L8 1.08%(부족 ≈443행), I-ORS L7 2.5%(≈311), I-ORS L6 1.4%(≈156), G-ORS L1 3.6%; 월별 G-ORS 01/04, I-ORS 04, S-ORS 01은 ≈0%. train 23개 계열 모두 3.99–4.19%(예외 S-ORS 2025 L3 5.46%)라는 강한 균일성과 충돌 | 챔피언 CSV 집계 |
| 17 | 낮음 | `windowing.negative_window_ratio 2.0`은 사실상 무의미(4% 유병·14일 창이면 거의 모든 창이 양성 포함) | `ms_tcn_asrf_data.py:223-257` |

유형별 취급 요약: **flatline** — O/B recall ≈1.0(plateau 특징), MS-TCN 디코더는 type 가중에서 제외; **spike** — singleton 규칙(트리), MS-TCN 불가, FP 다수 의심; **noise/offset/drift** — MS-TCN 장기사건 add-on의 표적이나 OOF에서 offset/drift recall 0.65가 병목, FN의 99%는 ≥25행 사건 내부.

---

## 4) 공식 제출 이력 (P1, Public)

모두 배포 데이터 전용 계보(외부자료 사용 제출 0건; I-ORS 외부자료 실험은 OOF 전용, `artifacts/one_shot_exposure_ledger.jsonl` 1건·업로드 0). 점수/점수차는 `score-ledger.json`·round 결과 파일 기준.

| 날짜 | 후보 | 양성 | Public F1 | 설명 |
|---|---|---:|---:|---|
| 08-25 | original (O) | 6,504 | 0.790709 | offline XGBoost 0813 전체 refit, hys 0.2/0.1, min run 12 |
| 08-25 | P1_IMPROVED_ENSEMBLE_V1 (A) | 6,728 | 0.786145 | O + causal LightGBM 이벤트 구조(로컬 +0.0006, 공식 −0.0046) |
| 08-25 | P1_EVENT_DAY_BALANCED_LGBM_FALLBACK_V1 (B) | 5,856 | 0.793710 | 이벤트일 균형 LightGBM 3-seed |
| 08-26 | P1_1_EXPLOIT_DISAGREEMENT_ROUTER | 6,061 | **0.817873** | B + O-only(G-L1, I-L2) 복원 217 − B-only(S-L1/5/6, I-L4) 제거 12 |
| 08-26 | P1_2_PROBE_INTERSECTION (O∩B) | 5,680 | 0.802928 | B-only 176 제거 → +0.0092(B-only 다수가 Public FP) |
| 08-26 | P1_3_PROBE_UNION (O∪B) | 6,680 | 0.782306 | O-only 824 추가 → −0.0114 |
| 08-27 | P1_1_MSTCN_E150_ROUTER_UNION_ALL | 6,394 | **0.833248** | router ∪ e150 제안 333(G15/I80/S238) |
| 08-27 | P1_2_…_UNION_GS_ONLY | 6,314 | 0.822488 | I 80행 제외 → I 효과 +0.0108 |
| 08-27 | P1_3_EXPLOIT_GI_NO_REMOVALS | 6,073 | 0.817968 | B + G/I 217 복원, 제거 없음(제거 12행 효과 ≈ −0.0001) |
| 08-28 | **P1_1_E150_PLUS_GI_SPIKE2** | 6,396 | **0.833548** | e150 + B-spike 2행(S-ORS L5/L6) — 현 챔피언 |
| 08-28 | P1_2_E150_PLUS_GI_S5 | 6,399 | 0.833548 | +S-ORS 5행(추가 3행은 Public 밖) |
| 08-28 | P1_3_E150_PLUS_GI_ALL6 | 6,400 | 0.833333 | +6행(I-ORS L4 1행이 Public FP) |
| 08-29 | P1_MSTCN_LOWER_BOUND_VETO_PROBE | 6,071 | 0.820339 | e150 추가분 325 제거·8 유지(부트스트랩 하한 veto) → 해로움 |
| 08-30 | remove_g_ors_e150 | 6,381 | 0.829029 | G 15행 제거 → −0.004519(15행 전부 Public TP) |
| 08-30 | remove_s_ors_e150 | 6,158 | 0.833548 | S 238행 제거 → 0.000000(Public 밖) |
| 08-30 | keep_i_ors_e150_only | 6,143 | 0.829029 | G·S 제거 = G 제거와 동일 |
| 08-31 | P1_2_HIST_GBDT_OOF_STACK_UNION | 6,400 | 0.833548 | +4행 → 동률(4행 Public 밖 또는 상쇄) |
| 08-31 | P1_1_LABEL_FREE_RELIABILITY_GUARDED_LABEL_SHIFT_EM | — | 0.798819 | label-shift EM add-only(v30/v31), 로컬 +0.0018 → 공식 −0.035 |
| 09-01 | P1_REMOVE_I_ORS_E150_V33A | 6,316 | 0.822795 | I 80행 제거 → −0.010753 |
| 09-01 | P1_V32G_E150_CATBOOST_PRECISION_UNION | 6,419 | 0.832905 | +CatBoost p≥0.8 23행 → −0.000643(≈3 FP) |

Public이 반응한 것: (a) 정점·층 라우팅(G-L1, I-L2 복원) +0.024, (b) e150 장기사건 추가 중 G·I 부분 +0.015, (c) 개별 행 단위 ±0.0002–0.0003. **S-ORS 관련 변화는 어느 제출에서도 Public이 관측하지 못했다.**

---

## 5) 실패한 시도 요약 (재실행 금지 exact family 포함)

| 계열 | 무엇을 했나 | 왜 실패했나(수치) | 근거 |
|---|---|---|---|
| 초기 deep(TCN, Patch Transformer) | 0813 OOF 비교 | 0.768 / 0.800 < XGB 0.860(FP 5,789·3,578) | `P1_FAILURE_RECON_2026-08-13.md` |
| 규칙·구간 복원(block inpaint, target-masked quantile, semi-Markov/long-event residual v1–v8, endpoint-unanimity bridge, seeded boundary completion) | anchor FN 내부 채우기 | CI 0 통과·worst slice −0.059; quantile −0.63; outer 0 rescue; bridge −0.0015 | `negative_evidence_registry_20260830_v1`, `promotion_retroaudit_20260827_v1` P1-F02/03/09/11/14 |
| 전이·합성(IORS 외부 point residual, Round A causal rescue, synthetic injection) | 외부/합성 데이터로 recall 보강 | −0.063; 로컬 +0.002→Public −0.0046; pooled −0.0057 | P1-F04/05/08 |
| incumbent 후처리(O∪B, density correction, fixed-24h peer reliability, symmetric depth mask/G-ORS depth invariance) | 결합·보정 규칙 | Public −0.0114; fallback만 재현; worst group −0.048; G-ORS −0.0079 | P1-F10/13/15/16 |
| MS-TCN v2 사전등록(Q2 e125/0.9) | 882격자 선택→Q3/Q4 refit | Q4 −0.0315, pooled −0.0051, 추가 precision 0.38 | `p1_incumbent_preserving_mstcn_asrf_v2` |
| Sobol HPO 32+4 fits, Group-DRO, environment-balanced replay, window-phase consistency, frozen-trust adapter, partial pooling | MS-TCN 변형 | pooled +0.0006→sealed Q3/Q4 −0.0119; −0.0135; Q3 음수; +0.0002 | `p1_mstcn_sobol_hpo_20260829_v1`, `parallel_robust_repair_cycle_20260829_v2` 등 |
| 사건단위 router/veto(segment-precision router, bootstrap lower-bound veto, microfragment veto) | e150 추가분 선별 | Q4 −0.015; Public 0.820339(−0.013) | `p1_mstcn_segment_precision_router_retroaudit`, `p1_mstcn_lower_bound_veto_v2` |
| 표현학습(TS2Vec prototype, degradation-mask Transformer, AnomalyBERT-style, async latent GP, SupCon F1 head, CAPA, Deep SVDD) | 새 표현 | TP=0/F1=0; offset/drift boundary gate 실패; 대부분 pre-Q2 gate 실패 | `execution_followup_20260828_v7`, `approved_parallel_execution_20260828_v9`, v25 |
| label-shift EM(v25→v28→v28m1→v30/v31) | 무라벨 유병률 보정 add-only | v28 내부 +0.0087이나 day/slice gate 위반; v28m1 공식 materialization은 target prevalence 0.999999로 **162,615행 전부 양성** 붕괴; v31 Public 0.7988 | `p1_public_transport_repair_cycle_20260831_v28*/v30/v31` |
| tree 재도전(CatBoost v32a/e/f/g/h, HistGBDT/logistic/extra-trees OOF stack, peer full-train ladder) | 독립 tree add-only | v32a −0.124; v32g 공식 −0.0006; HistGBDT +4행 동률; peer ladder −0.0001~−0.024 | `p1_last_chance_32point_cycle_20260831_v1/candidate-matrix.md`, `submission_ladders_internal_validation_20260831_v1` |
| 정점·층 ablation(v33a/b/c, v34a) | 제거 방향 탐색 | I 제거 공식 −0.0108; nested layer 선택 Q3 역전; S layer abstain | 해당 report |
| 인과 특징 add-only v10–v51(recurrence, wavelet scattering, MMD, visibility topology, discord, Koopman, delay embedding, surprisal, diffusion, echo state, T-S-D 일관성, VIB, Teager, DFA, GCE, focal, R-Drop, 0-1 chaos, co-teaching, Fishr, MixStyle, MoE, Thorpe, cross-station causality 등) | 사전등록 pre-Q2 Wilson LCB gate | 전부 `NO_GO_EXPLORATORY_ONLY` / `SEMANTIC_DUPLICATE` / `CROSS_QUARTER_TRANSPORT_VETO`(대부분 fit 0 또는 no-op) | `reports/p1_v10…v51_20260901_v1` |
| 감사 v52–v58 | 승격 가능 후보 재검색 | 0건; TabPFN-2.6(합성 전용)만 남았으나 라이선스·패키지 부재 | `p1_v54…`, `leaderboard_clean_headroom_research_20260901_v1` |

공통 실패 원인(자기감사 인용 + 내 해석): 같은 Q2–Q4 OOF 재사용, 계절·구조가 다른 test로의 “transport” 미검증, Public을 진리로 오독(부분집합), add-only 미세 후보에 자원 집중.

---

## 6) 데이터 집계 관찰 (train.csv / test.csv, 집계만)

**train 776,706행, 양성 32,126(4.136%)**. 정점·연도·층 23계열 모두 양성률 3.99–4.19%(예외 S-ORS 2025 L3 5.46%) → **계열별 고정 예산으로 이상을 주입한 정황**. 행 수: S-ORS 2024 7층(31k–52k), S-ORS 2025 8층(24k–49k, L8 신규 48,953), I-ORS 2025 7층(L5만 10,190), G-ORS 2025 L1 26,503. 결측: psal 최대 I-ORS L7 16.8%·S-ORS 2025 L1 10.7%, depth ≤1.2%.
- 월별 양성률: 2024-02 0.6% ~ 2024-03 8.6%; 2025-01/02/03 = 5.4/7.5/5.4%; 2025-06 2.4%. H1 유병률 2024 3.89%, 2025 4.76% → **test 기대 양성 ≈ 6,590–8,110행 vs 예측 6,396행(recall 제약 정황)**.
- 사건(라벨 run) 263개: spike 101(전부 1행), flatline 52(중앙 95행≈16h, p90 43h, 최소 12행), noise 50(중앙 33h, p90 55h), offset 32(중앙 41h, p90 65h, 최대 730행), drift 28(중앙 52h, p90 82h). 행 점유율 noise 29% / drift 27% / offset 24% / flatline 19% / spike 0.3%. ≥19행 사건이 양성 행의 99.5%. 복합 토큰(`flatline+drift` 등) 12개 run.
- 세그먼트: train 1,232개(중앙 70행, 2048행 미만에 31% 행), gap 1,209개(중앙 20분). **test 929개(중앙 31행, 2048행 미만에 88% 행, 19행 미만 323개/2,484행), gap 914개(중앙 60분, p90 950분)**.
- 특징 결측률 train→test: 24h 0.5%→4.0%, 72h 2.4%→9.9%, 168h/7d 5.8%→17.9%, 14d 10.4%→31.7%(S-ORS 11%→43%, G-ORS 33%→50%), peer 3.4%→10.6%, depth 0.15%→9.7%(G-ORS 100%). `plateau_elapsed` p99 train 1 → test 28(test에 긴 plateau 비율 증가).
- test 169,011행: G-ORS L1 16,331(2월 0행, depth 전결측), I-ORS L1/2/4/5/6/7 = 16,967/6,342/6,333/19,456/5,888/19,967(L3 없음), S-ORS L1–L8 = 15,208/5,476/5,782/6,435/16,795/6,020/7,343/14,668. 기간 2026-01-01~06-30(I-ORS는 06-30 01:00까지).
- 챔피언 예측률: 계열별 1.08%(S-ORS L8)~7.88%(I-ORS L2); 월별 G-ORS 01/04 ≈0.1%, I-ORS 04 0.08%(3/3,865), S-ORS 01 0.02%(1/4,115).

---

## 7) 개선 기회 (기대 이득 × 노력 순, 규정 준수)

| 순위 | 제안 | 왜 이전에 제대로 안 됐나 | 기대 이득 / 노력 |
|---|---|---|---|
| 1 | **계절·구조 정합 검증면 신설**: holdout = 2025-01-01~06-30 전 정점(train = S-ORS 2024 + 2025-07~12; P1은 offline QC라 역순 사용 정당). 추가로 holdout 계열에 test의 gap 분포(중앙 60분, p90 950분, 세그먼트 중앙 31행)를 **표본 추출해 인위 단절**을 넣고 특징을 재계산 → “fragmented Jan–Jun” 검증면. 여기서 유형별 recall·계열별 예측률을 필수 진단으로 | 모든 fold가 4–12월이었고 구조 shift는 한 번도 재현 안 됨. 노출된 Q2–Q4는 더 이상 선택면이 될 수 없음(자기감사도 “새 확인면 없으면 성능 lane 중단”) | 직접 점수 이득 없음, 그러나 아래 2–5의 선택 신뢰도를 좌우. 노력 중 |
| 2 | **gap 증강 재학습**: train 계열을 test 분포로 무작위 단절한 뒤 캐시를 재생성해 O/B/MS-TCN을 학습(장기창 결측이 30–45%인 입력에 노출). 단순 대안: 결측 플래그 dropout | 장기창 특징 결측이 test에서 2–4배 높은 사실 자체가 발견되지 않았음 | +0.005~0.02(추정, S-ORS·G-ORS 예측률 회복). 노력 중~상(캐시 재생성 ≈ 수십 분, 학습 3×37분) |
| 3 | **계열별 유병 사전정보 활용(무라벨)**: train 23/23 계열 4.0–4.2% → test 계열별 기대 양성 ≈ 4.1%·행수. 예측률이 크게 낮은 계열(S-ORS L8 1.1%, I-ORS L7 2.5%, L6 1.4%)과 월(G 01/04, I 04, S 01 ≈0%)에서만 MS-TCN `p_dec` 또는 O/B 확률 상위 구간을 add-only로 채워 예산의 ~75%까지 접근. 추가 집합 precision > F1/2 ≈ 0.42면 F1 상승 | 이전 label-shift EM은 전역 유병률 1.0으로 발산(v28m1). 계열별 고정 상한·add-only·순위 기반이면 붕괴 불가 | +0.005~0.02(구멍이 실제 FN이면). 노력 하. 위험: H1 예산이 3.9–4.8%로 흔들림 → 상한을 3%로 보수적으로 |
| 4 | **e150 최적화 안정화**: cosine 지평 150으로 재학습(또는 epoch 120–150 가중치 평균/SWA), 손실 spike 위 체크포인트 회피. 선택은 1번 검증면 | v2 보고서가 “low-LR tail·checkpoint averaging” 권고했으나 실행된 적 없음(Sobol HPO는 다른 축) | +0.002~0.01. 노력 하(3×37분) |
| 5 | **G-ORS depth 결측 정합**: 학습 시 G-ORS 창의 depth 관련 채널(depth_raw·diff·token)을 100% 마스킹(test와 동일 조건). 이전 “depth invariance”는 tree 모델·대칭 마스크였고 MS-TCN·비대칭 마스크는 미시도 | P1-F16은 exact symmetric mask만 종료 | G-ORS(전체 9.7% 행) recall 회복 ≈ +0.002~0.006. 노력 하 |
| 6 | **singleton spike FP 감축**: 83 singleton 중 train 유병 기준 기대 ≈22 → O∧B 합의 또는 train 분위 기반 `spike_min_abs_diff`/3점 복귀도 조건 요구 | spike 규칙은 0813 이후 손대지 않음 | +0.002~0.004(full test). 노력 하. 1번 검증면에서 spike recall 회귀 확인 필수 |
| 7 | **router anchor 재현 가능화**: O/B를 패키지 안에서 재학습(21일 purge OOF로 threshold 재선택)하고 router 규칙을 코드로 명시 → 규정·재현 서술 정합. 제거 12행 규칙은 Public 미검증(≈−0.0001)이므로 제거 안 함으로 단순화 검토 | 재현성 주장이 5%(MS-TCN)에만 성립 | 점수 중립~소폭, 규정 리스크 제거. 노력 중 |
| 8 | **Public 해석 교정**: Public = 약 28–31% 행, S-ORS 변화는 비가시. 최종 선택에서 Public 동률·미세 차이를 근거로 쓰지 말고 1번 검증면 우선. 남은 제출권은 S-ORS를 포함하는 큰 구조 변화(2·3·5)에만 | 팀은 Public을 전 test 진리로 간주해 S 238행을 “중립”으로 기록 | 잘못된 최종 선택 방지 |
| 9 | 장기사건 내부 완성(두 탐지 run 사이 gap을 residual 통계 유사성으로 채움) | bridge/inpaint/boundary completion 3종이 exact 규칙으로 실패 | 낮음, 후순위 |
| 10 | TabPFN-2.6(합성 전용) station×layer 분류 | 라이선스·체크포인트 필요(정책 §2 4조건) | 미정, 사용자 행위 필요 |

의심해야 할 것: 3·6번은 Public에 맞추면 안 되고(행 단위 노이즈), 반드시 1번 검증면에서 계열별·유형별 precision > 0.42, 예측 유병률 상한, day-block CI로 확인한 뒤 1회 제출한다.
