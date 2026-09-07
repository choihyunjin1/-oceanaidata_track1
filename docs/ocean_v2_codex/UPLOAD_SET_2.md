# UPLOAD_SET_2 — 공식 채점 갱신 (2026-09-07 12:23 KST)

## 최우선 갱신 — 15:12 KST / 모든 답안 슬롯 소모·최종 확인 보류

잔여 **P1 0 / P2 0 / P3 0**. P1 B5 `cbeb7426` 28.858163점, O_slow `c38ace7a` 28.711047점으로 원형을 넘지 못했다. [공식 receipt](../../reports/p1_original_learning_seed_ablation_20260907_v2/official-receipts.json). 현재 우선 P1 `57844ef2` / P2 `794268f1` / P3 `70761aff`이며, **[최신 패키지·재현·권한 정본](FINAL_SELECTION_REVIEW_HANDOFF_20260907.md)**을 따른다. P2 최종 확인창 OK는 누르지 않았고 접수 미확인이다. 사용자 Fable 검토 후 재개 지시 전 최종 제출 금지. 아래 P1 잔여2/패키징 중은 과거 상태다.

## 최우선 최신 갱신 — 14:45 KST

현재 잔여 **P1 2 / P2 0 / P3 0**. 아래 이전 시각의 잔여 숫자는 이력이다.

| 추가 채점 후보 | SHA 앞8자 | Public RMSE | Public 점수 | 증거 |
|---|---|---:|---:|---|
| P2 L120 s3 smooth7 projection | 794268f1 | 0.395254℃ | 28.373869 | [receipt](../../reports/p2_l120_s3_smooth7_projection_20260907_v1/official-receipt.json) |
| P3 CPU no-shrink | 70761aff | 0.595521m | 23.881592 | [receipt](../../reports/p3_numeric_cpudet_noshrink_20260907_v1/official-receipt.json) |
| P3 CPU equal-component / original shrink | c9fa5366 | 0.600933m | 23.795692 | [receipt](../../reports/p3_cpudet_mean_router_ablation_20260907_v1/official-receipt.json) |

P2新 후보는 별도 ZIP 추출의 빈 모델 학습→추론→후처리와 저장 모델 재생까지 exact SHA PASS. [답안 경로·bytes·SHA·노트북 실측](../../reports/p2_l120_s3_smooth7_projection_20260907_v1/candidate-ready.json). P3는 저장 모델 재생 PASS, fresh_cold_2 진행 중. Private 성능·공식 적격성·최종 모델 접수는 아직 증명하지 않는다. P1 두 새 후보는 별도 학습과 내부 QA를 마쳐 패키징 중.

## 최신 상태 — 답안 4건 제출·채점 완료

사용자의 최신 직접 지시 “그럼 제출하고 진행하세요”에 따라 리더보드 답안을 업로드했다. 아래 기존 10:32~11:56 기록의 '업로드하지 않음/미완료/점수 미확인'은 당시 이력이며, 현재 상태는 이 절과 [공식 채점 receipt](../../reports/final_day_candidate_official_submissions_20260907_v1/receipt.json)를 따른다. 모델 최종 지정·삭제·commit/push는 하지 않았다.

| 후보 | 답안 SHA 앞 8자 | 공식 지표 | 공식 점수 | 현재 판단 |
|---|---|---:|---:|---|
| P1 champion bracketB | 7538082a | F1 0.827921 | 28.7598 | 원형 57844ef2 / 28.909341점 보존 |
| P2 L120 s3 projection | 9c5fec38 | RMSE 0.405920℃ | 28.240037 | 기존 fee6118b 대비 +0.162757점; 우선 후보 |
| P2 L120 s10 projection | 672c1df1 | RMSE 0.418378℃ | 28.083729 | s3 projection보다 낮음 |
| P3 numeric CPU s3 completion | 2015b387 | RMSE 0.609836m | 23.654388 | ff42a6a0 / 23.741446점보다 낮음; cold 재현 대기 |

공식 제출관리에서 네 건 모두 채점완료(12:17~12:19 KST)를 확인했다. 업로드 후 남은 횟수 P1 **2/3**, P2 **1/3**, P3 **2/3**. 정확한 경로·행 수·bytes·SHA는 위 receipt에 있다. P1/P2 새 후보의 로컬 QA/패키지 재생은 완료했지만 P1 새 결합 전체 cold는 미실행이며, P3 fresh cold와 최종 패키지는 진행 중이다. P3 업로드를 재현 적격성 확정으로 표현하지 않는다.

다음 작업은 실행 중 P3 fresh_cold_2 보존과 사전 계획한 별도 no-shrink w=0 후보 준비(새 fit 0)다. 공개 점수로 계수나 특징을 조정하지 않는다.

## 결론

### 12:30 KST 추가 — P3 no-shrink 후보 준비 완료, 공식 미채점

- 정본: [결과·파일·독립 QA](../../reports/p3_numeric_cpudet_noshrink_20260907_v1/result.json), [보고서](../../reports/p3_numeric_cpudet_noshrink_20260907_v1/report-source.md).
- `C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_numeric_cpudet_noshrink_v2/ANSWER/submission_p3_numeric_cpudet_noshrink.csv`, 1,200행 / 39,869bytes / SHA256 `70761affca4d3fc6f1d24ae53467e5b185b23465926ebb4b851f0300872cddbd`.
- 새 fit 0. 고정 CPU 모델에서 긴 lead의 기존 0.2 shrink만 제거. 내부 RMSE 0.681321563→0.677450331m, P(개선)0.914, CI90[-0.008524918,+0.000733607]m. 저파고 부분집합 +0.009921904m 악화는 별도 위험이며 자동 탈락 조건은 아니다.
- 별도 PID 및 실제 ZIP 추출 노트북 SHA 일치 PASS. 이는 저장모델 replay이며 전체 cold 완료 주장이 아니다. fresh_cold_2는 계속 설정 불변. 아직 업로드하지 않았으며 위 남은 횟수는 그대로다.

### 아래는 11:56 KST 당시 기록

**P2_L120_s3_proj는 로컬 답안·QA·패키지·추출 노트북 재생을 완료했다. P3 무제한 completion_1은 11:56 KST 새 답안과 별도 PID exact replay를 완료했고 fresh_cold_2가 시작됐다. P3 최종 패키지·독립 cold SHA 일치는 아직 미완료다. P1 범위 룰은 철회하고 원형을 보존하며 별도 bracket B 교체를 학습했다.**

P3 최신 실행은 [무제한 실행 기록](../../reports/p3_numeric_cpudet_unbounded_20260907_v2/launch.md)을 따른다. 완료 모델 29개를 검증한 이어학습 후 빈 모델 폴더에서 독립 재학습 1회를 수행한다. 두 번 fresh cold가 아니다. 이 새 실행에는 4h/6h/15:00 강제 종료를 적용하지 않으며 일반 P3의 공식 6시간 적용 범위는 미확인이다. 아래 기존 실패 기록은 이력으로 보존한다.

업로드·삭제·최종 지정·commit/push는 이번 작업에서 수행하지 않는다. 사용자 확인은 문제별 잔여 3회이며, 공식 마감 시각·첨부 제한은 최신 화면 확인을 하지 못했다. 15:00 KST는 내부 후보 확정 목표이지 확인된 공식 마감이 아니다.

## 1. 후보별 확정 상태

### P3 completion_1 새 답안 (11:56 KST 생성, 독립 QA 확인)

- 파일: `C:\Users\cedis\Documents\OceanFinalDay_20260907\P3_numeric_cpudet_unbounded_v2\completion_1\05_answer\submission.csv`
- 1,200행 / 39,854bytes / SHA256 `2015b38750d357630d5b2e9eee32807d2961ce752e35eb2b454ee16e579dda56`.
- schema/key/order/finite/중복0 PASS. 학습41712·추론39576·replay31224는 별도 PID이며 답안 SHA 정확 일치. Root가 공식 index 키·순서와 답안 해시를 다시 대조했다.
- 103,602행 내부 OOF RMSE **0.681321563m**를 직접 재계산해 receipt와 일치했다. 새 공식 점수나 예상 점수가 아니다.
- 이 실행은 재사용29backbone + 새7backbone·5router이며 현재 이어학습~답안 wall **5,164.813초**. 빈 폴더부터의 전체 학습 시간으로 쓰지 않는다.
- `fresh_cold_2`는 11:56:39 KST 자동 시작. 독립 cold SHA 일치, 최종 SOURCE/SAVED 포장 및 ZIP 재생은 대기 중. 아직 최종 지정·업로드하지 않았다.
- 정본: [completion 답안 독립 QA](../../reports/p3_numeric_cpudet_unbounded_20260907_v2/completion-answer-independent-qa.json). 기존 ff42 fallback은 보존한다.

### 11:32 KST 신규 P1/P2 실제 학습 시작 (완료 아님)

사용자 "그럼 실행하세요"에 따라 남은 후보 준비를 P3와 병행한다. [작업 분리·독립 QA 계약](../../reports/final_day_remaining_candidates_20260907_v1/plan.md).

- P1 원형 bracket B 교체: 기존 bracket 단독 모델은 1seed/다른 선택 정책이라 원형 B3seed의 대체 가중치로 그대로 재사용하지 않는다. Q3/Q4의 O2fit + B80 control3seed↔B1073seed paired12fit + fullB1073fit =17fit를 사전등록하고 11:31:01 KST 시작했다(launcher8380, worker41780). 옛 O의7일purge와 MS의21일purge가 달라 O는 고정설정으로21일계약에재학습해두팔에공유한다. 기존 MS/셀정책/GI는 동결하고 같은 지원행에서 조합 효과를 검증한다. 현재 첫 O fit 진행이며 완료·공식 점수는 미확인이다. 정본: `artifacts/p1_champion_bracketb_20260907_v1/progress.json`.
- P2 L12010seed+투영: 새seed7개×8fold+full=63fit를 11:28:10 KST 시작했다(launcher2408, worker11152). 11:32 확인 시 7/63fit 완료, 다음 B1 학습 중이며 stderr0이다. 기존3seed와같은규칙의모델·OOF 출처를검증하여활용하고B3primary/all8/outage를평가한다. 이후 portable fresh cold10fit는 별도 재현 단계이며, 시행 시 새 학습 총73fit와 재사용27fit를 구분한다. GPU0배정. 첫공식채점결과대기는후속후보준비에한해앞당겼다. 업로드나최종선택승인은아니다. 정본: `artifacts/p2_l120_s10_proj_20260907_v1/progress.json`.
- 기존P3의CPU4/frozen설정은재현비교를위해유지한다. 이번신규후보에불필요한wallcap을두지않는다. CPU/GPU 소유 조정은경합방지이며성능기준변경이아니다.

아래 표는 기존 완료후보/실패 이력이다. 신규 P1/P2 모델·답안·공식점수가완성됐다는뜻이아니다.

| 문제 / 후보 | 확정 사실 | 가설 | 미검증 / 남은 일 |
|---|---|---|---|
| P2_L120_s3_proj | 26,061행, schema/key/order/finite/중복 PASS; 별도 PID SHA 일치; 새 SAVED ZIP 추출 노트북 PASS | 공개 endpoint 투영의 내부 개선이 공식 평가에도 일반화 | 공식 점수 미확인; 새 결합 파이프라인 full cold·새 OS 완전 오프라인 검증은 미실행 |
| P3_numeric_cpudet_s3 | 첫 cold 14,400초 제한, exit124; backbone29/36 완료 후 종료; cold2 미실행 | CPU 3-seed의 재현성·점수 개선 가설은 검증되지 않음 | 신규 CSV 없음; 두 cold SHA·새 점수·패키지 QA 불가. 재시작 없이 보존 |
| P1 원형 57844ef2 | 기존 답안 보존; 범위 룰 미적용 | bracket B 교체는 시간 여유가 있을 때만 별도 실험 | 이번 작업에서 bracket B 신규 fit/후보 없음 |

## 2. P2 업로드용 파일과 패키지

선택 후보 CSV 절대경로:

`C:\Users\cedis\Documents\OceanFinalDay_20260907\P2_L120_s3_proj_v2\ANSWER\submission_p2_L120_s3_proj.csv`

- 행 수: **26,061**, bytes: **1,190,759**.
- SHA256: `9c5fec385930ab5a997a70c10550f3ada18872a452938bfc3fa78232e17a5118`.
- 제안 제목: `P2 L120 3-seed endpoint projection`.
- 한 줄 요약: `배포 데이터 학습 L120 3-seed에 공개층 endpoint clip→방향 PAVA를 적용한 결정적 후처리 후보; 새 공식 점수 미확인.`
- 모델 패키지와 CSV는 다른 역할이다. 리더보드 답안 선택 시 위 `_proj.csv`만 선택하고, 중간 `submission_p2_L120_3seed.csv`와 혼동하지 않는다. 현재 UI FORM 필드·모델 첨부 제한은 별도 확인 필요.

같은 후보 루트의 압축파일:

| 패키지 | bytes | SHA256 |
|---|---:|---|
| P2_L120_s3_proj_SOURCE_ONLY.zip | 72,130 | cb2fbf6c87dd8c50f15573da8937d14eca9d6aff3c58e695aff5fca5afa11b4c |
| P2_L120_s3_proj_SAVED_MODELS.zip | 137,582 | e806d1f4048151059daaa9486f45c538619d40f30c1d7574494b9b18b07c69eb |

실행: `P2_DATA_DIR`에 배포 P2 폴더 지정. 기존 pinned 환경과 CUDA 필요. SOURCE_ONLY의 빈 모델 폴더에서 `TRAIN.ipynb → PREDICT.ipynb → PROJECT.ipynb`; SAVED_MODELS의 새 추출 폴더에서 `SAVED_PREDICT.ipynb`만 실행한다. 원본 README는 계보 보존용이며 **CURRENT_README.md**가 이번 후보의 선택·실행 안내다. 새 후처리는 `02_code/projection/`에 격리했다. 패키지에 원본 데이터·CSV 답안·attempt lock은 동봉하지 않았다.

### 재생성 증거 범위

1. 기존 L120 cold 3-fit, **173.010초** → `fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d` 정확 재생 증거를 계승했다.
2. 새 후처리 생성 PID 1124 / 독립 재생 PID 33956, **4.860 / 5.078초**, 위 후보 SHA 일치.
3. 새 SAVED ZIP을 `P2_L120_s3_proj_saved_verify_v2`에 실제 추출해 노트북 실행: 기존 가중치로 base 추론·재생 → 후처리·재생, **43.703초**, 후보 SHA 정확 일치. 여기서도 새 학습은 0이다.
4. 독립 scalar PAVA 대비 최대 차이 `3.552713678800501e-15`; schema·key·순서·중복·finite·불완전 프로필 no-op PASS.

**새 결합본의 빈 모델 full cold를 두 번 실행했다는 의미가 아니다.** 이번 사용자가 승인한 증거 정의(기존 cold + 후처리 독립 재생)에 따른 완료다. 첫 포장 v1은 추가 Python 파일이 기존 코어의 학습 시 fingerprint를 바꿔 추출 실행이 실패했으며 보존했다. v2는 원본 코드/검사/모델을 변경하지 않고 후처리 파일만 하위 폴더에 분리해 통과했다. [포장 정정 기록](../../reports/p2_l120_s3_proj_20260907_v1/packaging-amendment.md).

### 승인된 OOF 재계산

166,268행 `natural_L120`, OOF SHA `4ce3a80171e40cdb3ee760eec43e86b48441de05db9b2c587ef3e12ac2f695a5`. 완전한 2/3/4층, 유효 T1 및 T5→T6→T7→T8 최초 유효 endpoint만 사용한다. clip 후 endpoint 방향 exact PAVA; 결측 endpoint·불완전 프로필은 원값 유지.

| 구간 | 기준 RMSE ℃ | 후보 RMSE ℃ | Δ 후보−기준 ℃ |
|---|---:|---:|---:|
| pooled | 1.251771599 | 1.236732066 | −0.015039533 |
| B1 | 1.985490075 | 1.985610960 | +0.000120885 |
| B2 | 1.325582977 | 1.317290363 | −0.008292615 |
| B3 | 0.483505057 | 0.463529468 | −0.019975588 |
| B4 | 0.042661178 | 0.038657304 | −0.004003875 |
| B5 | 0.713066243 | 0.701440377 | −0.011625866 |
| B6 | 1.920018946 | 1.873266255 | −0.046752691 |
| B7 | 0.951496967 | 0.945775449 | −0.005721519 |
| B8 | 0.217714846 | 0.206509503 | −0.011205344 |

7개 개선 / B1 악화. 악화를 자동 탈락 사유로 쓰지 않는다. **이미 노출된 retrospective natural OOF**이지 새 검증이나 outage 전체 개선 증명은 아니다. 공식 RMSE/점수로 환산하거나 base의 공식 점수를 승계하지 않는다. 공식 26,061행 중 25,890행이 투영 대상, 10,781행이 수치 변경됐다. 이 행 수는 개선 정답 수가 아니다.

정본 집계: [P2 결과](../../reports/p2_l120_s3_proj_20260907_v1/result.json). 상세 로컬 OOF·답안·재생 receipt는 후보 루트의 `OOF_QA.json`, `ANSWER_QA.json`, `REPLAY_QA.json`; 추출 실행 증거는 형제 파일 `P2_L120_s3_proj_saved_verify_v2-receipt.json`.

Fable 독립 재계산([검토 2 §3](FABLE_INDEPENDENT_REVIEW_2_20260907.md))은 Codex 규칙 적용 후 166,268행 행 단위 일치(최대 절대차 0.0, pooled 1.236732066)를 보고했다. 이전 차이는 T1 결측 완전 프로필 34개(102행)의 fmin/fmax 처리 차이였고 배포 규칙은 해당 행 무변경이다. 이 문서·QA 정정에서는 재계산을 다시 실행하거나 result.json 수치를 변경하지 않았다. OOF 개선은 test 개선의 상한·하한이 아니다.

시간은 우리 PC 실측 L120 cold 173.010초, 별도 후처리 4.860초, 새 saved ZIP replay 43.703초다. Windows/Python 3.12.10/Ryzen 7800X3D/RTX 5090, L120 CPU2+CUDA0; 후처리는 별도 thread cap 코드가 없다. 일반 모델의 공식 6시간 적용 범위는 미확인(공지의 사전학습 예외 조건 3만 확인, 문제지 Ⅳ-2 미열람)이며 시간 수치로 적격성을 단정하지 않는다.

## 3. P3 진행·완료 조건

**09:20 업데이트: 아래 실행 시작 스냅샷 이후 09:14:27 KST에 첫 cold가 시간 제한으로 종료됐다. 완료 backbone29(단일15·multi14), full-data fit0·router0·완료cold0이다. 모든 source pin25개 및 완성 모델29개 SHA를 대조했다. 원인은 코드 호환성이나 점수가 아니라 CPU 전체 실행 비용 대비 4시간 예산 부족이다. heartbeat는 PAUSED, 자동 재시작/두 번째 cold/후속 포장은 하지 않았다. [실패 원인과 증거](../../reports/p3_numeric_cpudet_s3_20260907_v1/failure-analysis.md).**

실행 루트: `C:\Users\cedis\Documents\OceanFinalDay_20260907\P3_numeric_cpudet_s3_v1`.

사전등록: [P3 config](../../configs/experiments/p3_numeric_cpudet_s3_20260907_v1.json). 591 features, single 700 / multi 1200, seeds 20260817/18/19, seed별 clip 후 평균, 기존 TRAIN-OOF router·long-lead shrink 0.2 유지. **36 backbone + 5 router = 41 production fits/cold**, 2회 총82. Synthetic compatibility fits는 별도이며 production count에 포함하지 않는다.

시작 당시 기록: 05:14:20 KST, launcher PID30912 / supervisor25544, TRAIN PID41284. 각 cold 사전등록 wall cap은 4시간이었다. 첫 cold에서 이 제한을 초과해 전체 계획이 종료됐고 두 번째 cold의 시작 조건까지 도달하지 못했다. 소비된 attempt 재시작·설정 변경 금지.

완료 시 `terminal_result.json`의 두 cold SHA 일치 확인 → `scripts/package_p3_cpudet_completed_20260907_v1.py`로 독립 QA/포장 → 실제 추출 `SAVED_PREDICT.ipynb` 재생. 포장 스크립트는 준비됐지만 cold가 끝나기 전에는 실행하지 않는다. 불일치/기술 실패면 가중치·로그 보존 후 원인 보고; 아직 공식 답안 파일이 있다고 가정하지 않는다.

**shrink 제거는 라운드1 사용자 결과 기록 후 슬롯2에서만 진행**한다. 현재 slot1 모델에 섞지 않는다. Fable 문서의 남은 구 수치/반대 방향 문장은 기존 검증 receipt와 최신 사용자 지시보다 우선하지 않는다.

## 4. P1 및 운영 경계

P1 원형 fallback: `C:\Users\cedis\Documents\OceanFinalRelease_20260907\P1\ANSWER\P1_submission.csv`, SHA `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687`. 이번에는 범위 룰을 추가하지 않았다. bracket B는 필수 패키지 작업 완료 후 시간이 남을 경우에만 별도 실험한다.

모든 신규 후보는 사용자 업로드 전 상태다. 공개 점수로 계수/threshold를 조정하지 않는다. 공식 숨은 정답 접근 0. 기존 dirty worktree와 실패 기록 보존. 현재까지 최종 지정·삭제·업로드·커밋·푸시 0.
