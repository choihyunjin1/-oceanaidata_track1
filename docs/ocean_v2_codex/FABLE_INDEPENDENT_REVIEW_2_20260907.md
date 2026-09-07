# Fable 독립 검토 2 — 2026-09-07 10:45 KST (읽기 전용)

범위: (1) 공식 원문에서 일반 모델(CatBoost 포함)에 대한 6시간 제한 적용 여부, 오늘 마감 시각, 첨부 제한. (2) P1 셀별 조합 정책의 학습·선택 코드와 산출물 연결. (3) P2 투영 재계산 값이 Codex와 달랐던 원인.
경계: 새 학습·재튜닝 0. 실행한 계산은 P2 투영 분해 재계산 1회뿐이다(기존 OOF npz + observations.csv 읽기, 단일 스레드, 3.1초). 기존 파일 변경 0, 업로드·삭제·최종 지정·commit·push 0. hidden 정답 접근 0. 10:30 KST 시작한 P3 무제한 실행(`P3_numeric_cpudet_unbounded_v2`, 10:41 기준 `completion_1` CPU_TRAINING, backbone 29/36 계승)과 중복 작업 없음.
표기 원칙: 원문에서 확인한 것만 "확인", 그 외는 **미확인**으로 적는다. 방향이 같다는 이유로 수치 일치를 선언하지 않는다. OOF 개선을 test 개선의 상한·하한으로 표현하지 않는다.

## 0. 결론 요약

1. **6시간 제한**: 공식 원문(공지 12건 전문, 배포 README 3종, 문제 상세 화면, 1:1 문의 답변)에서 "6시간"은 09-01 공지의 사전학습 모델 사용 조건 3)에 **한 번만** 등장한다. 일반 CatBoost 모델에 6시간이 적용된다는 조항은 발견하지 못했고, 적용되지 않는다는 조항도 없다. 재현 검증 절차 원문(문제지 Ⅳ-2)은 로컬·포털 어디서도 열람하지 못했다 → **미확인**. 운영상으로는 "6시간이 적용된다"고 가정해 준비하는 것이 안전하지만, 준수 주장은 우리 PC 실측으로 한정해야 한다.
2. **마감 시각**: 공식 원문은 "2026년 9월 7일"까지만 있고 시각이 없다 → **미확인**. **첨부 제한**: 오늘 화면에서 모델 제출 모달을 열지 않았으므로 **미확인**(Codex 문서도 `portal_attachment_limits_rechecked: false`).
3. **P1 셀 정책**: O·B 학습, 불리언 조합, 후처리 임계, GI spike 규칙, anchor replay는 코드·산출물이 연결되어 **재생성 가능**하다. 6개 셀(add 2, remove 4)을 **고른 절차**는 코드가 없고 08-26 라운드 C 보고서의 OOF 수치 설명만 있다. 셀 목록은 배포 코드·포장 스크립트·anchor manifest 3곳에 **리터럴**로 고정되어 있다.
4. **P2 차이 원인 확정**: 내 재계산은 T1이 결측인 완전 프로필 34개(102행)에서 NaN 무시 `fmin/fmax`로 envelope를 `[deep, deep]`으로 붕괴시켰다. 이 102행만 Codex 규칙(무변경)으로 되돌리면 나머지 166,166행의 차이는 정확히 0.0이고 pooled 1.236732066으로 Codex와 **행 단위 동일**하다. 동률(T1 == deep) 처리 차이는 수치 영향 0.

## 1. 6시간 제한·마감 시각·첨부 제한 — 원문 인용

### 1-1. 오늘 열람한 공식 원문

로그인 상태에서 `/api/notices` 응답(공지 12건 전문)과 화면을 10:20~10:45 KST에 읽었다.

| 출처 | 일시 | 6시간 | 마감 시각 | 첨부 제한 |
|---|---|---|---|---|
| 공지 id 15 중·고등부 멘토링 | 09-04 17:34 | 없음 | 없음 | 없음 |
| 공지 id 14 리더보드 반환 점수 | 09-02 10:41 | 없음 | 없음 | 없음 |
| 공지 id 13 사전학습 가중치 규정 보완 | 09-01 12:26 | **조건 3)** | 없음 | 없음 |
| 공지 id 12 외부 데이터 | 08-31 13:58 | 없음(문제지 Ⅳ-2 언급) | 없음 | 없음 |
| 공지 id 11 배점 정정 | 08-26 20:52 | 없음 | 없음 | 없음 |
| 공지 id 9 제출·채점 방식(수정) | 08-12 16:25 (08-25 수정) | 없음 | 날짜만 | 없음 |
| 공지 id 8 일정 조정(3일 연기) | 08-07 17:44 | 없음 | 날짜만 | 없음 |
| 공지 id 5 FAQ | 08-04 (08-28 수정) | 없음 | 없음 | 없음 |
| 공지 id 4 개최·참가 안내 | 07-27 | 본문은 이미지 1장 + 첨부 PDF | (첨부 미열람) | (첨부 미열람) |
| 문제 상세 `/app/problems/5` (OCN-01) | 오늘 | 없음 | 없음("오늘 남은 제출 3 / 3"만) | 없음(모달 미개봉) |
| 홈페이지 일정 섹션 | 오늘 | 없음 | "2026.09.30 예선 종료 · 결과물 제출 마감"(전 부문 공통 표시) | 없음 |
| 1:1 문의 내역(답변 1건, 08-25) | 오늘 | 없음 | 없음 | 문의 첨부 "최대 5개, 개당 10MB"(모델 제출 제한 아님) |
| 배포 README P1/P2/P3 | 로컬 | 없음(grep "6시간" 0건) | 없음 | 없음 |

### 1-2. "6시간"이 나오는 공식 원문 (전부)

**공지 id 13 (09-01 12:26, 참가자 전용) 원문:**

> 규정상 "사전학습 가중치 사용 금지"는 사전학습 과정에서 실제 관측자료를 학습한 모델을 제한하기 위한 조항입니다.
> 따라서 구조적 인과 모델 기반의 합성 데이터로만 사전학습된 테이블형 파운데이션 모델(TabPFN 계열 등)은 사용하실 수 있습니다. 다만 아래 4가지 조건은 지켜주셔야 합니다.
> 1) 합성 데이터로만 사전학습된 모델일 것
> 2) 가중치를 제출물에 동봉하고 로컬에서 로드할 것
> 3) 6시간 제한을 지킬 것
> 4) README.md에 모델명과 사전학습 데이터 성격을 명시할 것

이것이 공식 원문에서 "6시간"이 등장하는 유일한 곳이다.

**재현 검증을 언급하지만 시간 제한이 없는 원문:**

- 공지 id 12 (08-31): "참고로 마감 후 상위권 팀에 대해 인터넷이 차단된 환경에서 재현 검증을 수행합니다(문제지 Ⅳ-2). 배포 데이터 외의 자료를 사용한 제출은 이 절차에서 확인되며, 검증을 통과한 팀만 본선에 진출합니다."
- 공지 id 14 (09-02): "이러한 방식으로 정해진 값은 코드에 상수로 남게 되어, 재현 검증 1항(과도한 상수 리터럴)과 4항(학습 산출물 제거 후 예측 재생성)에서 확인됩니다. 재현 검증을 통과한 팀만 본선에 진출합니다." → 번호가 붙은 재현 검증 항목(최소 4개)이 문제지 Ⅳ-2에 있음을 시사한다. 2항·3항의 내용은 미확인이며, 그중 하나가 시간 제한일 가능성은 있으나 원문을 보지 못했다.
- 공지 id 9 (08-12): "운영진은 제출된 모델이 업로드한 예측 답안을 실제로 재현하는지, 정상적으로 학습되었는지 검증합니다."
- 공지 id 5 FAQ (08-28 수정): "재현 검증은 인터넷이 차단된 환경에서 실행하므로 외부 네트워크 접속 코드가 있으면 실격 처리됩니다."

**저장소 내부 문서의 "6시간"은 공식 인용이 아니다.** `00_ORGANIZER_DATA_POLICY.md:30`("3. 전체 재현 실행을 6시간 이내에 완료한다.")은 위 공지 조건 3)을 옮긴 것이고, `:40`("6시간 제한 및 README 공개 항목을 독립 QA")과 `README.md:86`("배포 자료만으로 네트워크 없이 6시간 안에 재현해야 합니다")은 우리가 세운 내부 규범이다. `PACKAGING_SPEC.md:4`, `FINAL_RELEASE_20260907.md:70`, `CONTRACT.md:50` 등도 같은 내부 규범을 인용한다. Codex의 `UPLOAD_SET_2.md:7`("일반 P3의 공식 6시간 적용 범위는 미확인이다")과 `reports/p3_numeric_cpudet_unbounded_20260907_v2/launch.md`("General P3 applicability of an official six-hour reproduction rule has not been established from the primary notice")는 이 검토와 같은 판단이다.

### 1-3. 판정: 일반 CatBoost 모델에 6시간이 적용되는가

**미확인.** 원문 해석은 두 가지가 모두 가능하다.

- (a) 조건 3)의 "6시간 제한을 **지킬** 것"은 이미 존재하는 제한을 전제한 표현이므로, 문제지 Ⅳ-2에 모든 제출에 적용되는 재현 시간 제한이 있고 사전학습 모델도 예외가 아니라는 뜻. 문장 구조상 이쪽이 더 자연스럽다.
- (b) 사전학습 가중치를 쓰는 경우에만 부과되는 조건.

문제지 Ⅳ-2를 읽기 전에는 확정할 수 없다. 열람 시도 결과: 저장소·Downloads 데이터 폴더에 문제지 원문 없음(grep "문제지", "Ⅳ", "6시간": 배포 README에는 0건, P3 README는 "문제지에 고정해 공개한 정책 상수"라고 문제지의 존재만 언급). 포털 문제 상세는 본문 1-1~1-6만 표시하고 Ⅳ장은 없다. 07-27 공지 첨부 `공고문_2026_해양과학AIx빅데이터_경진대회.pdf`(163.3 KB)는 내려받지 않았다(파일 다운로드는 사용자 승인 사항). Downloads의 `1234.pdf`, `4321.pdf`는 카카오톡 캡처 이미지라 텍스트 추출이 되지 않는다.

**권고(행동 아님, 판단 자료):** 6시간이 적용된다고 가정해 준비하되, 문서에는 "우리 PC(7800X3D, 4 threads / RTX 5090) 실측"으로만 적고 "6시간 준수"를 일반 사실로 쓰지 않는다. 사용자가 공고문 PDF에서 "Ⅳ-2", "6시간", "재현"을 검색하면 바로 확정된다. 1:1 문의는 평일 10:00~18:00 운영이며 오늘 중 답변은 보장되지 않는다.

**P3와의 관계(실측 수치, 판정 아님):** 첫 cold는 backbone 29/36을 13,967.968초에 마쳤고(`reports/p3_numeric_cpudet_s3_20260907_v1/failure-analysis.md`), 같은 fold의 multi fit이 1,068.969초·1,075.297초였다. 단순 비례로 36 backbone은 약 17,300초이고, 여기에 router 5개·QA·추론이 더해진다. 즉 이 후보는 우리 PC에서도 5시간 안팎이며 검증 PC가 느리면 6시간을 넘을 수 있다. 6시간 적용 여부와 무관하게 "P3 cpudet 후보는 6시간 준수를 아직 증명하지 못했다"로 적어야 한다.

### 1-4. 오늘 마감 시각

- 공지 id 8 (08-07): "문제 제출 마감: (기존) 9월 4일 → (변경) 9월 7일"
- 공지 id 9 (08-12): "참가 전원, 예선 마감(2026년 9월 7일)까지 실제 모델(코드 및 학습 가중치)을 제출해야 합니다."
- 홈페이지 일정 섹션: "2026.09.30 예선 종료 · 결과물 제출 마감 — 온라인 해커톤 종료, 부문별 결과물 마감". 전 부문 공통 표시이며 대학부 3일 순연 공지와 맞지 않는다. 대학부에는 적용되지 않는다고 보는 것이 합리적이지만 이는 해석이다.
- 대시보드·문제 상세·제출관리·1:1 문의 답변: 시각 표시 없음.

**판정: 미확인.** 시각이 적힌 공식 원문이 없다. 답안 업로드 "하루 3회"의 리셋 기준(KST 자정 여부)도 원문에 없다. 보수적으로는 09-07 중 이른 시각(예: 운영시간 18:00)을 내부 마감으로 두는 편이 안전하나, 이것도 가정이다. 문서의 "15:00 KST"는 내부 목표이지 공식 마감이 아니다(`UPLOAD_SET_2.md` 서두와 동일 판단).

### 1-5. 첨부 제한

- 오늘 문제 상세 화면에는 첨부 제한 문구가 없다. "모델 최종 제출하기" 모달은 열지 않았다(제출 흐름 진입은 사용자만 한다).
- `docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md:37`은 "현재 UI의 파일 크기/첨부 제한을 보고 필요한 분할과 manifest를 검증합니다"라고만 적고 숫자가 없다. `docs/FINAL_RELEASE_20260907.md:37`은 "45MB는 로컬 보수적 분할 목표일 뿐 현재 포털 허용 [한도가 아니다]"라고 적고, `RELEASE_MANIFEST.json:161`은 `"portal_attachment_limits_rechecked": false`다.
- 1:1 문의 첨부 "최대 5개, 개당 10MB"와 자격 서류 "최대 10MB"(공지 id 7)는 다른 기능의 제한이며 모델 제출 제한이 아니다.

**판정: 미확인.** P1 `SAVED_MODELS.zip` 590,000,061 B의 15분할(각 ≤40,000,120 B)은 확인되지 않은 한도를 가정한 것이다. 확인 절차(사용자): 문제 상세 → "모델 최종 제출하기" 클릭 → 모달의 파일 개수·크기 문구를 읽고 → **제출 버튼을 누르지 않고 닫기**. 08-25 운영본부 답변("'모델 최종 제출'은 리더보드 채점과 다른 항목이니, 예선 기간에는 누르지 마시고")대로 확인 목적으로만 연다.

## 2. P1 셀별 조합 정책 — 학습·선택 코드와 산출물 연결

기준 패키지: `C:\Users\cedis\Documents\OceanFinalRelease_20260907\P1\SOURCE_ONLY` (답안 SHA `57844ef2…`, 28.909341점).

### 2-1. 배포 실행 경로에서 확인한 구성 요소

| 구성 요소 | 코드 위치 | 산출물·상수 | 재생성 가능? | 근거 |
|---|---|---|---|---|
| O 모델(XGBoost offline, 700 iter) | `02_code/tree_recipe.json` `O_selection`; `02_code/run.py` | 학습 가중치(빈 폴더에서 재학습) | **예** | Codex whole-cold 7 fit 6,323.356초 → 답안 SHA 정확 재생(`docs/FINAL_RELEASE_20260907.md:56`) |
| B 모델(LightGBM 700, lr 0.035, leaves 63, min_child 60, subsample/colsample 0.85, reg_alpha 0.2, reg_lambda 1.0) | `tree_recipe.json` `B_parameters` | 학습 가중치 | **예** | 같은 cold |
| 후처리 임계(close_gap 0 / high 0.2 / low 0.1 / min run 12) | `tree_recipe.json` `postprocess`; `run.py` `apply_postprocess` | 고정 하이퍼파라미터 | 재선택 코드 없음(고정값) | `tree_recipe.json:26` "parameter_provenance: Historical training-only CV selection and preregistered B recipe; frozen for deployment retraining, not re-selected from Public scores" — 선택 로그로의 링크는 없음 |
| **셀 조합(add G-ORS/1, I-ORS/2; remove S-ORS/1,5,6, I-ORS/4)** | `02_code/composition.py:21-22` `ADD_CELLS`/`REMOVE_CELLS`; `run.py:170-171` `compose_tree(...)` | anchor replay `artifacts/p1_current_router_oof_anchor_v1/anchor.parquet`(SHA `fc5c594a…`) | **조합 연산은 예 / 셀 목록은 리터럴** | `composition.py:3-4` "Restores source-defined Boolean operations, NOT the lost policy-selection algorithm. Historical cells are passed explicitly for retrospective audit only." |
| GI spike 규칙 | `p1_qc.rules.detect_singleton_spikes`; `run.py:205` `compose_mstcn_spike` | 규칙 코드 | **예** | 라벨·점수 무관 규칙 |
| MS-TCN spike 추가(add-only) | `run.py` / 저장 가중치 | 가중치(bf16) | 저장 모델 추론은 정확; 재학습은 이번 cold에서 정확 재생됐으나 이전 시도(`BASELINE_REGEN_CHECK.md:9`, v5 재생 "최소 110 label 차이")가 있어 **타 머신 정확 재생은 미보장** | 두 기록 모두 Codex 문서 |

조합 연산의 정의(`composition.py`): router = B ∪ (O-only 행 ∩ add 셀) − (B-only 행 ∩ remove 셀); 최종 = router ∪ MS-TCN ∪ (GI spike ∩ ¬base). 이 부분은 O·B 출력만 있으면 결정적으로 재생되므로 재현 검증 4항(학습 산출물 제거 후 예측 재생성)의 대상으로 문제가 없다.

### 2-2. 셀 목록이 정의된 곳 (모두 리터럴)

1. `OceanFinalRelease_20260907/P1/SOURCE_ONLY/02_code/composition.py:21-22` — 배포 코드.
2. `scripts/package_preregistered_submission_20260826.py:56-62` — 08-26 라운드 C 포장 스크립트. `add_router = o_only & ((station=="G-ORS")&(layer==1) | (station=="I-ORS")&(layer==2))`, `remove_router = b_only & ((station=="S-ORS")&isin(layer,[1,5,6]) | (station=="I-ORS")&(layer==4))`.
3. `scripts/build_p1_current_router_oof_anchor_v1.py:84-85` → `artifacts/p1_current_router_oof_anchor_v1/manifest.json:18-22` (`add_only_when_o1_b0`, `remove_only_when_o0_b1`; 421,032 OOF 행, additions_vs_b 43, removals_vs_b 31).
4. `scripts/p1_historical_path_audit_20260906_v1.py:21-22` — 09-06 회고 감사(같은 리터럴; `:147-148` "selection_scope: Historical prescribed cells; original selector NOT recovered. Same development OOF was used for historical selection; not fresh.").

### 2-3. 셀을 고른 근거 — 설명만 존재

- 라운드 C 보고서 `Downloads/해양 해커톤 제출용/20260826_round_C_preregistered_P1x3_P2x1/report-source.md`: `:7` "현 베스트 B와 구 모델의 불일치 중 로컬 OOF에서 시간 블록을 넘어 효용이 확인된 정점-층만 되돌리는 보수적 disagreement router"; `:28` "동결된 로컬 OOF에서 B와 구 모델 O의 불일치만 다룬다"; `:34` "B F1 0.86467009 대비 후보 F1 0.86690000, delta +0.00222991 … bootstrap delta CI90은 [+0.00091805, +0.00370742] … 3개 시간 fold 중 2개가 개선하고 1개는 −0.00022479"; `:106` "P1 router의 작은 OOF 개선은 독립적인 새 test가 아니라 선택에 사용한 검증면의 결과".
- `SET_MANIFEST.json` lineage: `router_additions: 217, router_removals: 12`(test 제출 수준). anchor manifest의 43/31은 train OOF 수준이라 서로 다른 집합이며 모순이 아니다.
- `scripts/analyze_rounde_local_official_calibration_20260827.py:178-192`: 셀별 효용의 **사후** 진단(`G_add_beneficial/harmful`, `test_cell_counts {"G_add": 81, "I_add": 136, "removal": 12}`). 선택 이전에 실행된 선택기가 아니다.
- 검색 결과: 정점×층 격자에서 효용을 계산해 임계로 셀을 채택하는 스크립트는 `scripts/`, `src/`에 없다. 이름이 비슷한 스크립트를 확인했으나 다른 목적이다: `audit_p1_checkpoint_disagreement_cell_20260828.py`(e125 체크포인트 disagreement 셀 재현, 다른 셀), `run_p1_ts_disagreement_20260905_v4.py`(T/S 특징 ablation), `p1_historical_path_audit_20260906_v1.py`(조합 재생, 선택 아님).
- Codex도 같은 결론을 기록했다: `reports/p1_historical_path_audit_20260906_v1/report-source.md:37` "router의 원래 셀 선택 알고리즘 전체는 아직 발견하지 못했다. 원형 셀과 OOF 증거를 회고적으로 복원했을 뿐"; `claim-source-ledger.md:26` "원래 셀 선택의 전체 알고리즘이 미확인인 점을 숨기거나, 사후 작성한 선택 알고리즘을 역사적 원본이라고 부르지 않는다." 패키지 `SOURCE_ONLY/README.md:14-15`도 "원래 router는 로컬 OOF에서 선택된 고정 일반 정책입니다. 전체 셀 선택 알고리즘은 복구되지 않았으며 이 패키지는 셀/HPO 재선택을 하지 않습니다"라고 적는다.

### 2-4. 구분 결과

| 구분 | 항목 |
|---|---|
| **실제 재생성 가능**(코드 + 산출물 연결됨) | O 학습, B 학습, 불리언 조합(add/remove/union), GI spike 규칙, 후처리 적용, anchor replay 검사, 저장 모델 추론 |
| **고정값이며 재선택 코드 없음** | 후처리 임계 4개, B 하이퍼파라미터, O 700 iter — 출처 문구는 "training-only CV selection"이며 선택 로그 링크 없음 |
| **설명만 존재** | 6개 셀을 고른 절차(라운드 C 보고서의 OOF 수치·bootstrap CI만 존재), MS-TCN 채택 결정 |
| **타 머신 미보장** | MS-TCN bf16 재학습 정확 재생(이번 cold는 정확, 이전 시도는 ≥110 label 차이) |

재현 검증 관점: 1항(과도한 상수 리터럴)에서 심사자가 볼 것은 정점·층 이름이 박힌 6개 셀 리터럴이다. 방어 근거는 "리더보드가 아니라 08-26 로컬 OOF로 정했다"는 문서(라운드 C 보고서, anchor manifest, tree_recipe `parameter_provenance`, 패키지 README)이지, 재실행 가능한 선택 코드가 아니다. 4항(산출물 제거 후 재생성)은 조합 부분이 결정적이므로 통과가 기대되나, MS-TCN 재학습의 타 머신 정확성은 별도 위험이다. 지금 바꿀 것은 없다. 패키지 README에 셀 목록의 출처(로컬 OOF, 리더보드 비사용, 선택 절차 미복원)가 이미 적혀 있음을 확인했다.

## 3. P2 투영 재계산 값이 Codex와 달랐던 원인

### 3-1. 세 번의 계산본과 Codex 정본

| 버전 | 규칙 | pooled | B1 | B2 | B3 | B8 |
|---|---|---:|---:|---:|---:|---:|
| v0 (09-07 04:30, `fable_evidence_scripts_20260907.md`) | 근사 PAVA(인접 평균 2회), NaN 무시 fmin/fmax, 불완전 프로필에도 적용, 방향 sign(deep−T1) | 1.23962 | 1.9883 | 1.3256 | 0.4633 | 0.2065 |
| v1 (09-07 05:40 재구현) | 정확 3점 PAVA, NaN 무시 fmin/fmax, 동률 skip | 1.239202 | 1.988283 | 1.323701 | 0.463964 | 0.206510 |
| v2 (09-07 05:50, 오늘 10:41 재확인) | Codex 규칙: 완전 프로필 + T1·deep 모두 유한, clip 후 정확 PAVA, 동률→increasing | 1.236732 | 1.985611 | 1.317290 | 0.463529 | 0.206510 |
| Codex `reports/p2_l120_s3_proj_20260907_v1/result.json` | 배포 코드 `scripts/p2_final_day_projection_20260907_v1.py`와 동일 규칙 | 1.236732066 | 1.985610960 | 1.317290363 | 0.463529468 | 0.206509503 |

검토 1(05:55)에서 "완전 프로필 판정·동률 방향 처리 차이 추정"이라고 적은 것은 추정이었다. 아래에서 원인을 행 단위로 확정했다.

### 3-2. 분해 재계산 (오늘 10:41 KST, 3.1초, 단일 스레드)

입력: `artifacts/p2_c3_multiseed_completion_20260906_v1/evaluation.npz`의 `natural_L120` 166,268행, `observations.csv`. 재계산 결과:

| 항목 | 값 |
|---|---:|
| 2/3/4층 완전 프로필 시각 | 55,258 |
| Codex 규칙 적용 프로필(T1·deep 모두 유한) | 55,224 |
| v1 규칙 적용 프로필(fmin/fmax가 유한) | 55,258 |
| 차이 프로필 | **34** = "T1 결측·deep 유한" 34 + "deep 결측·T1 유한" 0 |
| 차이 행 | **102** (fold별 B6 36 / B1 24 / B7 24 / B2 15 / B3 3) |
| 이 102행의 RMSE: raw / Codex 규칙(무변경) / v1 규칙 | 3.836124 / 3.836124 / **4.968331** |
| 102행만 Codex 규칙으로 되돌린 v1의 pooled | 1.2367320655 (= Codex) |
| 나머지 166,166행에서 v1 − v2 최대 절대차 | **0.0** |
| 동률 T1 == deep 프로필(행) | 20 (60) |

블록별 v1 − v2: pooled +0.002470, B1 +0.002672, B2 +0.006411, B3 +0.000434, B4 0, B5 0, B6 +0.002055, B7 +0.004725, B8 0. B4·B5·B8에 차이가 없는 이유는 그 블록에 T1 결측 완전 프로필이 없기 때문이다(위 fold별 분포와 일치).

### 3-3. 원인 설명

1. **주원인(차이의 100%)**: `np.fmin/np.fmax`는 NaN을 무시하므로 T1이 NaN이면 envelope가 `[deep, deep]`이 된다. 세 층 예측이 모두 deep 값으로 잘려 성층기(B1·B2·B6·B7)에서 오차가 커진다. Codex 규칙은 `np.isfinite(temp_1) and np.isfinite(deep)`를 요구해 이런 프로필을 **건드리지 않는다**. 102행에서 v1 규칙은 raw보다 나빴다(3.836 → 4.968). 배포 코드가 Codex 규칙이므로 test에서도 T1 결측 프로필은 무변경이며, 검토 1에서 test의 투영 대상 25,890행·불완전 프로필 171행 무변경을 확인한 것과 정합한다.
2. **동률 처리**: v1은 `sign(deep−T1)==0`이면 PAVA를 건너뛰고, Codex는 `increasing = temp_1 <= temp_5`로 increasing 처리한다. 그러나 동률이면 clip 구간이 `[a, a]`라 세 값이 이미 같고 PAVA는 어느 방향이든 no-op이다. 수치 영향 0 — "나머지 행 최대 절대차 0.0"으로 확인.
3. **v0의 별도 결함**(근사 PAVA, 불완전 프로필 적용, B2 무변경)은 v1에서 제거됐고, v0→v1 차이(1.23962 → 1.239202)는 그 몫이다. 이미 폐기된 계산이며 어떤 문서의 정본 수치에도 쓰이지 않는다.

### 3-4. 표현에 대한 주의

- 일치의 근거는 "방향이 같다"가 아니라 **행 단위 동일값**(최대 절대차 0.0, pooled 소수 10자리 일치)이다.
- 위 수치는 이미 노출된 retrospective natural OOF의 재계산이지 새 검증이 아니다. test 개선의 상한도 하한도 아니며, test에서의 효과 크기는 별도의 미확인 값이다. test에는 T5 17일 연속 결측 구간(전체의 29.3%)처럼 OOF에 대응물이 약한 조건이 있고(B4/B8 outage는 NOT_ESTIMABLE), B1은 OOF에서 +0.000121℃ 악화였다.
- 이 검토는 후보를 바꾸지 않는다. `P2_L120_s3_proj`(SHA `9c5fec38…`)의 배포 규칙이 위 두 변형 중 더 안전한 쪽(T1 결측 무변경)임을 확인했을 뿐이다.

## 4. 사용자 확인이 필요한 미확인 항목 (우선순위)

1. **모델 제출 마감 시각** — 공식 원문에 날짜만 있음. 공고문 PDF(07-27 첨부, 163.3 KB) 또는 1:1 문의로 확인.
2. **문제지 Ⅳ-2** — 재현 검증 항목(1항·4항은 09-02 공지로 확인, 2·3항 미확인)과 6시간의 적용 범위. 같은 PDF에서 "Ⅳ-2", "6시간" 검색.
3. **모델 제출 모달의 첨부 개수·크기 제한** — 모달을 열어 읽고 제출하지 않고 닫기. P1 15분할의 전제.
4. **P3 cpudet의 6시간** — 6시간 적용 여부와 별개로, 현 설계는 우리 PC에서도 5시간 안팎이라 "준수 증명 없음"으로 기재.

이 문서는 기존 문서를 수정하지 않는다. 검토 1(`FABLE_INDEPENDENT_REVIEW_20260907.md`) §1-1의 "완전 프로필 판정·동률 방향 처리 차이 추정" 문장은 본 문서 §3으로 대체해 읽는다.
