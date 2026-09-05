# P2-A/B 실제 학습 결과 — 기존 C 유지, 두 가설의 개선 범위는 분리해서 보존

작성: 2026-09-05 KST. 실행 근거: [승인된 점수 개선 설계 v2](../../docs/SCORE_IMPROVEMENT_PLAN_20260905_V2.md). 이 문서는 설계안의 미실행 상태를 소급 변경하지 않는 실행 영수증이다.

## 결론

**P2-A 신규 9 fits와 후속 P2-B 신규 3 fits를 모두 실제 학습·내부 테스트했다. 사전고정 가을 주평가에서는 기존 clean blockmask C가 이겨 새 제출 후보로 승격하지 않았다.** 기존 C의 3-seed 모델을 유지하며, 새 후보의 공식 점수는 미산정이다. 오늘 C의 공식 반환값 `RMSE 0.455143℃ / 27.622418점`은 새 모델의 성적이 아니다.

다만 모든 방법이 모든 조건에서 실패한 것은 아니다. M은 전체 계절 pooled에서 개선했고 R/MR은 인위적 가을 공개층 결측 구간에서 크게 개선했다. P2-B의 고정 절반 결합도 pooled를 개선했다. 이를 가을 intact 개선으로 바꾸어 주장하거나 결과를 보고 주평가·라우팅을 바꾸지 않았다.

## 동일 평가 모집단과 대조

- 원본은 배포 `observations.csv` 하나, SHA-256 `cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a`다. 공식 test/sample/baseline/hidden 입력은 읽지 않았다.
- 원본 target 수온이 유한하고 공개 수온이 2개 이상인 학습 적격 행은 166,268개다. 원본 QC 필드가 없어 실제 hidden QC 모집단과 완전히 같다는 보장은 없다. 임의 outlier 삭제는 없다.
- 내부 평가 총 69,850행: 2024 Sep–Oct 26,273행, 2025 Jul–Aug 26,693행, 2025 Nov–Dec 16,884행. full-history 복원 분할과 ±7일 purge를 유지했다.
- **주평가는 intact 2024 Sep–Oct**다. 다른 두 계절 pooled, bias, 층별, 결측 스트레스는 보조표이며 주평가를 사후 변경하지 않았다.
- target layer 2/3/4의 temp·psal을 공개 프로파일에서 제외한 뒤 baseline·scale·특징을 만들었다. 기존 C 9개 모델의 새 재추론과 기존 OOF의 key/truth/fold/prediction이 모두 exact 일치했다. 따라서 C를 다시 학습하지 않았다.
- 첫 seed `20260901`끼리 선별했다. 표의 C 1-seed와 최종 유지 C 3-seed를 혼동하면 안 된다. 기존 C 3-seed 평균의 주평가/pooled는 `0.488284326 / 0.859249914℃`다.

## A: 손실 × 가중치 2×2 실측

| arm | 데이터 손실 / 가중치 | Sep–Oct 주평가 RMSE ℃ | pooled RMSE ℃ | 신규 fits |
|---|---|---:|---:|---:|
| C, 첫 seed | normalized Huber / domain | 0.465330203 | 0.896730522 | 0, 기존 재사용 |
| M | 절대 ℃ MSE / domain | 0.516945976 | 0.834329697 | 3 |
| R | normalized Huber / 원본행 균등 | 0.505493478 | 0.920044589 | 3 |
| MR | 절대 ℃ MSE / 원본행 균등 | 0.528891834 | 0.897039408 | 3 |

v23 구조·공개입력 정규화·blockmask·60 epochs·batch 4096·learning rate 0.001·weight decay 0.0001은 같다. 절대 ℃ MSE는 `scale² × (z_hat−z)²`로 구현했다. 기존 normalized-Huber input-gradient penalty, 그 domain 가중치, 계수 0.01을 네 arm 모두 유지했다. 데이터 손실 단위가 바뀌어 penalty의 상대 크기는 달라진다는 한계는 남는다. 원본/증강 복제의 가중치 합은 원본별로 보존했다.

C가 첫 seed 선별에서 이겼으므로 후보 추가 2 seeds × 3 folds, 즉 6 fits는 실행하지 않았다. 최종 `NO_PRIMARY_IMPROVEMENT_NEXT_P2_B`, 선택 `C_mean`, C 대비 주평가 delta 0이다. 계획에 정한 B로 바로 이어갔다.

### 결측 스트레스: 버리면 안 되는 단서

실행 전 정한 2024-10-18부터 11-01 전까지 T5/S5를 함께 가린 뒤 입력·baseline·scale을 재계산했다. 아래는 첫 seed 비교이며 인위적으로 가린 6,031개 supported 행이다. 전체 가을 26,273행의 support는 유지됐다.

| arm | 가린 구간 RMSE ℃ | 전체 가을 스트레스 RMSE ℃ |
|---|---:|---:|
| C, 첫 seed | 0.465796410 | 0.514116777 |
| M | 0.300471642 | 0.534506887 |
| R | 0.224904126 | 0.515495135 |
| MR | 0.229025255 | 0.537628770 |

R의 가린 구간 개선은 −0.240892283℃이지만 전체 가을 스트레스는 C와 거의 같고 intact 주평가는 악화됐다. 이는 결측 상태에 따라 학습 가중치의 효과가 달라질 수 있다는 **개발용 가설 근거**다. 공식 결측 상태와 일치한다는 증거, 3-seed 확인, 별도 평가면의 증거가 아직 없으므로 routing·혼합비를 추가로 맞추거나 새 제출본을 만들지 않았다.

## B: 고정 물리 프로파일 트리 실측

자세한 구현/독립 QA는 [B 보고서](../p2_physical_profile_tree_20260905_v2/report-source.md)와 [B 결과](../p2_physical_profile_tree_20260905_v2/result.json)에 있다. 고정 LightGBM 한 개, 69개 특징, 400 rounds, CPU 2 threads를 사용했다. 기존 `src/p2_restore/model.py::_estimator`의 recipe를 재사용하고 resource thread만 2로 제한했다. target 누출을 막은 공개 층들의 절대 T/S, 물리 수심 slot, ±6/12h 공개 수온 차이, 이웃 수심차, 달력 주기를 사용했다. 결측 증강을 시간차 특징 생성보다 먼저 적용했다.

| 첫-seed 정책 | Sep–Oct 주평가 RMSE ℃ | pooled RMSE ℃ | 신규 fits |
|---|---:|---:|---:|
| C | 0.465330203 | 0.896730522 | 0 |
| tree | 0.710383091 | 0.919654290 | 3 |
| 사전고정 50:50 평균 | 0.518502426 | 0.823186375 | 0 |

고정 평균의 pooled 개선은 −0.073544146℃이지만 주평가가 +0.053172223℃ 악화됐다. C가 이겼으므로 B 추가 6 fits도 실행하지 않았다. `NO_PRIMARY_IMPROVEMENT_P2_AB_COMPLETE`, C 유지로 종료했다. 학습된 stacking/calibration은 사용하지 않았으며 공개 점수에서 계수를 역산하지 않았다.

## 실제 비용과 재사용 효과

| 항목 | 실행 / 재사용 | 실측 또는 절약 범위 |
|---|---|---|
| 기존 C | 9개 historical 모델 exact 재사용 | 재학습 0; 새 재추론으로 동일성 검증 |
| A | 새 9 fits | 전체 runner wall 216.703초, GPU 1개 독점·CPU 1 thread·DataLoader 0 |
| A 추가 seeds | 조건 불충족으로 미실행 | 6 fits를 실행하지 않음 |
| B | 새 3 fits | 전체 runner wall 39.781초, CPU 2 threads, GPU 미사용 |
| B 추가 seeds | 조건 불충족으로 미실행 | 6 fits를 실행하지 않음 |
| A+B | 새 12 historical fits | 두 runner wall 합 256.484초; 설계/코딩/검증 시간은 이 숫자에 포함되지 않음 |
| calibration/fulltrain/official/CSV/upload | 전부 0 | 승격 후보가 없으므로 불필요한 배포 학습·중복 제출 없음 |

A+B 최대 신규 historical 24 fits 중 12 fits만 필요했다. 기존 C 9 fits도 다시 돌리지 않았다. 이 재사용/분기 중단은 실험 결과를 임의 수정한 것이 아니라 사전정의 실행 계약대로다. 절약된 벽시계 시간은 직접 측정하지 않아 추정하지 않는다. 두 실행은 정상 terminal이며 GPU·CPU는 해제됐다.

## QA와 재현 자산

- `tests/test_p2_objective_alignment_20260905_v2.py` 7개, `tests/test_p2_physical_profile_tree_20260905_v2.py` 6개, 기존 score-repair 14개: **27 pytest PASS**.
- 새 runner 2개·독립 QA 2개·전용 tests 2개에 **Ruff PASS**.
- A [independent-recalculation.json](independent-recalculation.json) 34-check PASS, B [독립 재계산](../p2_physical_profile_tree_20260905_v2/independent-recalculation.json) 30-check PASS. 동일 키/분모·원본증강질량·fit cap·저장모델 replay·RMSE·hash·공식 접근0을 검증했다. A의 34개에는 강조한 스트레스 두 분모/수치의 독립 재계산도 포함한다.
- B 독립 QA 결과를 JSON으로 저장할 때 NumPy bool 직렬화 오류가 한 번 있었고, **QA 스크립트만 native bool 직렬화로 수정 후 재계산 PASS**했다. 학습 runner/config/seal/모델/예측/결과는 변경·재학습하지 않았다.
- A runner SHA-256 `f0600e94135a41aef2e1a8c43a0d7c6e7eb03861ab750993cb0927398e91e8aa`, config `d5cf02c83fc3060fab9e767211b677a5eaba7087d969970e588ea7a4a181011b`.
- B runner SHA-256 `76138e8c6766013851ba17e78525522bd59b97111632889bb6a87bb961e407d3`, config `cf6c22ee0311bdeb99645362c92f66ebb11b076ff09c250c038c06534b325ea3`.
- 각 [실행 전 seal](preregistration-seal.json), [A result](result.json), [B seal](../p2_physical_profile_tree_20260905_v2/preregistration-seal.json)가 source/dependency/OOF 지문을 연결한다. 로컬 모델·행별 예측은 각 `artifacts/<experiment_id>/03_training`, `04_models` 아래 별도 보존했다. 이를 문서/소스처럼 stage하거나 외부 업로드하지 않았다.

검증 재실행은 학습이 아닌 `scripts/qa_p2_objective_alignment_20260905_v2.py --execute` 및 `scripts/qa_p2_physical_profile_tree_20260905_v2.py --execute`다. 학습 runner의 소비된 attempt lock을 지우거나 동일 ID를 다시 실행하면 안 된다.

## 다음 판단과 해석 한계

현재 공개 기준은 `27.622418점`, 새 예상 점수는 **미산정**이다. 손실 또는 구조의 단순 교체만으로 intact 가을이 좋아진다는 가설은 이번 첫-seed 화면에서 지지되지 않았다. 기존 3-seed C의 낮은 첫 seed만 골라 제출하는 것도 이번 실행에 포함되지 않는다.

다음 연구를 승인한다면 우선 질문은 “특정 결측 상태의 보완을 전체 intact 정확도 손실 없이 사용할 수 있는가?”다. 그러나 이번 스트레스 결과로 선택한 rule을 같은 스트레스 표에서 검증됐다고 주장할 수 없다. target-free 결측상태 분류, 여러 사전고정 outage episode, outer-train 안 inner OOF와 별도 비교면을 설계해야 한다. 지금은 추가 ad-hoc 분기를 실행하지 않고 증거를 남겼다.

모든 평가면은 반복 노출된 historical development다. 단일 가을·첫 seed 위주 선별·원본 QC 부재·인위적 outage라는 한계가 있다. 선택이 C 그 자체여서 독립 QA의 선택후 delta bootstrap가 `[0,0]`인 것은 C와 C의 항등 비교이지 새 후보 불확실성이 0이라는 뜻이 아니다. `validate-data`와 `analyze-data-quality` 지침은 분모/단위/출처/동일성/지원범위 구분에 적용했다.
