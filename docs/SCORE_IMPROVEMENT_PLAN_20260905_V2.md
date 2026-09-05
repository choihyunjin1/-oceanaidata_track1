# 점수 개선 실행 설계 v2 — 새 모델보다 학습·배포·채점의 불일치부터

작성: 2026-09-05 KST. 상태: **설계 완료 / 미실행**. 기준 HEAD: `535f94a1791f2398f82ad27659b58701513ab327`와 현재 dirty worktree. 기존 계획과 실험 기록을 대체·수정하지 않는 후속 문서다.

## 1. 결론

다음 실행은 **P1 연도에 안전한 수심 특징과 강한 기존 부품 복구, P2 절대 ℃ 손실, P3 저자유도 직접 SSE 결합**을 병렬로 시작한다. 첫 가설이 실패하면 아래에 정한 두 번째 가설까지 이어간다. 같은 실험의 근접 파라미터를 무제한 바꾸거나 +0.1점에서 전체 작업을 종료하지 않는다.

Deep Research로 평가·손실의 근거를 확인하고, 현재 Astra와 문제별 세 병렬 검토가 저장소 코드·집계 결과를 직접 대조했다. 다음 판단은 문헌의 성능 약속이 아니라 **우리 코드와 실제 결과에서 도출한 검증 가설**이다.

| 문제 | 확인된 사실 | 다음에 시험할 가설 |
|---|---|---|
| P1 | train의 station×year×layer 수심 사전에 2026 키가 없어 이번 공식 169,011행의 nominal-depth가 모두 missing. 오늘 모델은 최고점 MS-TCN 조합이 아닌 2-tree control | 연도 lookup을 제거하면 배포 손실을 줄일 수 있는가? 적격 MS-TCN·router 부품을 복구하면 tree 단독보다 강한가? |
| P2 | 정규화 residual의 SmoothL1와 domain 가중치를 학습하지만 채점은 절대 수온 RMSE. 내부 개선은 주로 다른 계절에서 발생 | 같은 구조가 실제 ℃ 제곱오차를 학습하면 가을 복원 성능이 좋아지는가? |
| P3 | clean 재학습은 적법한 옛 원형 수준을 재현. 현 router는 최종 결합 SSE가 아닌 성분별 log-loss를 예측 | 기존 성분의 직접 결합 또는 낮은 자유도의 편향 보정만으로 개선되는가? |

**불일치의 존재는 확인됐지만, 그것이 점수 하락의 주원인이거나 수정 후 상승한다는 사실은 아직 입증하지 않았다.**

## 2. 이번 계획의 정확한 기준선

아래는 [공식 제출 영수증](../reports/official_score_repair_submissions_20260905_v1/receipt.json)에 기록된 9월 5일 19:28~19:29 KST 반환값이다. 개별 모델의 내부 점수와 공식 점수는 서로 다른 모집단이다.

| 문제/후보 | 오늘 Public 지표 | 오늘 Public 점수 | 같은 계보의 내부 지표 |
|---|---:|---:|---:|
| P1 clean O/B control | F1 0.790733 | 27.771400 | pooled F1 0.851174 |
| P2 v23 blockmask 3-seed | RMSE 0.455143℃ | 27.622418 | pooled RMSE 0.859250℃ |
| P3 clean baseline | RMSE 0.607183m | 23.696500 | pooled RMSE 0.779105m |
| P3 6h-only TabPFN 혼합 | RMSE 0.608143m | 23.681268 | pooled RMSE 0.778340m |

- P3의 이번 고정 혼합은 공식 RMSE +0.000960m, 점수 −0.015232로 baseline보다 낮았다. 다른 혼합비 탐색으로 즉시 이어가지 않는다.
- P2는 비증강 control을 공식 제출하지 않았으므로 blockmask의 **공식 증분 효과는 미확인**이다.
- P1의 과거 우리 최고는 F1 0.833548 / 28.909341점이었다. 오늘 대비 1.137941점은 복구를 검토할 역사적 격차이지 새 계획의 예상 상승폭이 아니다. 두 파일은 모델 구성이 다르다.
- P2/P3의 외부자료 또는 Public 역산 계수 계보는 새 후보의 적격 control이 아니다. 과거 숫자만 가져와 복구 목표로 강제하지 않는다.
- 새 실험의 예상 공식 점수는 아직 산정할 수 없다. 매 보고에 실제 F1/RMSE, 현재 공식 기준점, 예상점수 `미산정`을 함께 쓰고 채점 후 실측으로 갱신한다. 반환 점수로 기울기·정답·Public membership·보정계수를 역산하지 않는다.

## 3. 공통 실행 계약

1. 최상위 규칙은 [운영진 정책](../00_ORGANIZER_DATA_POLICY.md), 원본 README, 문제별 must-read다. 배포 관측만 사용하고 외부 KMA/ERA5/인터넷 원자료는 검증용으로도 금지한다. 합성-only 사전학습 예외는 버전·가중치 출처·동봉·로컬 로드·6시간 조건까지 증빙해야 한다.
2. 모든 새 실험은 새 ID·설정·코드 hash·fit budget·평가키 manifest를 실행 전에 고정한다. 기존 one-shot/lock/config/result를 수정하거나 재개하지 않는다.
3. **hard gate:** 자료 적격성, target 누출 없음, 동일 평가키, 유한값/스키마, 재학습·저장·재로딩 재현, 최종 6시간 조건. **soft evidence:** 월별 악화, seed 편차, CI, 특정 구간 위험, anchor 제거량. 자체 +3점·모든 월 양수·제거0을 자동 탈락 조건으로 추가하지 않는다.
4. 점수 개선 후보는 사전고정 주평가의 평균 개선으로 순위를 정하고 위험표를 함께 낸다. CI가 0을 포함해도 자동 탈락하지 않으며, 평균 악화 후보는 개선 후보와 분리한다. 대안이 전혀 없고 새로운 질문에 답하는 경우에만 별도 `INFORMATION_ONLY` 후보로 다루며 같은 파일을 재제출하지 않는다.
5. 역사 평가면은 이미 반복 노출됐다. 새 실행은 **재사용 historical development 평가**이며 fresh holdout 또는 확증적 유의성으로 포장하지 않는다. 같은 행의 delta와 블록 단위 불확실성을 기록한다.
6. 전체 OOF를 보고 계수를 맞춘 뒤 같은 OOF를 독립 성능으로 보고하지 않는다. P3는 엄격한 과거 OOF 순차 검증, P2는 outer-train 안의 masked inner OOF를 사용한다. 단순 leave-one-outer-fold-out은 다른 base 학습에 해당 outer 정답이 들어가 있을 수 있어 불충분하다. [공식 stacking 설명](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html), [시간·그룹 분할 안내](https://scikit-learn.org/stable/modules/cross_validation.html).
7. 원본별 outlier 임의 삭제는 이번 기본안에 없다. 오류 의심과 실제 극한을 구분하고, 정답이 있는 평가행은 모두 유지한다. 특히 P1 이상 탐지와 P3 고파고에서 큰 값을 지우면 과제 자체를 바꿀 수 있다.
8. 내부 테스트와 독립 QA를 마친 후보만 봉인 후 공식 입력에 추론한다. 공식 답안으로 calibration하지 않고 최종 제출 ZIP도 연구 폴더와 분리한다. 이번 **설계 턴에서는 학습·코드 수정·CSV·업로드·Git 변경을 실행하지 않았다.**

## 4. P1 — 배포 계약을 고친 뒤 강한 부품을 복구

### P1-A: year-safe depth, 최우선

근거: [현재 특징 코드](../scripts/run_p1_score_repair_20260905_v1.py)의 `stats_fit`/`feature_pair`, [배포 보고서](../reports/p1_clean_control_fulltrain_20260905_v1/report-source.md)의 전체 nominal-depth missing 기록. 단순 station×layer median fallback은 층 번호가 연도마다 같은 물리 수심을 뜻하지 않아 채택하지 않는다.

- 변경은 nominal-depth 계약 하나: **현재 행의 관측 depth를 고정 2m 단위로 반올림하고, 진짜 결측은 explicit unknown**. 연도 사전에 의존하지 않는다. 기존 [current-depth 구현](../src/p1_qc/features.py)의 수학을 재사용하되 offline 시간 특징 전체를 causal로 바꾸지 않는다.
- train/inner/outer/fulltrain/inference에 동일 함수를 쓴다. 관측 depth와 nominal-depth를 구분해서 보존한다. raw depth 이상·결측 구간도 평가에서 삭제하지 않는다.
- synthetic QA: 시간·관측을 고정하고 year 키만 변경해도 수심 특징 동일; missing 표시 정확; 행 순서 불변; category fit은 train-only. unseen-year와 진짜 raw missing을 별도 집계한다.
- 동일 Q2/Q3/Q4, purge21일, inner60일, seed20260813, 같은 XGB/LGBM recipe를 사용한다. 새 candidate **O/B×3fold×inner/outer=12 fits**. 기존 exact control 재사용이 불가능하면 control12를 추가해24fits 상한이다.
- primary는 pooled TP/FP/FN의 F1. Q2는 배포 상반기와 관련된 부분표, Q3/Q4는 계절 스트레스표다. Q1을 새 blind 데이터로 부르거나, 단일 정점뿐인 2024→2025 이동을 모든 정점의 동일 검증으로 취급하지 않는다.
- 기존 고정 threshold 결과도 진단용으로 병기하되 실제 후보 선택은 earlier-inner에 한정한다. **최종 모델은 마지막60일 final-inner용2fits로 정해진 동일 grid에서 threshold를 선택한 뒤 fulltrain2fits**. 오늘의 Q4 threshold를 이유 없이 그대로 복사하지 않는다. 이 final-inner 절차도 배포 계절을 완벽히 대표하지 않으므로 한계로 기록한다.
- 비용: 과거18fit screen 전체763초, fulltrain2fits160초. A와 QA에는 우선45분 계획 예산을 두고 실제 fit·CPU·시간을 기록한다. 보장시간이 아니다.

### P1-B: 기존 최고점 경로의 적격 부품 복구

A와 병행하여 source/provenance만 정찰한다. 오늘 2-tree control과 과거 MS-TCN e150 3seed+O/B router+GI spike2는 다른 모델이다.

1. `배포 train → O/B → 일반 router → e150 → 일반 spike 규칙`을 분해한다. `router_anchor.csv`와 GI 특정 키 patch를 완성답안 입력으로 복사하지 않는다.
2. [router builder](../scripts/build_p1_current_router_oof_anchor_v1.py)의 station/layer 규칙, [후보 원장](../reports/historical_model_reaudit_20260831_v1/candidate-ledger.json)의 Round D, [GI builder](../scripts/build_deadline_probe_set_20260828.py)를 대조한다. 사전고정 완성 후보 간 공식 비교와 점수 역산을 구분한다.
3. 출처가 적격인 e150 OOF가 동일 split/purge/key를 만족하면 backbone0-fit으로 tree 단독/적격 기존 결합/새 tree 결합을 대조한다. 다른 행·미래를 학습한 checkpoint는 억지로 맞추지 않는다.
4. 출처 불명인 부품만 train-inner에서 다시 적합한다. **전체 P1 과거 모델을 일괄 폐기하지 않는다.** 필요한 OOF가 없으면 새 e150 confirmation은 최대3fold×3seed=9fits, fulltrain3fits의 별도 비용 항목이다. 단일 GPU 예산과 final6시간을 확인하기 전 자동 시작하지 않는다.
5. 결합은 no-op 포함 최대3개 일반 정책으로 제한한다. 불확실한 행별 patch는 후보에서 제외한다.

### P1-C: 제한적 decoder 재검토

A/B 뒤에만 OFF와 이미 존재하는 고정 always-ON 두 정책을 비교한다. 기존 inner선택 전략 −0.000883과 always-ON 진단 +0.001617은 다른 결과다. 후자는 이번 development 후보로 명시할 수 있지만 과거 FAIL을 PASS로 바꾸지 않는다. lambda/threshold를 추가 탐색하지 않는다.

**다시 하지 않음:** long-flank 재튜닝, 실패한 trial18 frozen confirmation 반복, Q3 우승 정책과 Q4 fallback을 결과 보고 사후 조합.

## 5. P2 — 목적함수를 ℃로 되돌리고 배포 계절을 따로 본다

### P2-A: 손실 × 가중치 2×2 대조, 최우선

현재 `z=(T−baseline)/scale`에 domain-weighted SmoothL1를 적용한다. 작은 residual에서는 절대 ℃ 오차가 대략 `w/scale²`로 가중되고 큰 residual은 선형 처리된다. 공식 rowwise RMSE와 다른 목적이다. 이것이 나쁘다는 결론은 아니므로 실제 대조한다. [PyTorch MSE](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html), [SmoothL1](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html).

| arm | 데이터 손실 | 데이터 가중치 |
|---|---|---|
| C | 현재 normalized SmoothL1 | 현재 layer×month×day weights |
| M | `(scale×(z_hat−z))²`, 절대 ℃ MSE | 현재 domain weights |
| R | 현재 normalized SmoothL1 | 원본 행 균등 |
| MR | 절대 ℃ MSE | 원본 행 균등 |

- 입력 정규화·v23 구조·blockmask·60epochs·batch4096·lr0.001·weight_decay0.0001·기존3fold와purge7일은 보존한다. `scale`은 target-free 공개 프로파일에서만 계산한다.
- 증강 전 원본 행의 가중치 총량을 원본/복제에 배분한다. 기존 normalized-Huber 기반 gradient-penalty와 **그 계산용 기존 domain weights 및 계수0.01**을 네 arm에서 동일하게 유지해 정규화항까지 바뀌는 혼동을 막는다. 데이터 손실 단위 변화에 따른 상대 정규화 강도 차이는 남으므로 진단표에 기재하고 무제한 계수 탐색은 하지 않는다.
- 첫 seed20260901,3fold×새3arm=9fits. C의 동일 artifact 재사용 검증 후 한 arm만 선택해 seeds20260902/03으로 추가6fits. **최대15 new historical fits**, 승자fullfit3. C 재현이 불일치하면 성능 비교를 중지하고 계약 오류를 먼저 확인한다.
- primary를 이번 실행 전에 **2024 Sep–Oct의 intact RMSE**로 고정한다. 동일 가을의 label-independent T5/S5 outage 스트레스와 전체3fold pooled는 반드시 함께 표시한다. 나중에 잘 나온 계절로 primary를 바꾸거나 임의 혼합점수를 만들지 않는다.
- 가을 표본이 한 해뿐인 선택의 취약성을 공개한다. primary 동률이면 전체 pooled, 다음은 낮은 복잡도로 고른다. 내부 개선이 없으면 B로 진행한다.
- 과거 `v32 score-aligned MSE`는 normalized target·prefix학습·bin17 comparator의 다른 실험이다. 이번 절대 ℃/clean restoration 실험을 이미 실패한 것으로 분류하지 않는다.
- 비용: 기존24fits end-to-end435초. A+QA **10~20분 계획 예산**. 새 성능/시간은 미측정이다.

### P2-B: 절대 프로파일 정보를 쓰는 단일 트리

현재 relative T/S와 주기 중심 DeepSets에서 빠진 absolute 수온·염분·시간문맥을 하나의 고정 LightGBM이 보완하는지 시험한다.

- 같은 공개층 mask, 수심 기준 정렬 슬롯, absolute T/S, 층간 차이, 기존 interpolation, 검증된 ±6/12h 공개층 특징. 목표는 절대 ℃ residual/L2.
- 목표 temp/psal를 **모든 lag·rolling·baseline 계산 전에** 가린다. fold purge7일을 유지하고 feature dependency12h로 정확히 기록한다. T5 outage도 시간 특징 계산 전에 적용해 이웃 특징으로 가린 값을 되가져오지 않게 한다.
- 기존 트리 recipe 하나를 코드 출처와 함께 실행 전 봉인한다. 3fold×1seed=3fits, 개선 시 추가2seed×3fold=6fits, 최대9fits; fullfit3. 첫 fold는 시간 측정에도 사용하며 성능을 보고 조기 범위를 확장하지 않는다. 초기30분 예산.
- tree 단독부터 비교한다. DeepSets와 adaptive stack은 **outer-train 내부에서 추가 생성한 inner OOF가 있는 경우만** 허용한다. 없으면 학습없는 고정50:50과 no-op만 개발 대조하고, 부족한 OOF를 전체OOF 혼합학습으로 대체하지 않는다.
- 새 bias/projection 추가는 이번 A/B와 동시에 하지 않는다. 불필요한 강제 수온 단조성·이상치 삭제·old bin17/OAS 보정 복구도 하지 않는다.

## 6. P3 — 새 백본 없이 시작하고, 이어서 사건 가중치를 바꾼다

### P3-A: 1~2 자유도 직접 결합, 최우선

적격 clean OOF의 `single`, `multi`, `persistence`, `final_prediction`만 사용한다. 현재 single/multi 저장값은 이미 `[0,30]` clip 후이며 unclipped 원예측으로 부르지 않는다. 기존 router의 log component-loss Ridge→softmax는 조합의 오차 공분산을 직접 최소화하지 않으므로 아래 두 후보를 고정한다.

1. **Long-simplex:** 12/18/24h에 공통 비음수 합1의 single/multi/persistence 가중치, 실질 자유도2. 단기3/6/9h는 기존 final을 그대로 둔다. 과거 train OOF의 해당 long-lead SSE를 최소화한다.
2. **Global-bias:** 기존 final에 과거 train OOF 평균 `(truth−prediction)` 한 개를 더한다. station별·lead별 보정은 하지 않는다.

- no-op 기준선 포함. 두 후보를 동시에 결합하지 않는다. 출력의 기존 범위처리는 보존하고 실제 후처리 후 성적을 평가한다.
- fold1은 baseline 그대로. fold2 메타학습은 이전49cases, fold3은 이전128cases만 사용한다. lead target가용시간과 기존 purge까지 검증한다. **미래 fold를 LOFO 보정학습에 넣지 않는다.**
- 정책별 historical2 소규모 fits; 선택된 정책의 전체181cases 배포용 fit1. 연구 basefit0으로 시작 가능하나 최종 완전 재현에는 원래 source→OOF3fold×2+full2=8CatBoost가 여전히 필요하다.
- primary pooled RMSE, 이어 실제 개입이 평가되는 fold2/3, 각lead/station/episode peak를 별도 표시한다. 평균 signed error−0.10339m는 보정 후보를 시험할 단서이지 배포 시 그대로 더할 정답이 아니다.
- 기존6h TabPFN 비율 재탐색, station×lead18 자유도 calibration, Public 역산α 계보 복구는 하지 않는다.

### P3-B: 긴 폭풍의 반복 anchor 가중치 완화

A 성공 여부와 별개로 GPU 시간이 있으면 다음의 독립 가설까지 실행한다. 희소181개 평가 사건과 촘촘한24,360개 학습 anchor의 대표성 차이를 시험한다.

- CatBoost·591features·split·target·기존 threshold weight는 보존. 새 weight만 `기존 weight / sqrt(outer-train 내 episode anchor 수)`로 하고 평균1 정규화한다.
- 기존 사건분할/가중치의 순수 함수를 검토해 재사용한다. background가 하나의 거대 사건으로 잘못 묶이지 않는 synthetic QA를 추가한다. 미래 validation/test를 보고 사건 크기를 계산하지 않는다.
- 공식 조건에 맞는 최소 context/target 지원·6leads·78h 분리는 유지한다. first-eligible가 운영진의 정확한 표집법이라고 단정하지 않고 context 시각을 역추적하지 않는다.
- 기존 fold별 seed `[20260816,20260817,20260818]`을 첫 세트로, 두 번째를 `[20260916,20260917,20260918]`로 고정한다. control 첫 세트가 재사용 가능하면 control 추가6fits+candidate12fits=**18 new historical CatBoost fits**. 승자 배포는 seeds20260817/20260917의 single/multi4fits.
- 동일 시점/동일 seed로 paired 비교한다. 각 arm의 두 seed single/multi 예측을 먼저 평균하고 **그 arm의 새 OOF로 기존 동일 router 설정을 과거-only 방식으로 재적합**한다. 옛 saved router 계수는 복사하지 않는다. 기존 persistence shrink0.2는 유지한다. 작은 router fits는 arm당 historical2, 선택된 배포 arm1을 base fits와 별도로 센다.
- A의 simplex/bias와 B의 사건가중 변경을 동시에 넣지 않는다. A/B 최종 후보는 동일181case keys의 완성정책 pooled RMSE로 비교한다. A+B 결합은 각 단독 결과 후 별도 후보로 등록하기 전에는 하지 않는다.
- 과거single+multi fold pair177/196/237초, full pair255초를 외삽하면 특징 재생성·QA 포함 **45~75분 계획 예산**이다. GPU contention과 source 준비에 따라 더 걸릴 수 있다.

## 7. 병렬 순서와 제출 기회

조사 중 시각은 2026-09-05 **19:39 KST**였다. 오늘 남은 횟수는 19:29 영수증 기준 P1/P2/P3=**2/2/1**이며 실행 시 홈페이지에서 다시 확인한다. 날짜가 바뀌면 이 숫자를 재사용하지 않는다. 최종 마감일은 공지 간 차이가 있었으므로 라이브 확인 없이 단정하지 않는다.

| 자원/순서 | 시작 작업 | 다음 분기 |
|---|---|---|
| CPU lane | P1-A + P1-B 출처 정찰 | 적격 OOF 결합, P1 final-inner/전체학습 |
| GPU 1순위 | 짧은 P2-A | P2-B는 CPU 중심으로 분리 |
| 작은 CPU 메타 lane | P3-A | 준비 후 P3-B GPU 대기 |
| GPU 2순위 | P3-B | 시간이 검증된 경우에만 P1 e150 새 학습 |
| root QA | key/hash/동일 split/권한·성적 대조 | 승인 범위의 별개 후보만 제출 |

GPU heavy 프로세스를 세 문제 모두 무작정 동시 실행하지 않는다. GPU는 단일 소유 큐, CPU threads는 총량과 기존 프로세스를 확인해 배분한다. 실행 중인 모델의 seed/epoch/설정은 자원 사정이나 중간 성능 때문에 바꾸지 않는다.

- 목표는 실행 승인 후 첫60~90분 내 A군 내부테스트·독립QA·후보 준비. 이것은 일정 목표이지 모든 개선이 나올 보장이 아니다. 각 실패 시 사전정의 B로 이어가되 새로운 무제한 연구로 확대하지 않는다.
- 일일마감이 자정으로 확인되면 22:30에 신규 장기 작업 시작을 중단하고, 23:00부터 준비된 최고 후보의 QA/업로드/채점을 우선한다. 현재 실험은 중단 없이 artifact를 보존한다. 지연 제출 때문에 개선된 후보를 놓치지 않는다.
- P1 두 기회는 depth-repair winner와 적격 strong-component winner, P2 두 기회는 objective winner와 별개 tree/stack winner, P3 한 기회는 A/B 중 **내부평가가 끝난 하나**에 배분한다. 꼭 전부 소비하는 것은 목적이 아니며 동일 SHA는 제출하지 않는다.
- 공개 점수는 봉인 후보 간 선택·실측 근거로만 쓴다. 첫 제출 반환값으로 다음 후보의 계수/threshold를 역산하지 않는다.

## 8. 낮은 추론 강도 실행자에게 넘길 완료 정의

실행 승인 후 이 문서를 주 계획으로 읽고 과거 NO_GO와 다른 가설인지 확인한다. 코드명을 아직 존재하는 runner처럼 호출하지 말고 새 ID로 구현·synthetic tests·config freeze를 먼저 수행한다.

각 문제는 `source manifest → train/inner/outer receipts → 같은 행 내부테스트 → 독립 metrics QA → frozen fulltrain → saved-model fresh-process replay → answer schema/hash → 승인 범위 업로드/실측 기록`까지 한 작업 단위다. 단순 계획·1seed 실행만으로 완료하지 않는다.

산출물은 독립 문제 폴더의 `01_data`(로컬 전용 입력/manifest), `02_code`, `03_training`, `04_models`, `05_answer`, `06_report`로 분리한다. 노트북은 이 전체 학습 경로의 실행 입구로 만들며 완성모델 추론만을 재학습 검증으로 부르지 않는다. checkpoints/원본/답안/credentials는 Git에 포함하지 않는다. Git 커밋·푸시는 별도 사용자 요청 범위에 따른다.

필수 보고 필드: 후보 ID/hash, source/recipe 계보, base/calibration/full fits 각각의 수, train·replay runtime, 내부 F1/RMSE·동일 baseline 대비 delta, 반복노출 여부, seed/slice 위험, 현재 공식 기준점, 예상 공식점수 산정 여부, 실제 제출 후 metric/points, CSV/업로드 수, 다음 분기.

이번 문서는 [근거 원장](../reports/score_improvement_redesign_20260905_v2/claim-source-ledger.md)과 [남은 불확실성](../reports/score_improvement_redesign_20260905_v2/gap-matrix.md)을 함께 사용한다. 문헌과 과거 성공은 새 후보의 점수 상승 증명이 아니다.
