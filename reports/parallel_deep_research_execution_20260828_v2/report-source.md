# P1·P2·P3 병렬 딥리서치 및 실제 실행 결론

**판정일:** 2026-08-28 (KST)
**기준 커밋:** `7fa140996beb451023c63b0e1fbc56039fbb6d19` (dirty worktree 보존)
**범위:** P1 품질관리, P2 수온 복원, P3 파고 예측의 새 구조를 각각 1회 사전 고정 실행하고, 기존 후보까지 새 승격기준으로 재평가
**금지 준수:** hidden answer/mirror 복구, 결과 기반 재탐색, 공식 업로드, git commit/push 미수행

## 결론부터

하드 기준이었던 “공식 점수 +3점 예상”은 폐기했다. 앞으로는 **같은 누출 방지 split에서 반복 가능한 개선이 확인되면 개선으로 인정**한다. 다만 과거 fold가 이미 여러 연구에 노출됐으므로, 로컬 통과는 최종 우승 모델의 증명이 아니라 공식 검증 후보 승격을 뜻한다.

이번에 병렬로 새 구조 세 개를 실제 실행한 결과, **P1과 P3은 NO_GO, P2의 새 CMFPCA 구조도 NO_GO**였다. 세 모델 모두 새 기준에서도 개선이 아니므로 제출 파일을 만들지 않았다. 이것은 각 문제의 이론적 최대치에 도달했다는 뜻이 아니라, 이번에 검증한 구조 세 개를 후보 공간에서 제거했다는 뜻이다.

반면 기존 **P2 seasonal OAS α40**은 파일 QA가 완전히 통과했고, 공식 α10→α20 계보가 개선된 점과 α40 공식 벡터 기하가 추가 개선을 예측한다. 로컬 OOF는 반대로 악화하므로 champion 승격은 아니지만, **정확히 한 번 공식 점수 곡률을 확인할 정보가치가 있는 `OFFICIAL_PROBE_READY`**로 판정한다. 명시적 업로드 승인을 받기 전에는 제출하지 않는다.

## 갱신한 승격기준

1. **로컬 개선:** 사전 고정된 blocked/rolling split에서 incumbent 대비 방향이 좋아야 한다. 절대 +3점 환산은 요구하지 않는다.
2. **재현성:** fold 방향, paired bootstrap 또는 동등한 불확실성, seed 안정성을 함께 본다. 단일 pooled 숫자만으로 승격하지 않는다.
3. **안전성:** leakage, target-late opening, artifact seal, no-result-based-retry를 독립 QA한다.
4. **공식 probe 예외:** 로컬 surrogate와 공식 점수의 transport가 반복적으로 어긋났고, 이미 관측된 공식 계보가 반대 방향을 지지하면 정보가치 probe를 허용한다. 이는 champion 승격과 구분한다.
5. **정지 규칙:** probe 결과를 기록하기 전 동일 축의 다음 강도(예: α60/α80)를 만들거나 제출하지 않는다.

리더보드 반복 적응은 숨은 test에 대한 과적합 위험을 만든다. Ladder는 점수 공개를 제한해 적응적 과적합을 줄이는 문제를 다뤘고, Cawley와 Talbot은 모델 선택 기준의 분산 자체가 과적합을 유발한다고 지적했다. 따라서 “작은 개선을 인정”하되, 제출 횟수는 명시적인 가설 검증으로만 소비한다 ([Blum & Hardt, ICML 2015](https://proceedings.mlr.press/v37/blum15.html); [Cawley & Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)).

## 문제별 실제 실행 결과

| 문제 | 사전 고정 구조 | 핵심 결과 | 판정 | 공식 CSV |
|---|---|---:|---|---|
| P1 | frozen direct interval proposal + hard-negative event verifier | 123 proposals; train utility-positive 2, calibration 0; positive cell share 100% | `NO_GO_SUPPORT_PREFLIGHT` | 생성 안 함 |
| P2 | depth-registered conditional multivariate FPCA, OAS20과 같은 anchor/blend | RMSE 0.943540 vs 0.920806; Δ +0.022734°C; CI90 +0.016453~+0.028952 | `FAIL_GATE_NO_CANDIDATE` | 생성 안 함 |
| P3 | observed-weather TSMixer residual, 12/18/24h만 20% blend | RMSE 0.783355 vs 0.779949; Δ +0.003406m; 0/3 folds 개선 | `TERMINAL_NO_GO` | 생성 안 함 |

표의 양수 Δ는 incumbent 대비 악화를 뜻한다. 모든 수치는 봉인된 aggregate artifact에서 가져왔고, 원본 관측값을 보고서에 노출하지 않았다.

### P1 — verifier가 아니라 proposal support가 병목

첫 실행은 pandas datetime 정수가 µs인데 ns cadence와 직접 비교한 구현 결함 때문에 과학적 결과가 아니었다. QA가 모델 fit 0회, qualification truth 0회인 상태에서 이를 잡았고, 새 실험 ID v2에서 시간 비교만 `Timedelta` 기반으로 고쳤다. v1/v2의 의미 설정은 ID·출력 경로 외 동일함을 pre-execution guard로 봉인했다.

v2에서는 최소 19행 event proposal 123개가 생성됐다. train 45개 중 utility-positive는 2개(기준 10), calibration 2개 중 positive는 0개(기준 4)였고 train positive가 단일 station×layer에 100% 집중됐다(허용 70%). hard-negative/positive 21.5는 통과했지만, 지원도 부족 때문에 모델을 fit하지 않고 qualification truth도 열지 않았다. 따라서 현재 frozen direct generator와 verifier pairing은 제출 가치가 없다.

이 방향 자체는 문헌과 합치한다. BREM은 proposal의 boundary/region quality를 별도로 추정해 localization confidence를 보정하고, Rank-DETR은 quality-aware ranking으로 낮은 품질·negative 예측을 억제한다. 그러나 우리 데이터에서는 verifier를 학습할 positive event support가 먼저 부족했다 ([Hu et al., BREM, 2022](https://arxiv.org/abs/2204.11695); [Pu et al., Rank-DETR, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html)). 시계열 비교의 비교환성 때문에 일반 conformal 보장을 그대로 쓰지 않고 blocked calibration을 유지한 것도 SPCI의 문제 설정과 일치한다 ([Xu & Xie, ICML 2023](https://proceedings.mlr.press/v202/xu23r.html)).

증거: `artifacts/p1_frozen_direct_event_verifier_blocked_20260828_v2/aggregate_metrics.json`, `independent_qa.json`, `qa_manifest.json`. QA manifest SHA-256은 `20e00f3ce101276a8b1908f295d972720581aacdd4fcd3a07be84cb7db6590ea`이다.

### P2 — 깊이 정합 아이디어는 타당했지만 rank-4 구조가 악화

물리 깊이에 등록한 온도·염분 곡선에서 conditional multivariate FPCA latent score를 복원하는 구조를 구현했다. 부분 관측된 다변량 수직 프로파일을 조건부 FPCA로 복원하는 접근은 최신 해양 프로파일 연구와 직접 대응한다 ([Fonvieille et al., 2026](https://arxiv.org/abs/2608.05376)). 결측 시계열에서 low-rank 구조와 temporal modeling을 결합하는 ImputeFormer도 저차원 구조 활용의 근거를 제공한다 ([Nie et al., KDD 2024 공식 구현](https://github.com/tongnie/ImputeFormer)).

그러나 frozen OOF에서 OAS20 RMSE 0.920806 대비 CMFPCA20은 0.943540으로 0.022734°C 악화했다. 5,000회 KST-day paired bootstrap CI90은 +0.016453~+0.028952°C이고 개선확률은 0이다. 4개 fold 중 2025 Nov–Dec만 0.020396°C 개선했고, 나머지 3개 fold와 layer 2/3/4가 모두 악화했다. rank cap 4가 fold별 train variance의 약 69~72%만 설명한 것은 underfit 진단이지만, 결과를 본 뒤 rank를 바꾸지 않았다.

증거: `artifacts/p2_depth_registered_cmfpca_v1_20260828/result.json`, `independent_qa.json`, `manifest.json`. focused pytest 10개와 Ruff가 통과했다.

### P3 — TSMixer residual은 장기 lead를 모두 악화

48시간 observed history에서 파고·주기·풍향·풍속·기온·습도·mask를 섞는 TSMixer residual을 학습하고, incumbent의 3/6/9시간 예측은 bit-exact로 보존하며 12/18/24시간만 TSMixer와 20% blend했다. TSMixer는 time-mixing과 feature-mixing MLP로 auxiliary 정보를 통합하는 구조다 ([Google Research, TSMixer/TMLR 2023](https://research.google/pubs/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/)). 공식 저장소는 기본 구현은 공개하지만 auxiliary-feature 확장 구현은 미제공이라고 명시하므로, 우리 확장은 논문 개념을 자체 구현한 것이며 공식 코드 재현이라고 주장하지 않는다 ([공식 저장소](https://github.com/google-research/google-research/tree/master/tsmixer)).

181 cases/1,086 rows에서 candidate RMSE는 0.783355, incumbent는 0.779949로 0.003406m 악화했다. 3개 fold가 모두 악화했고 12/18/24시간 Δ는 각각 +0.005195, +0.007583, +0.005492m였다. bootstrap CI90은 -0.000784~+0.007798m, 개선확률은 8.92%다. 3개 station 중 G-ORS만 0.001987m 좋아졌지만 S-ORS가 0.010707m 악화해 router 없는 공통 blend로는 가치가 없다.

PatchTST의 patching·channel-independence는 강한 대안 계열이지만, 이번 실험은 이미 검증된 incumbent를 보존하면서 auxiliary feature가 있는 residual만 검증하도록 범위를 좁혔다 ([Nie et al., ICLR 2023 공식 구현](https://github.com/yuqinie98/PatchTST)).

증거: `artifacts/p3_tsmixer_observed_residual_20260828_v1/result.json`, `qa.json`, `manifest.json`. result SHA-256은 `175f3a9aa1469d032b1491630fb2dc1ce55f148d5c84e0ee71d25cd7e853489c`; root 재계산·focused pytest 4개·Ruff가 모두 통과했다.

## 남은 유효 결과물 — P2 OAS α40 공식 probe

**정확한 파일**
`C:\Users\cedis\Downloads\해양 해커톤 제출용\20260828_P2_SEASONAL_OAS_TS40_PROJECTED_READY\P2_submission.csv`

**SHA-256**
`6e28ddb8d78c0969e5104d7efbe28e1762f51e80d759fceb86cdef52baa29b96`

파일 QA는 26,061행, 열 `station,layer,time,temp`, key 순서·고유성, finite/range, canonical byte identity, PAVA idempotence를 모두 통과했다. temp 범위는 16.302557~29.081289°C이며 layer 2/3/4 행 수는 8,713/8,712/8,636이다. answer 파일은 열지 않았다.

공식 동일 계보에서 α10은 RMSE 0.507628·점수 26.963865, α20은 RMSE 0.483661·점수 27.264587로 0.300722점 개선됐다. 정확한 α10/α20/α40 예측 벡터 기하로 추정한 α40 공식 RMSE 범위는 0.431042~0.465548°C이며, 보수적 상한도 α20보다 예상 점수 0.227268점 높다. 이 추정은 공식 scorer가 배포된 26,061행 통합 RMSE와 같고 기록 RMSE가 6자리 반올림이라는 가정에 의존한다.

반증도 강하다. exposed OOF의 unprojected α40은 α20보다 0.016402°C 나쁘고, KST-day bootstrap CI90도 +0.007860~+0.024632°C다. 따라서 **최종 champion으로 부르지 않는다.** 공식 α10→α20 추세와 local saturation의 충돌을 구분할 단 한 번의 probe로만 추천한다. 결과가 나오기 전 α60/α80은 금지한다.

## 의사결정·다음 단계

- **지금 제출할 가치가 있는 파일:** P2 OAS α40 한 개만 `OFFICIAL_PROBE_READY`.
- **제출하면 안 되는 것:** P1 verifier, P2 CMFPCA, P3 TSMixer residual. 세 구조 모두 후보 CSV가 없다.
- **업로드 상태:** 0건. 정확한 P2 파일과 SHA에 대한 사용자 명시 승인 후에만 업로드한다.
- **probe 해석:** α40 공식 점수가 α20을 개선하면 official-axis의 추가 탐색을 새 실험으로 설계한다. 악화하면 α 축을 종료하고 다른 구조로 이동한다.
- **연구 한계:** 로컬 fold는 이미 여러 실험에 노출됐고 완전히 fresh하지 않다. 공식 점수도 제한된 public surface이므로 장기적으로는 새로운 temporal holdout을 한 번만 여는 체계가 필요하다.

## 독립 QA 요약

- P1/P2 focused pytest 합계 11개 PASS, 관련 Ruff PASS.
- P3 focused pytest 4개 PASS, 관련 Ruff PASS.
- P3 평가 parquet에서 root가 RMSE·fold Δ를 직접 재계산해 result와 일치.
- P2 α40 answer-free QA를 immutable `P2_DATA_DIR`로 재실행해 `PASS_OFFICIAL_PROBE_ELIGIBLE_PENDING_EXPLICIT_UPLOAD_APPROVAL` 확인.
- `git commit`, `push`, 공식 upload는 수행하지 않음.

## Claim-to-source ledger

| 주장 | 1차 출처 | 저자/발행처·연도 | 확인 내용 | 한계/충돌 |
|---|---|---|---|---|
| event proposal의 quality를 별도 추정할 수 있다 | [BREM](https://arxiv.org/abs/2204.11695) | Hu et al., ACM MM 2022 | boundary·region context로 proposal quality/misalignment 추정 | P1은 학습 support가 먼저 부족 |
| quality-aware ranking이 low-quality/negative를 억제한다 | [Rank-DETR](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html) | Pu et al., NeurIPS 2023 | quality-aware classification/ranking | 시계열 event 직접 검증은 아님 |
| 비정상 시계열에 일반 conformal 가정이 깨진다 | [SPCI](https://proceedings.mlr.press/v202/xu23r.html) | Xu & Xie, ICML 2023 | sequential predictive intervals for nonexchangeable series | P1은 support preflight에서 중단 |
| 부분 관측 T/S 프로파일을 조건부 다변량 FPCA로 복원할 수 있다 | [Conditional multivariate FPCA](https://arxiv.org/abs/2608.05376) | Fonvieille et al., 2026 preprint | physical depth function·conditional score reconstruction | 이번 rank-4 구현은 OAS보다 악화 |
| low-rank + temporal modeling은 결측 복원의 유력 구조다 | [ImputeFormer 공식 저장소](https://github.com/tongnie/ImputeFormer) | Nie et al., KDD 2024 | low-rank transformer implementation | P2의 작은 표본·depth mapping과 조건이 다름 |
| TSMixer는 time/feature mixing으로 auxiliary 정보를 활용한다 | [TSMixer](https://research.google/pubs/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/) | Google Research, TMLR 2023 | all-MLP time-series architecture | 공식 저장소의 auxiliary 확장 코드는 미제공 |
| patching·channel independence는 장기예보 대안이다 | [PatchTST 공식 저장소](https://github.com/yuqinie98/PatchTST) | Nie et al., ICLR 2023 | patching, channel independence, self-supervised pretraining | 이번 P3는 TSMixer residual만 검증 |
| adaptive leaderboard 사용은 과적합 위험이 있다 | [Ladder](https://proceedings.mlr.press/v37/blum15.html) | Blum & Hardt, ICML 2015 | leaderboard feedback와 adaptive overfitting | 실제 대회 공개 방식과 동일하다고 단정하지 않음 |
| 모델 선택 기준의 분산도 overfitting 원인이다 | [Selection over-fitting](https://www.jmlr.org/papers/v11/cawley10a.html) | Cawley & Talbot, JMLR 2010 | model-selection criterion overfitting | 공식 점수를 완전히 대체하는 로컬 기준은 아님 |

모든 웹 출처는 2026-08-28에 확인했다. 실험 수치의 1차 출처는 저장소의 봉인된 JSON/Parquet/QA artifact이며, 이 보고서는 해당 aggregate 값만 인용한다.
