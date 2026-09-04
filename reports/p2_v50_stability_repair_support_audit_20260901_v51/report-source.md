# P2 v51 안정성 보정 전향 지지 감사

## 결론

`NO_GO_SUPPORT_AUDIT_ZERO_FIT`입니다. v50 OOF를 이용해 사전에 고정하려던 leakage-safe LOFO layerwise shrink는 held-out에서 pooled RMSE를 `+0.001478482 C` 악화시켰고, fold×layer non-harm은 `6/9`, 최대 악화는 `+0.030862566 C`, canonical transport 예상점수는 `-0.140233`점이었습니다. 네 개 승격 조건을 모두 실패했으므로 v51 모델 학습·config·runner·negative fingerprint·READY preflight·attempt lock을 만들지 않고 0 fit에서 fail-closed했습니다.

## 질문과 판단 기준

질문은 v50의 평균 성능을 유지하면서 fold×layer 안정성을 전향적으로 복구할 수 있는 고정 메커니즘이 있는가입니다. 다음 네 조건을 동시에 만족해야 실행 후보로 볼 수 있도록 고정했습니다.

- pooled `delta RMSE < 0`
- fold×layer non-harm `>= 8/9`
- 최대 fold×layer `delta RMSE <= +0.003 C`
- canonical transport 예상점수 `>= +0.01`

결과에 맞춘 gate 완화나 탐색 범위 확장은 허용하지 않았습니다.

## 고정 메커니즘

후보는 `leakage_safe_leave_one_fold_out_layerwise_minimax_shrink`입니다. 각 held-out historical fold와 layer에 대해 v50 residual correction에 곱할 `alpha`를 `{0, 0.25, 0.5, 0.75, 1}`에서 고르되, held-out label은 사용하지 않고 나머지 두 fold의 최악 `delta RMSE`가 가장 작은 값을 선택하며 동률이면 작은 `alpha`를 선택했습니다.

선택된 `alpha`는 다음과 같습니다.

| held-out fold | layer 2 | layer 3 | layer 4 |
|---|---:|---:|---:|
| 2024 Sep-Oct | 0 | 0 | 0 |
| 2025 Jul-Aug | 1 | 1 | 0 |
| 2025 Nov-Dec | 0 | 0 | 1 |

## 전향 결과

| 항목 | 결과 | 조건 | 판정 |
|---|---:|---:|---|
| pooled delta RMSE | `+0.001478482 C` | `< 0` | FAIL |
| non-harm fold×layer | `6/9` | `>= 8/9` | FAIL |
| 최대 fold×layer 악화 | `+0.030862566 C` | `<= +0.003 C` | FAIL |
| canonical transport 예상점수 | `-0.140233` | `>= +0.01` | FAIL |

held-out fold별 delta RMSE는 2024 Sep-Oct `0`, 2025 Jul-Aug `+0.009135208 C`, 2025 Nov-Dec `+0.005946243 C`였습니다. 즉 calibration fold에서 안전한 shrink를 고르는 규칙이 새로운 fold로 운반되지 않았습니다.

## 고정 구조 스크린

추가 학습 없이 완전한 profile에서 layer-2 correction을 layer 2-4에 고정 broadcast하고 불완전 profile만 own-layer correction으로 fallback하는 label-free 구조도 진단했습니다. full strength는 pooled `-0.041458257 C`, non-harm `8/9`, transport `+0.398517`점이었지만 최대 cell 악화가 v50과 같은 `+0.030862566 C`라 safety gate를 실패했습니다.

고정 shrink 경계에서는 `alpha=0.10/0.15`가 최대 악화를 `+0.003 C` 아래로 낮췄지만 transport 예상점수가 각각 `-0.06381/-0.03536`점이었습니다. `alpha=0.25`에서 transport가 `+0.02058`점으로 양수가 되지만 최대 악화가 `+0.00473318 C`로 gate를 넘었습니다. 따라서 감사한 경계 안에는 safety와 정보가치를 동시에 만족하는 교집합이 없습니다. 이 스크린은 진단일 뿐 새 후보 선택이나 실행에 사용하지 않았습니다.

## 데이터·계보·실행 봉인

- 유일한 원천은 주최측 배포 `observations.csv`이며 v50에 기록된 SHA-256은 `cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a`입니다.
- v50 result SHA-256은 `517aac76ded7c7ae67e6189d73ca94d174842e17546ee81d22801d95f732617b`, independent QA SHA-256은 `e2b851b76b1dc27593cde2244a5d9480bd562bbef3742c5c830c6e66a7a65c26`입니다.
- 이번 감사에서 observations `789,408`행과 historical scoring/OOF `69,850`행만 사용했습니다.
- 공식 test index, sample, baseline, score, query support, hidden truth는 모두 `0`행 접근입니다.
- 외부 관측·재분석·예보, pretrained weight, submission CSV, upload는 모두 `0`입니다.
- model fit, attempt lock, READY preflight, config, runner는 모두 `0`입니다.

## 최종 판단

v50의 평균 개선은 실재하지만, 현재 고정 가능한 안정성 보정 규칙은 held-out support를 얻지 못했습니다. 따라서 v51을 억지로 실행하는 것은 안전 gate를 완화하거나 결과 기반 탐색을 늘리는 행위가 됩니다. 이 감사로 해당 경로를 닫고, v49/v50은 재실행하거나 patch-in-place하지 않습니다.
