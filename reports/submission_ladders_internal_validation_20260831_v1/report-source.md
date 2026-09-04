# 제출 후보 내부 검증 — 2026-08-31

> 공식 결과 갱신: 내부 strict pass였던 `P2_3`은 Public RMSE `0.430800`,
> 점수 `27.927863`으로 현 챔피언보다 `-0.007601점` 악화했다. 고보상·고위험
> 탐침이었던 `P3_2`도 `-0.021529점` 악화했다. 두 문제 모두 현 챔피언을
> 유지한다. 원장은 [p2_3_official_result.md](p2_3_official_result.md)와
> [p3_2_official_result.md](p3_2_official_result.md)에 기록했다.

## 결론

모든 제출 후보는 공식 업로드 전에 내부 historical test를 반드시 거친다. 통계적 내부 판정과 제한된 제출기회의 우선순위는 분리한다. `INTERNAL_PASS_STRICT`가 기본 공식 탐침 후보지만, 안정성은 낮아도 조건부 점수 보상이 큰 후보는 위험을 명시한 `HIGH_REWARD_HIGH_RISK_OFFICIAL_PROBE`로 별도 관리한다. 정확한 배포 챔피언 OOF가 없는 P1/P2는 프록시임을 숨기지 않는다.

| 문제 | 후보 | 내부 변화 | 판정 |
|---|---|---:|---|
| P1 | `P1_1_PEER_HIGHCONF_UNION` | `-0.000138754` | `INTERNAL_NO_GO` |
| P1 | `P1_2_PEER_FULL_UNION` | `-0.005100287` | `INTERNAL_NO_GO` |
| P1 | `P1_3_PEER_STANDALONE` | `-0.024065800` | `INTERNAL_NO_GO` |
| P2 | `P2_1_BIN17_DROP_LAYER2` | `+0.000042057` | `INTERNAL_NO_GO` |
| P2 | `P2_2_BIN17_DROP_LAYER3` | `+0.000058526` | `INTERNAL_NO_GO` |
| P2 | `P2_3_BIN17_DROP_LAYER4` | `-0.000030854` | `INTERNAL_PASS_STRICT` |
| P3 | `P3_1_KMA_A18_0425_A24_0600` | `+0.001576137` | `INTERNAL_NO_GO` |
| P3 | `P3_2_KMA_A18_0200_A24_0425` | `-0.001767848` | `INTERNAL_SIGNAL_ONLY_UNSTABLE` |
| P3 | `P3_3_KMA_A18_0200_A24_0600` | `-0.000188143` | `INTERNAL_SIGNAL_ONLY_UNSTABLE` |

P3 제출 우선순위 오버레이는 [p3-official-probe-priority.md](p3-official-probe-priority.md)에 기록했다. `P3_2`의 1:1 수송 조건부 중앙값은 `+0.028057점`으로, 내부 안정성 판정을 보존하면서도 P3 공식 탐침 1순위로 둔다.

## 고정 운영 규칙

1. 후보 레시피와 비교 기준을 점수 전에 고정한다.
2. 시간 순서가 보존된 내부 OOF/forward test에서 현 챔피언 또는 가장 가까운 프록시와 비교한다.
3. pooled 지표뿐 아니라 fold/기간 최악값과 paired bootstrap을 확인한다.
4. 내부 `NO_GO`는 제출하지 않는다. 불안정한 `SIGNAL_ONLY`는 자동 탈락시키지 않고, 점수 환산 보상과 정보가치가 충분할 때만 위험 표지를 붙여 별도 공식 탐침 후보로 둔다.
5. `PASS_STRICT` 또는 사전 기록된 `HIGH_REWARD_HIGH_RISK_OFFICIAL_PROBE` 중 우선순위가 가장 높은 한 후보를 Public에 올려 내부-공식 수송 오차를 기록한다.
6. 공식 점수를 본 뒤 같은 날 같은 후보군을 재튜닝하지 않는다. 다음 묶음을 다시 공동 동결한다.

## 비교 기준의 한계

- P1: Historical raw e150 OOF is the closest available comparator. The deployed official champion additionally contains two official-only GI rows, so this is not an exact champion OOF reconstruction.
- P2: The exact full-fit official bin17 correction has no historical OOF lineage. The closest sealed three-way cross-fit rank-1 correction is restricted to bin17 and label-blind endpoint/PAVA projection is reapplied.
- P3: The official uniform alpha=.425 is evaluated on the frozen KMA OOF correction axis. Public and local transport have previously disagreed by station, so local pass is necessary evidence but not a guarantee of Public improvement.

## 데이터 품질 QA

- P1: OOF 키 1:1, truth mismatch 0행.
- P2: prediction/truth join multiplier 1.0, missing 0행, KST bin17 고정.
- P3: 182개 사례마다 6개 lead 완전, pair-key 중복 0행.
- 공식 hidden label 읽기 0, 새 submission CSV 0, upload 0.
