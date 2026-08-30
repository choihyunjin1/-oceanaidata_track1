# Historical re-audit status taxonomy

한 레코드는 `primary_status` 하나와 0개 이상의 `status_tags`를 갖는다. 태그는 서로 다른 평가면의 사실을 보존한다. 예를 들어 trial18은 exact confirmation상 `CLOSED_EXACT`이면서 선택면에서는 `CHECKPOINT_PEAK`, 수송 관점에서는 `PROXY_EXPOSED`다.

| status | operational meaning |
|---|---|
| `CLOSED_EXACT` | The exact tested recipe, split, feature set, postprocess, and gate is closed. The broader model class is not closed. |
| `INVALID_TECHNICAL` | No scientific conclusion is admissible because execution, schema, dependency, or observability failed. |
| `DISCOVERY_ONLY` | Useful for mechanism discovery or lineage comparison, but not independently confirmed or submission-ready. |
| `OLD_GATE_REJECTED` | A positive or inconclusive signal was rejected by an old unsupported hard gate; only a frozen confirmation may reopen it. |
| `CHECKPOINT_PEAK` | A best intermediate checkpoint or selected point exists; it is preserved as a candidate, not treated as final-epoch evidence. |
| `INFORMATION_POSITIVE` | The result yielded verified directional information or a positive official/local mechanism signal; this is not automatically deployment readiness. |
| `PROXY_EXPOSED` | Selection/proxy evidence changed sign or materially weakened on a later sealed or official surface. |

## Counting rule

48 historical families, 35 canonical groups, 20 key cases, and 4 workflow exceptions are overlapping grains. The exhaustive historical-family denominator is 48; never report 107 as 107 unique experiments.
