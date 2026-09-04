# P1 v26 ECDF action-fraction preseal audit

## 결론

v16/v24의 실패는 GCE score 자체가 완전히 무정보라서라기보다, prefix에서 정한 **절대 확률 threshold가 outer block으로 수송되지 않은 현상**을 포함한다. v24 Q3은 inner threshold `0.2657745297`가 outer anchor-negative의 `3.1186%`를 작동시켜 전체 changed cap `0.5%`를 크게 넘었고, Q4 threshold `0.0111861316`는 반대로 `0.0261%`만 작동시켰다. v24 Q4 score의 p99는 Q3 p99의 1/1000 미만이었다. v16도 고정 `0.95`에서 Q3 91행, Q4 0행으로 붕괴했다.

따라서 inner label로 **최소 유효 action fraction**만 고르고 outer에서는 label 없이 같은 ECDF top fraction을 적용하는 v26은 절대 scale drift에 대한 구조적으로 맞는 수리다. 이는 precision의 전이를 보장하지 않으므로 Q3/Q4 nonnegative, bootstrap, slice, raw `>=0.015383691373120248`, calibrated `>=0.01` gate를 그대로 유지한다.

## 봉인 규칙

1. 각 prefix의 첫 75%로 exact v16 GCE(q `0.7`, L2 `0.001`)를 새로 1회 fit한다.
2. 나머지 25%의 incumbent-negative score를 내림차순으로 정렬한다. 동률은 `SHA256(station|layer|UTC_ns)` first64 오름차순이다.
3. 전체 calibration row의 `0.5%` 이하인 k 중 add-only ΔF1 `>0`, central precision `> anchor F1/2`를 처음 만족하는 최소 k를 선택한다. 없으면 fraction 0이다.
4. outer incumbent-negative score는 label 없이 같은 방식으로 정렬하고 `floor(k_inner * n_outer_negative / n_inner_negative)`행만 추가한다. 최소 1행 override나 사후 trimming은 없다.
5. Q2→Q3 및 Q2+Q3→Q4의 새 fit은 정확히 2회다. outer 결과로 fraction·model·cap을 바꾸지 않는다.

## 소급 사용 금지

v16/v24 sealed probability는 scale/quantile drift 진단에만 읽었다. v26 selector를 기존 probability와 historical label에 소급 적용하거나, 어떤 fraction이 과거 outer score를 개선했는지 계산하지 않았다. preflight의 historical truth, official, hidden, CSV, upload, lock, fit은 모두 0이다.

## 한계

- ECDF 수송은 단조 scale 변화에는 불변이지만 score ranking 자체의 concept drift에는 불변이 아니다.
- 최소 k 규칙은 매우 작은 action count를 만들 수 있으며 outer에서 floor 결과가 0일 수 있다. 이를 임의로 1로 올리지 않는다.
- v24 Q3의 대규모 false positive는 scale 문제 외에 ranking failure도 시사한다. 그래서 outer block과 supported slice gate를 통과하지 못하면 즉시 NO_PASS로 닫아야 한다.
