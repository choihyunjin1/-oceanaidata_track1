# P1 v21 결론

`P1_1_CAUSAL_SCAR_PU_LINEAR_ADDONLY`는 **NO_GO_SAFE_ABSTENTION**이다. 두 chronological prefix 모두 inner precision-LCB 조건을 만족하는 threshold가 없어 `Infinity`를 선택했고, Q3/Q4에 추가한 행은 0이다. 따라서 pooled `ΔF1=0`, raw expected `0점`, transport penalty 반영 후 `-0.121682092점`이며 calibrated `+0.01점` 기준을 통과하지 못했다.

## 실행 및 독립 QA

- exactly-once 2 fits, runtime `33.774799s`
- Q2→Q3 propensity `0.05`(lower clip), inner/outer additions `0/0`
- Q2+Q3→Q4 propensity `0.2564582473`, inner/outer additions `0/0`
- prediction 421,032행은 incumbent anchor와 bit-exact하며 additions/removals 모두 0
- bootstrap `P(ΔF1>0)=0`, CI90 `[0,0]`
- official/hidden/CSV/upload 모두 0
- py_compile PASS, Ruff PASS, focused pytest 5/5 PASS

한 fixed L-BFGS fit이 500 iteration cap에 도달했다. 모델·threshold·fit budget은 사전봉인 그대로 유지했고 retry는 하지 않았다. 출력은 finite이고 최종 후보가 exact no-op이므로 이 경고를 성능 승격 근거로 사용하지 않는다.

## 해석

SCAR 보정은 v16의 일부 row-level 신호를 안정적인 sparse addition으로 바꾸지 못했다. 특히 첫 prefix의 propensity가 lower clip까지 내려갔고, inner precision uncertainty를 통과한 추가행이 없었다. 이는 SCAR 가정이 확인됐다는 뜻도, 모든 PU 학습을 닫는다는 뜻도 아니다. 다만 이 정확한 165-feature selection-logistic + inner propensity/threshold recipe는 종료한다.
