# v12 exact-family duplicate audit

## 결론

독립 감사에서 base∧peer consensus union은 과거 window-phase/v5–v8과 실질 중복으로 판정해 실행 전 취소했다. 취소 receipt는 `cancelled-candidate-receipt.json`에 별도 보존한다.

남은 `ONE_ROW_TRAILING_EDGE_DILATION_ADD_ONLY`는 미래 endpoint를 쓰는 internal-gap bridge나 `close_gap`과 달리 동일 station-layer-year의 직전 10분 anchor 양성만 현재 행으로 한 칸 연장한다. leading/future edge, erosion, station split, learned row router, threshold grid, 결과 기반 morphology 선택은 없다. 과거 learned RF boundary extension은 음수였지만 literal anchor-bit 1행 규칙은 실행되지 않았으므로, 낮은 prior를 명시한 one-shot falsification만 허용한다.

중복 검색어: `consensus union`, `anchor lag1`, `causal dilation`, `bridge gap`, `morphology`, `deployment_prediction_base deployment_prediction_peer`. 가장 가까운 기존 연구는 endpoint-unanimity 7–48행 internal-hole bridge, 0–6행 gap fill, learned RF boundary extension이었다. 이들은 각각 미래 양끝, hole fill, learned router를 사용해 literal trailing-edge anchor-bit rule과 exact topology가 다르다.
