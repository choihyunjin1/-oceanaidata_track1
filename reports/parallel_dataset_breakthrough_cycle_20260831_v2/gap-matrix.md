# Gap matrix

| 문제 | 현재 champion | 실제 격차/반증 | 새 정보 또는 구조 | 중복 판정 | 지금 실행 | 재개 조건 |
|---|---|---|---|---|---|---|
| P1 | e150 + G/I spike 보정, Public F1 0.833548 | 2026-08-29 문제 최고와 3.096057점; local offset/drift recall 약 0.65; I1/I5/S2와 June 취약 | v6 clean-state/cross-layer geometry + CAPA point/collective segment likelihood | 표현 고중복, decoder materially distinct; v6 fit 0 | Stage-0 PASS, fit HOLD | 새 executable contract + independent QA; exposed-fold 결과는 RESEARCH_ONLY |
| P1 차선 | 동일 | external point residual -0.063301, thermocline FP 위험 | ERA5 forcing-conditioned peer reliability | direct external temperature residual과 distinct | manifest 설계만 | timestamp/license/assimilation/target exclusion gate + frozen with/without ablation |
| P2 | bin17, Public RMSE 0.430194°C | fixed rank-5 DINEOF +0.181752°C; CMFPCA +0.022734°C; BayOTIDE active 0 | mask-matched joint T/S DINEOF residual rank 1/2 | 기존 exact DINEOF/CMFPCA/PLS/OAS/copula와 distinct | Stage-0 only | untouched same-season 61-day label block 또는 blind evaluator |
| P3 | KMA α=.425, Public RMSE 0.575233m | perfect future local wind +0.001340m; fresh 1 block +0.022617m | as-of GFS/GEFS regional-fetch forcing-error residual | local-wind MOS, ERA5 Hs², past-only forcing과 distinct | STOP_NO_DATA / TRAIN_0 | signed case→issue-time/좌표/cycle manifest + as-of archive + fresh block ≥8 |

## Kill rules

- P1: pooled row micro-F1 delta ≤0, add-only proposal precision ≤same-surface incumbent F1/2, thermocline FP가 pooled harm을 만들거나 기존 decoder와 proposal fingerprint가 동일하면 종료한다.
- P2: simultaneous 61-day mask support, active-pattern support, design rank/condition check 중 하나라도 실패하면 0-fit 종료한다.
- P3: issue-time manifest, publication cutoff, archived operational forecast, independent storm block 중 하나라도 없으면 다운로드와 학습을 시작하지 않는다.
