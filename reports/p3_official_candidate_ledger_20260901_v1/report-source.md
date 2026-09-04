# P3 official candidate ledger — 2026-09-01

## 결론

현재 공식 champion은 uniform KMA `alpha=0.425`, Public RMSE `0.575233m`, `24.203599점`이다. v19 wave-power 후보는 2026-08-31 Public RMSE `0.589840m`, `23.971758점`으로 champion보다 `-0.231841점` 낮았으므로 v19 계보를 폐쇄한다.

기존 unsubmitted ready pack 중 다시 제출할 가치가 있는 exact candidate는 없다. Station S/I ablation, lead-factor, ERA5 Hs², target-fit physical router, v19은 이미 공식 실패했거나 같은 계보의 내부 NO_GO/불안정 후보다. 새 연구는 uniform `.425`를 bit-exact default로 보존하는 직교 family에서만 진행한다.

## 접근 경계

저장된 공식 aggregate 영수증만 read-only로 감사했다. 공식 test/sample/submission CSV와 hidden truth는 읽지 않았고 materialization/upload는 모두 0이다.
