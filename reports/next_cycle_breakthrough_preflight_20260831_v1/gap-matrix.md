# 다음 실행을 여는 증거 gap

| 문제 | 현재 남은 유효 자산 | 닫힌 exact 축 | 핵심 gap | 재개 trigger | 현재 action |
|---|---|---|---|---|---|
| P1 | best checkpoints, anchor-union, event/range 평가 코드 | exposed Q2–Q4 재선택, old-gate replay | 미사용 chronological label 0행 | checkpoint rule까지 선고정한 새 label block | `NO_NEW_FIT` |
| P2 | bin17-only Public champion, low-rank recipe | bin18 확장, same-Public bin/rank 선택 | untouched same-season surface | Public과 독립된 same-season block 또는 사전 고정 factor | `NO_NEW_FIT_OR_BIN_EXPANSION` |
| P3 | uniform KMA 0.425 champion, ERA5 manifest/feature infra | alpha micro-sweep, station removal, exact ERA5 Hs² blend | multiple fresh storm episodes와 새 forcing target | 최소 3개 episode-disjoint block + frozen residual rule | `NO_NEW_KMA_OR_ERA5_MICROTUNE` |

## 공통 gap

| gate | 요구 | 현재 상태 | 판정 |
|---|---|---|---|
| 새 정보면 | 모델 선택에 쓰이지 않은 label/score/data source | P1/P2/P3 모두 없음 | 차단 |
| 비중복 mechanism | 기존 exact recipe와 다른 target/representation | 이름만 다른 후속 후보가 다수 | 사전 hash 대조 필수 |
| 시간 독립성 | chronological/episode-disjoint validation | 반복 노출된 historical fold | blocked/rolling-origin 새 block 필요 |
| 이상치 계약 | label-blind immutable sensor QC | result-based extreme removal 미고정 | 자동 제거 금지 |
| 운송 확인 | Public와 다른 population에서 방향 유지 | Private/새 local surface 없음 | research-only |
