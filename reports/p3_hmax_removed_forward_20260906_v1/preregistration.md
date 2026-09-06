# hmax 완전 제거 — numeric와 분리한 단일변경

2026-09-06 numeric 결과 열람 전 고정. 기준은 새 `p3_numeric_lead_forward_gpu_20260906_v2`의 categorical CPU single+GPU multi 완성정책이며, 그5fold exacthash/키/source/recipe가일치하는 baseline만 재사용한다. numeric lead는적용하지않는다.

가설은 raw hmax 및 hmax_hs_ratio의 모든 current/lag/window통계/valid 파생특징을 base single과multi에서 제거하고, router의 hmax_current도 제거하는 정보집합변경하나다. hmax/hs비율이일정해보인다는관찰은 제거안전성증거로쓰지않는다. `summarize_context` raw hmax를서로다른값/NaN으로교란했을때 남은feature와router입력에변화가없는지합성검증하고,기존targets/current_hs/weights/keys는동일하게유지한다.

동일v5 5fold/24360anchors/17267validationanchors/103602rows/78h+episode/earlier-OOF순서를고정한다. singleCPU2, multiGPU0 Plain; params/seed/iterations/shrink/router절차동일. 새후보single5+multi5=10backbone+router4, baseline학습0(새numeric실행의완성categorical대조exactreuse).60분cap;첫예정후보single+multi fit을전체규모선형1.2margin으로예측하고성능보기전초과면중단한다. 예상30–45분이며성능최적화목적의재시도0.

unweightedpooledRMSE/pairedstation-episodeCI90/worstquarter/station/lead/wind-slice. 평균delta<0후보보존;안정성악화는별도표이고자동탈락아님. onset/greedy NOT_ENABLED. numeric와결합/이득lead별배합/추가threshold없음. 완료후savedmodel새PIDreplay/독립산술QA를수행한다. 개선한완성정책만별도full2+fullrouter1후내부QA→승인된localCSV/replay/portable로이어간다. 본historical단계공식입력/CSV/upload0.
