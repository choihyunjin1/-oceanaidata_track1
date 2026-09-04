# P3 continuous-energy KMA factor v14b

## 결론

- 판정: **NO_GO_ENERGY_KMA_DIRECTION_CLOSED**; PASS 0/1; CSV 0; upload 0.
- pooled delta RMSE: -0.003547692m.
- episode CI90: [-0.006731288734263413, -0.0003220527715006179]; block×station CI90: [-0.0072935723407009986, 0.0003657921322616334].
- raw LCB 0.000000000점, family penalty 0.049586054점, calibrated -0.049586054점.
- improved blocks 4/6; worst station×lead +0.007404054m; changed share 0.333333.

## 설계

- α18=.20 고정, α24=.20+.40·train-prefix ECDF(wave_energy_current); hard threshold와 결과 기반 검색은 없다.
- 각 outer block 시작 78시간 전까지의 feature-only history로 ECDF를 고정한 뒤 outer truth를 채점했다.
- 중복된 90,601-point α-grid v14는 실행하지 않고 취소 receipt로 보존했다.
