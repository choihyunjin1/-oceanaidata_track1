# P1 v58 conductivity-cell thermal-lag research-frontier audit

## 결론

`NO_GO_ZERO_FIT_CONDUCTIVITY_CELL_THERMAL_LAG_CONTRACT_UNAVAILABLE`로 종료한다.

감사한 축은 정확히 하나, conductivity cell 벽의 열관성 때문에 raw conductivity가 받는 온도 오차를 물리 response model로 예측한 causal residual이다. Lueck의 논문은 cell 내부 유동, 경계층, 벽에 저장된 열이 측정 conductivity에 미치는 동역학을 유도한다 ([Journal of Atmospheric and Oceanic Technology, 1990](https://journals.ametsoc.org/view/journals/atot/7/5/1520-0426_1990_007_0741_tiocct_2_0_co_2.xml)). 이는 P1 instrument type·raw channel·response parameter·성능의 근거가 아니다.

P1 schema는 `temp`와 이미 처리된 practical salinity `psal`, `depth`만 제공한다. raw conductivity, sensor-to-cell time alignment, pressure와 conductivity-to-salinity 처리 이력, cell geometry/material/thermal coefficient, pump 또는 free-stream flow, instrument·firmware와 upstream correction 상태가 없다. `psal`은 temperature·pressure·calibration·thermal-lag correction을 이미 포함했을 수 있으므로 원래 cell-wall state를 유일하게 역산할 수 없다.

repo-wide exact conductivity-cell thermal-lag/inertia P1 구현은 0건이었다. 하지만 processed `temp`–`psal` lag나 hysteresis 대용은 새 관측축이 아니다. TS matched filter가 동일 경계의 salinity evidence를, v7이 ordered temp-psal signed path area와 increments를, base model이 difference/rolling temp·psal을 이미 사용한다. v21 physical-consistency transform도 닫혀 있다.

따라서 support와 semantic gate에서 fail-closed했다. v28/v33, feature, split, add-only 계약은 변경하지 않았다. executable preflight, lock, train/target read, fit, optimizer, action, removal은 0이며 official/test/sample/submission/hidden/CSV/upload도 모두 0이다.

최소 신규 증거는 train/deployment 공통의 동기화된 raw conductivity·thermistor temperature·pressure, stable instrument identity, cell geometry/material, flow, sampling alignment, calibration 및 prior-correction provenance다.
