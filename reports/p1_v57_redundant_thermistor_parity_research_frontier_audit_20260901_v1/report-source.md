# P1 v57 redundant-thermistor parity research-frontier audit

## 결론

`NO_GO_ZERO_FIT_REDUNDANT_SENSOR_CHANNEL_UNAVAILABLE`로 종료한다.

감사한 축은 정확히 하나, 동일 station-layer에 공위치한 독립 thermistor 두 개의 동시 측정 차이를 causal parity residual로 쓰는 방식이다. Chow와 Willsky의 analytical-redundancy 연구는 알려진 redundancy relation으로 residual을 만들고 fault signature를 검사하는 방법론의 근거다 ([IEEE TAC 1984](https://doi.org/10.1109/TAC.1984.1103593)). 이 논문은 P1 layer가 중복 센서라는 주장이나 P1 성능 근거가 아니다.

P1 README에는 `station, year, layer, time, temp, psal, depth`만 있다. probe/instrument/channel ID, 동일 깊이의 독립 temperature channel, calibration·uncertainty metadata, 인증된 parity relation이 없다. 저장소의 `data.py`도 layer ordinal은 연도 간 안정적인 sensor identity가 아니라고 명시한다. 따라서 true redundant-channel residual은 현재 배포 관측으로 계산할 수 없다.

repo-wide exact analytical-redundancy/parity-space/redundant-probe P1 구현은 0건이었다. 하지만 다른 layer를 중복 probe로 대용하는 것은 과학적으로도 잘못이며 새 feature도 아니다. 기본 feature가 이미 same-station/time peer mean·residual·spread를 계산하고, stratification gate가 peer change coherence를, long-event family가 temp/psal/depth other-layer residual을 사용한다. v21 physical consistency와 v49 vertical-profile 계열도 이미 닫혀 있다.

따라서 support와 semantic gate에서 fail-closed했다. v28/v33, feature, split, anchor add-only 계약은 변경하지 않았다. executable preflight, lock, train/target read, fit, optimizer, action, removal은 0이며 official/test/sample/submission/hidden/CSV/upload도 모두 0이다.

최소 신규 증거는 train과 deployment 모두에서 안정적인 instrument ID로 구분되는 동일 station-layer 공위치 thermistor 두 개 이상의 동기 측정값과 calibration·uncertainty provenance다.
