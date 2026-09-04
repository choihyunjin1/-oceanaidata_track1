# P1 v56 ADC code-density research-frontier audit

## 결론

`NO_GO_ZERO_FIT_ADC_CODE_DENSITY_SOURCE_CONTRACT_UNAVAILABLE`로 종료한다.

이번에 감사한 축은 단 하나, station-layer별 raw ADC output-code 점유로 missing code 또는 differential nonlinearity(DNL)를 찾는 causal 진단이다. [IEEE 1241-2023](https://standards.ieee.org/ieee/1241/6797/)은 ADC 용어와 시험방법의 근거일 뿐, P1 성능이나 P1 `temp`가 raw code라는 근거는 아니다.

허용된 README/schema에는 `station, year, layer, time, temp, psal, depth`만 있으며 `temp`는 물리 수온값이다. raw integer code, converter bit depth, LSB/code width, transition level/transfer function, gain·offset·calibration·serialization provenance, code-density stimulus 계약은 없다. 따라서 소수점 값의 빈 구간을 ADC missing code로 해석할 수 없다. 센서 양자화, 보정, 단위변환, 반올림, CSV 직렬화, 실제 해양상태 반복을 구분할 수 없기 때문이다.

P1 전체 fingerprint에서 exact ADC/DNL/code-density 구현은 0건이었다. 그러나 임의 decimal grid로 만든 대용 feature는 새 관측축도 아니다. `src/p1_qc/rules.py`는 exact/near-exact plateau와 quantized quiet-series의 zero step을 이미 다루고, `causal_soft_symbolic.py`는 물리량 구간의 symbol/transition을, v42는 prefix state-space occupancy를 이미 다룬다. bit depth나 bin width를 결과 전에 임의 고정해도 계측 의미가 복구되지는 않는다.

이에 semantic/support gate에서 fail-closed했다. executable preflight, lock, train row, target, fit, optimizer, action, removal은 모두 0이다. official/test/sample/submission/hidden/CSV/upload도 모두 0이다. v54와 v55는 그대로 immutable이며 gate, feature, split에는 손대지 않았다.

다음 연구에 필요한 최소 새 증거는 competition-allowed raw integer sensor code와 converter resolution/code-transition/calibration provenance, train/deployment 공통의 정당한 occupancy contract다. 이것이 없으면 authenticated deployment-time scientific covariate를 새로 확보해야 한다.
