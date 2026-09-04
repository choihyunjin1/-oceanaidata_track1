# P1 v30 최종 결론

`P1_1_LABEL_FREE_RELIABILITY_GUARDED_LABEL_SHIFT_EM`은 사전봉인한 모든 내부 gate를 통과했고, frozen full-history 계약으로 공식 169,011행 후보를 exactly once 생성했다. 내부 pooled ΔF1은 `+0.0018209304`, Q3 `+0.0031047830`, Q4 `0`, raw 예상 `+0.048396908점`, v3 transport penalty 후 `+0.043013217점`이다. Bootstrap P(improve)는 `1.0`, CI90은 `[+0.000764423,+0.003172068]`이며 33개 내부 추가가 모두 TP였다.

최종 CSV는 `C:\Users\cedis\PycharmProjects\PythonProject\submissions\p1_public_transport_repair_cycle_20260831_v30\P1_submission.csv`, SHA-256은 `639c26cda576da74880e3f887b8653465db16c8572deedf225e4ae39ae51efa3`이다. 169,011행, exact schema/key/order, key·row duplicate 0, binary finite integer label, anchor removal 0, KST-day addition share 최대 `0.004995005`를 독립 재계산했고 공식 validator도 PASS했다. Hidden truth와 upload는 0이다.

## 비중복성과 누출 경계

v29는 inner group label의 precision과 ΔF1으로 group eligibility를 정하지만, v30의 guard 함수는 label 인자를 받지 않는다. Prefix score에서 calibrator posterior와 frozen three-source mean 사이 discrepancy만 계산하고, outer에서는 corrected-score margin lower bound와 label-free day cap만 쓴다. Outer label은 prediction seal 뒤 내부 평가에만 사용했다.

## 중요한 수송 caveat

Official label-free EM target prevalence가 `0.999999`로 경계에 수렴했고 day cap 전 margin-eligible row가 133,153개였다. 사전봉인 cap 때문에 실제 추가는 755개로 제한됐지만, 이는 official score distribution에서 label-shift 가정이 매우 강하게 흔들렸을 가능성을 뜻한다. 이 관측을 보고 EM·threshold·group bound·cap을 변경하거나 재시도하지 않았으며, 후보는 **내부 PASS이지만 수송 위험이 명시된 제출 준비본**으로 취급한다.

## QA

- exactly-once historical: 2 fits, 22.792s, result SHA `5f478359...b04826`
- exactly-once materializer: 94.136s, runner SHA `432766ee...e64ab5`, lock SHA `e40330b2...5c7f5`
- py_compile PASS, Ruff PASS, focused pytest `10/10` PASS
- `scripts/validate_submission.py` PASS
- postrun independent QA: `reports/p1_public_transport_repair_cycle_20260831_v30/postrun-qa.json` PASS
- official input은 materialization/schema QA에만 사용했고 hidden label과 upload는 0
