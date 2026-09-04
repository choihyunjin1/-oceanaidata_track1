# P1 v28 결론

`P1_1_PREQUENTIAL_LABEL_SHIFT_EM_STACK_ADDONLY`는 pooled 성능과 v3 transport gate에는 강한 개선을 보였지만 **strict NO_GO**다. Q3 `ΔF1 +0.0147666`, Q4 `+0.0001334`, pooled `+0.00874657`, raw expected `+0.232467점`, calibrated `+0.227084점`, bootstrap CI90 `[+0.002100,+0.017095]`, `P(improve)=0.993`였다. 그러나 maximum KST-day changed fraction `0.033066 > 0.005`이고 I-ORS layer3 `-0.008838`, layer4 `-0.011480`으로 사전봉인 safety gate 두 개를 위반했다.

Exactly-once 2 fits, runtime `20.349578s`, 222 additions 중 188 TP/34 FP로 precision `84.68%`다. Q3은 221행을 추가했고 Q4는 1행만 추가했다. 두 EM은 각각 78회에 수렴했고 outer labels는 prediction seal 전에 0행 읽었다. Anchor removal, official read, hidden read, CSV, upload는 모두 0이다.

이 결과는 label-shift EM 가설의 정보가치를 보여주지만 제출 후보가 아니다. Day concentration이나 I-ORS slice를 결과 뒤 hard-router로 막는 것은 사후 조정이므로 금지한다. 별도 신규 가설이 필요하면 이 분산 문제를 train-prefix에서만 연속 regularization하는 새 candidate로 사전등록해야 한다.
