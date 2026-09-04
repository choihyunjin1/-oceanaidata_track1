# P1 v28 duplicate, leakage, and assumption audit

## 결론

`P1_1_PREQUENTIAL_LABEL_SHIFT_EM_STACK_ADDONLY`는 synthetic preflight만 허용해 exact preseal한다. Historical fit, attempt lock, official/hidden read, CSV, upload는 모두 0이다.

## 비중복 경계

- v23은 continuous HGB lineage가 보존되지 않았고 posthoc stable-top-k 위험 때문에 0-fit fail-closed됐다. v28은 HGB와 top-k를 쓰지 않는다.
- v24는 exact v16 GCE 165-feature score에 inner F1 threshold를 붙인다. v28은 GCE·165-feature representation을 쓰지 않고 frozen `base/peer/e150` probability 세 개의 logit만 쓴다.
- v5-v8의 tree/calibration/consensus/drift 실험과 과거 heterogeneous event head는 fixed threshold, discrete source signature, event bank 또는 learned router였다. Repository search에서 exact Saerens-style unlabeled-target EM prior correction over these three frozen logits는 없었다.
- v25 oracle의 Q2/Q3/Q4 precision 숫자는 threshold, coefficient, EM stopping rule, source selection에 사용하지 않았다. 그것은 label-shift 가설 선택에만 사용했다.

## 봉인된 선택 순서

각 outer train-prefix의 전체 timestamp 앞 75%에서 anchor-negative rows의 three-logit logistic calibrator를 한 번 fit한다. 뒤 25% label만으로 source prevalence와 add-only F1 threshold를 정한다. Outer에서는 labels를 읽기 전에 calibrator score distribution만 EM에 넣어 target prevalence와 posterior odds를 연속 보정한다. Q3/Q4 label은 candidate bits seal 뒤 평가에만 사용한다.

Hard quarter/station router, top-k, score-mass cutoff, fixed consensus, outer-result threshold update는 금지한다. EM이 200회 내 tolerance `1e-10`으로 수렴하지 않으면 retry 없이 terminal technical failure다.

## 가정 및 gate

Label shift는 `p(x|y)`가 유지되고 prevalence만 바뀐다는 가정이다. v25 reversal은 이 가정을 입증하지 않으며 conditional shift라면 v28은 실패할 수 있다. Q3/Q4 역시 재사용 development surface여서 독립 confirmation이 아니다.

Prospective calibration v3 SHA `0f448207...21a10`, P1 penalty `0.005383691점`, inclusive raw gate `0.015383691점`, calibrated gate `+0.01점`을 사전 고정한다. Q3/Q4 각각 비악화, dependent bootstrap CI90 low>0 및 P(improve)>=0.8, anchor removal0, daily/station-layer-quarter sparse-change gates를 모두 유지한다. 최대 model fit은 2회다.
