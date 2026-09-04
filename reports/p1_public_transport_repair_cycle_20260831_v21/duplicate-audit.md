# P1 v21 duplicate, leakage, and deployability audit

## 결론

`P1_1_CAUSAL_SCAR_PU_LINEAR_ADDONLY`는 합성 preflight만 허용해 사전봉인한다. historical fit, attempt lock, 공식 입력, hidden truth, CSV, upload는 모두 0이다. 현재 Q3/Q4는 반복 사용된 development surface이므로 향후 결과도 독립 confirmation으로 부르지 않는다.

## 선택 및 중복 배제

- event-level MIL은 채택하지 않았다. 과거 heterogeneous proposal bank의 contiguous event를 partial-pooling binomial head와 precision LCB로 선택한 `p1_addonly_hierarchical_event_precision_lcb_20260830_v1`이 이미 2-fit 실행됐고, Q3 `+0.008107` 뒤 Q4 `-0.017499`, pooled `-0.002381`이었다. long-event segment proposal/rescore와 event-balanced encoders도 같은 bag/segment 축을 넓게 사용했다.
- v17은 causal MiniRocket PPV 표현, v18은 soft-symbol transition 표현, v19는 결정론적 G-ORS run extension, v20은 완전지도 두 클래스 Student-t likelihood ratio다. v21은 새 표현·event router·Student-t를 쓰지 않는다.
- v16은 동일 165열 causal projection에서 fixed GCE q=0.7 및 fixed p=0.95를 사용했다. v21은 관측 양성을 reliable selected-positive, 0을 unlabeled로 두고 train-prefix inner tail에서 SCAR label propensity를 추정한 뒤 selection probability를 보정한다. 저장소 검색에서 이 propensity-corrected SCAR 실행은 없었다.
- v16의 91 additions 중 45 TP/46 FP, pooled `ΔF1 +0.000393`은 행 단위 정보가 완전히 무작위는 아니라는 가설 선택 근거일 뿐, v21 threshold나 PASS 기준을 정하는 데 사용하지 않았다.

## 봉인된 누수 경계

각 outer prefix에서 전체 timestamp의 앞 75%만 선형 selection model을 fit한다. 마지막 25%의 관측 양성 score 평균만 propensity `c`를 추정하며, 같은 inner tail에서 corrected probability `clip(p(s=1|x)/c,0,1)`의 threshold를 결정한다. threshold는 add-only inner F1 최대화, Wilson-90 added-precision lower bound가 inner incumbent F1/2보다 큼, changed share 0.5% 이하를 동시에 만족해야 한다. Q3/Q4 outer labels는 fit, propensity, threshold 어느 단계에도 사용하지 않는다.

## 가정과 실패 조건

SCAR은 확인된 데이터 생성 사실이 아니다. station/layer/event severity에 따라 positive label propensity가 달라지면 보정은 잘못 지정된다. 따라서 v21은 PU의 latent class prior를 식별했다고 주장하지 않으며, 결과가 좋아도 reused development evidence일 뿐이다. 행 삭제, 양성 삭제, outlier 제거, outer-result threshold 재조정, retry는 금지한다.

## 승격 계약

새 smooth learned family로 등록해 calibration v2의 `SMOOTH_LEARNED_PROFILE` penalty `0.121682092`점을 적용한다. 중앙 raw expected delta가 `0.131682092`점 이상이고, pooled 및 Q3/Q4 비악화, bootstrap 개선확률 0.8 이상, anchor removal 0과 sparse-change 안전장치를 모두 만족해야만 calibrated `+0.01` PASS다. 최대 historical model fit은 2회다.
