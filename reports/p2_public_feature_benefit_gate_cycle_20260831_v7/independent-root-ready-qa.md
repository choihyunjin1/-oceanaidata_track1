# P2 v7 independent root-ready QA

별도 재계산 경로에서 result/prediction/submission hash, Sep→Oct split 경계, Oct RMSE와 layer별 delta, 2,000회 KST 일자-block bootstrap, 점수 환산과 최악 수송 penalty, inclusive `>=0.01` gate를 다시 계산한다. Train/test timestamp 중복은 0이고 경계는 한 sampling interval인 10분이다. 입력 lag는 public past-only covariate이며 target-derived lag는 없다.

공식 CSV는 26,061행, exact schema/key/order, duplicate 0, finite, SHA-256 일치를 대조한다. 내부 result의 공식 접근은 0행이고 PASS 확정 후 materialization 단계에서만 test_index 26,061행을 읽었으며 hidden truth와 upload는 0이다.

기계 판정은 `independent-root-ready-qa.json`을 따른다.
