# P1-C 고정 decoder 사전등록

2026-09-05 20:59 KST, A 학습 terminal 및 성능 공개 전에 기록. B key-only 감사는 완료됐으며 exact-contract OOF 복구는 차단됐다.

- 실행: A terminal 및 independent QA PASS 이후만 `run_p1_depth_contract_postaudit_20260905_v2.py --decoder`.
- runner SHA-256: `d3024318e4cecfc611d1a69031945e6956086f78b4db1bf5bf3e0a03d51734c9`.
- 재사용 decoder source SHA-256: `d5ed9ff15745b7e400066eff2331c533d6fd3a420e941d04cf317ef40b530db9`.
- A의 새 year-safe tree 확률과 earlier-inner threshold는 고정한다. OFF와 고정 ON(lambda1, laplace1, probability clip1e-6)을 대조하며 새 threshold/강도 탐색은 없다.
- transition은 해당 outer 학습 prefix에서만 추정(3개), 배포용은 full train에서1개 추정. 신규 backbone fit0.
- 원래 control/no-op, year-safe OFF, year-safe ON의 pooled F1를 비교한다. 동률은 앞의 단순 정책을 보존한다. 7일 공통시간블록 bootstrap2,000회(seed20260905)의90% 구간과 월/정점층 위험을 함께 표시하지만 hard gate로 쓰지 않는다.
- 이 선택은 반복 노출된 historical development 결과다. 독립 confirmatory 성능 또는 공식 예상 점수로 해석하지 않는다.
- 공식 입력·CSV·업로드0. A 결과/모델/코드/lock를 변경하지 않고 `p1_c_fixed_decoder/`와 별도 decoder-result만 추가한다.
