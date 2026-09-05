# P1 v2 남은 간극

| 질문 | 현재 증거 | 판단/다음 조건 |
|---|---|---|
| year lookup 없이 같은 관측을 같은 특징으로 만드는가 | synthetic9 tests 중 invariance/missing/order/legacy equivalence PASS; 실제 observed-but-nominal-missing0 | 기술 수정 확인 |
| 동일 temporal 평가에서 점수가 올랐나 | AΔF1−0.002213, CΔ−0.001303;24-check QA PASS | 개선은 확인되지 않음, control 유지 |
| 선택 임계값의 문제만인가 | 기존 control threshold 그대로 A에 적용해도Δ−0.001113 | 단순 threshold 복사도 해결하지 못함; 추가튜닝 안함 |
| 옛 최고e150 계보를 그대로 결합할 수 있나 |119행 fold소속 차이, oldrouter7일/newtree21일 purge | exact-contract0fit결합 불가. 모델 전체 부적격이 아니라비교계약 불일치 |
| e150은 과거 전혀 성과가 없었나 | 원래Q3+Q4에서+0.003887, Q4−0.015441, CI0포함 | 위험한 양의 평균 단서는 남음. 현재계약으로 별도재학습/QA 필요 |
| 제출용 모델은 학습됐나 | 배포train776,706행 fullO/B2fits, 저장hash/재로딩PASS | year-safe정보후보로 보존; 공식입력은아직0, 기존최종ZIP대체안함 |
| 공식 점수 상승을 예측할 수 있나 | 새A/C는 공식 미제출; internal개발면반복노출 | 예상점수미산정, Public역산 금지 |
| 다음branch자동확장은 | A→B감사→C완료, GPU12fit미승인 | 추가threshold/epoch/feature탐색없이 root에완결receipt |
