# Chart map

- 보고서 구간: 실제 residual에서 과민했던 fixed likelihood
- 분석 질문: 각 historical fold에서 incumbent F1, candidate F1, addition-only precision이 얼마나 벌어졌는가?
- 선택 형식: grouped vertical bar
- 필드: `scope`, `metric`, `value`; 보조 감사 필드 `rows`, `additions`, `delta_f1`
- 근거 주장: 세 fold와 pooled 모두 candidate가 incumbent를 크게 하회했고 addition precision은 3.46%~4.90%였다.
- 팔레트: blue/orange/gold의 3개 범주, 축과 legend를 함께 사용해 색만으로 구분하지 않는다.
- 전달 표면: Data Analytics MCP technical report
- QA: 4개 scope와 3개 같은 단위 rate 비교라 grouped bar가 적합하며, percent 축은 0~1 decimal rate를 사용한다.
