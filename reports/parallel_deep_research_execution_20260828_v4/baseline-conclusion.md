# P1·P2·P3 다음 병렬 사이클 기준선 결론

- 기준 시각: 2026-08-28 KST
- 독자: 분당독고다이 연구·제출 의사결정자
- 기준 보고서: `reports/parallel_deep_research_execution_20260828_v3/P1_P2_P3_병렬_딥리서치_실행보고_20260828.docx`
- 기준 보고서 SHA-256: `d37ae76c5160c34dc6b4a3fef0be51bc4c2e57b965bbd23a714e9788b7976554`
- 기준 독립 QA: 16/16 PASS
- 관련 테스트: 42 PASS

## 고정 결론

### P1

- 실제 장기 이벤트 donor support는 확보했다: train 22 events / 7 cells, calibration 26 events / 9 cells.
- 그러나 event recall은 `10/26 = 38.46%`, row precision은 `22.27%`로 실패했다.
- 결론은 `NO_GO_CALIBRATION`이며, donor 생성량이 아니라 proposal localization과 정상구간 배제가 병목이다.
- 기준 result SHA-256: `b32fc6df07b30d315d1d3b09add4455686660ea69ba3b10134ac7f4e0a8c58f4`

### P2

- 저장소의 현재 로컬 1순위는 adaptive exposed surface에서 검증한 `p2_extrapolated_soft_gate_v2`다. 물리투영 기준 `0.7744179316°C`에서 `0.7683674566°C`로 `-0.0060504749°C` 개선했지만 fresh holdout이나 공식 개선 주장은 아니다.
- α40 보존 quasi-periodic GP residual은 RMSE를 `0.000008606°C`만 낮췄고 69,850행 중 75행만 보정했다.
- 3개 fold 중 1개만 개선되어 제출 가치가 없으며 `FAIL_GATE_STOP_NO_CSV_NO_RESEARCH_LOOP`다.
- GP threshold 사후 완화는 금지한다. 다음 구조는 현재 강한 로컬 1순위 또는 그 exact no-op 기준을 보존하는 물리적 two-mode vertical-displacement correction이다.
- 기준 result SHA-256: `c04755750357b8613f7372f98840ba8f8df365173af46524b4be339ee362da2e`

### P3

- ERA5 source gate는 persistence 대비 `-0.140897 m`, matched local-only 대비 `-0.023440 m`로 통과했다.
- 그러나 강한 incumbent 대비 `+0.002325 m`로 악화했고 3개 window 중 1개만 개선했다.
- 결론은 `NO_GO_LOCAL_OR_VIEWPOINT_GATE`이며, ERA5 transfer 자체의 가치는 입증됐지만 incumbent 대체는 실패했다.
- 다음 구조는 incumbent를 정확한 no-op 기준으로 보존하는 ERA5 residual/router다.
- 기준 result SHA-256: `ac92a530d230ea29c475e0b03acb7e16d577633b64401632cebb46fa4e0bbd2f`
- blind seal: 1,086행, `target_hs` 없음, SHA-256 `25accc81915e95bebcf4e69cd313b73520c36969b88521a186f5be214c4ba2a7`

## 이번 사이클의 연구 질문

1. P1: 기존 83개 proposal 위에 event-level ranker 또는 normality veto를 붙이면 recall을 보존하면서 precision을 충분히 높일 수 있는가?
2. P2: 공개층으로 추정한 1~2개 수직 변위 mode가 강한 incumbent보다 일관되게 RMSE를 낮출 수 있는가?
3. P3: ERA5 transfer의 이득이 확인된 상태·리드에서만 incumbent 잔차를 안전하게 보정할 수 있는가?

## 사전 고정 운영 경계

- 공식 33점 척도와 로컬 F1/°C/m를 같은 숫자로 직접 비교하지 않는다.
- 공식 test, sample submission, submission CSV를 읽거나 생성하거나 업로드하지 않는다.
- 결과를 본 뒤 같은 실험의 threshold, feature, fold, epoch, router 규칙을 다시 맞추지 않는다.
- 각 문제는 새 experiment ID와 격리된 artifact 경로를 사용하고, 단일 bounded 실행 뒤 PASS/NO_GO를 판정한다.
- 이미 노출된 historical surface는 fresh holdout으로 부르지 않으며, official probe 가치는 별도 판단한다.
- 원본 관측값은 보고서·로그·Git에 복제하지 않고 집계 지표와 해시만 남긴다.

## 승격 원칙

- 작은 로컬 개선도 방향성과 강건성이 있으면 연구 증거로 보존한다.
- 공식 후보 승격은 incumbent 대비 동일 검증면의 통합 지표 개선, 다수 fold/window 방향 일치, 안전장치, 재현 QA가 함께 필요하다.
- 명목 개선만 있고 개입률이 거의 0이거나 한 fold에만 집중되면 제출 후보로 올리지 않는다.
- 이번 사이클에서 후보가 없으면 실패가 아니라 병목을 다음 구조로 좁힌 결과로 종료한다.
