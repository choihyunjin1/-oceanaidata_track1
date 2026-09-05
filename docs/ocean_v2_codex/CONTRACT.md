# 구속 평가 계약 — 2026-09-06 분리 실험

결론: **ocean_forward_v5의 분할·지표·선택 표면을 유지**한다. 원본 지원 검사는 완료됐지만, 각 모델 어댑터의 누출·특징 검사를 통과한 새 실행만 학습할 수 있다. 과거 v4 점수는 이 평가의 기준 수치가 아니다.

## 근거와 상태

- 고정 계약: [ocean_forward_v5.json](../../configs/evaluation/ocean_forward_v5.json), SHA256 `ca6f610aa087c5d2f4c3c25e7af487178c2d344b527d987f0571ddeb178b8a5b`.
- 구현: [ocean_evaluation_contract_v5.py](../../scripts/ocean_evaluation_contract_v5.py).
- 지원 감사: [report-source.md](../../reports/ocean_forward_support_20260906_v1/report-source.md). 원본 지원·키·시간 검증이며 실제 모델 특징 검증을 대신하지 않는다.
- 수정 로드맵: [ROADMAP_v2_amended_20260906.md](ROADMAP_v2_amended_20260906.md), 다음 설계: [NEXT_SCORE_DESIGNS_20260906.md](NEXT_SCORE_DESIGNS_20260906.md).
- 고정 JSON의 `fit_budget: 0`, `NOT_RUN`은 계약 구현 당시 상태로 보존한다. 이후 지원 검사와 승인된 학습 예산은 새 문제별 config/receipt에 기록하며, 고정 파일을 사후 변경하지 않는다.
- 배포 데이터만 사용한다. 리더보드 역산 상수·과거 답안 의존·외부 관측 계보는 금지한다. 정상적인 사전 고정 하이퍼파라미터와 학습 산출물은 출처를 기록한다.

## 문제별 표면

| 문제 | 구속 평가 / 주 지표 | 지원과 제한 |
|---|---|---|
| P1 | KST forward 3반기, 21일 purge. H1_2025 달력 행 pooled F1 | 평가 행 172,638 / 208,093 / 287,862. 주 구간의 107,125행(51.48%)은 학습에 없는 정점·층 조합. 제외하지 않고 별도 위험 보고 |
| P2 | KST 양측 7일 purge, B1~B8. B3 자연 T5 조건 pooled RMSE(℃) | 총 166,268 eligible keys. 목표 2/3/4층 temp·psal 동시 은폐. 마지막 17일 T5 동시 은폐 보조 검사. B4/B8 해당 구간은 0행으로 NOT_ESTIMABLE |
| P3 | UTC forward 5분기, 78시간 purge 및 동일 정점·episode 제외. 6lead 전체 pooled RMSE(m) | dense 24,360 anchors 중 평가 17,267 anchors / 103,602 lead rows. 과거 sparse 181 cases와 다른 표면 |

키와 순서는 정확히 일치해야 한다. 결측·비유한 예측이나 일부 행 누락을 inner join으로 숨기지 않는다. pooled F1은 합산 TP/FP/FN, RMSE는 합산 SSE/행 수로 계산하며 fold별 지표의 단순 평균으로 바꾸지 않는다.

P1은 양측 반기 CV를 주 지표로 바꾸지 않는다. H1 및 정점·층 신규/기존 조합을 보고한다. P2는 0.71/0.29 합성 점수를 사용하지 않는다. P3 onset 가중/greedy 평가는 아직 `NOT_ENABLED`이며, 정점·리드·분기 위험을 보고하되 onset 검증 완료를 주장하지 않는다. 원본 episode는 20분 연속 고파고 run으로, 독립 기상 사건임을 보장하지 않는다.

## 어댑터와 이번 승인 예산

모든 새 fit은 CPU 전용·문제별 2 threads·90분 cap이다. 세 문제 동시 6 threads를 배정한다. 자원 pilot은 첫 실제 fit으로 재사용하며, 성능을 보기 전에 시간 초과 예상이면 보고한다. 기존 GPU 결과를 새 CPU 기준 수치로 이식하지 않는다.

| 실행 ID | 변경 / 비교 | 최대 fit |
|---|---|---|
| p1_bracket_forward_20260906_v1 | B에 6/24/72h bracket 특징만 추가; O 및 decoder 알고리즘 고정 | full O 결정론 검사 2 + 기준 inner/outer 12 + 후보 inner/outer 6 = 20 |
| p2_crossfit_copula_forward_20260906_v1 | 같은 CPU C3 위 in-sample 잔차와 inner purged OOF 잔차 copula 비교 | outer C3 24 + inner C3 48 + copula 16 = 88 |
| p3_numeric_lead_forward_20260906_v1 | single CatBoost의 lead categorical→numeric만 변경; CPU multi 공통 | CatBoost 15 + 이전 OOF 기반 소규모 router 최대 8 |

P1의 hysteresis decoder는 유한 21일 footprint라고 주장하지 않는다. train/inner/outer 입력을 먼저 분리하고 해당 구간 안에서만 encoder·통계·특징·decoder를 처리한다. 다른 구간 값의 변경이 대상 구간 결과에 영향을 주지 않는 sentinel 검사가 필요하다. full O 두 실행의 고정 학습 특징 probe는 **결정론 검사 전용**으로, held-out 성능 검사나 두 공식 답안의 동등성 검사가 아니다.

P2는 모든 파생 특징을 target/T5 mask 뒤 재계산한다. 실제 깊이가 부족한 B1/B3 33/34행을 조용히 삭제하지 않고 기존 nominal fallback의 적격성을 확인한다. 각 outer train에 완전히 속하는 지원 있는 월을 날짜순 1/3·2/3에서 선택한 두 inner 구간과 ±7일 purge를 성능 노출 전에 봉인한다. outer label은 잔차 모델 학습에 사용하지 않는다.

P3 cache는 raw source와 특징 계보·키·해시가 일치할 때만 재사용한다. router는 각 arm의 이전 OOF만 사용하고, meta-fit에도 target-ready 및 경계 purge/동일 episode 제외를 적용한다. 최초 평가 fold는 이전 OOF가 없으므로 router no-op이다.

## 판단과 보고

- Δ = 후보 − 대조. F1 양수, RMSE 음수가 개선이다.
- paired cluster bootstrap 2,000회, seed 20260906, 5/95% CI90. P1 KST 하루, P2 KST 7일(기준 2024-01-01), P3 정점·episode 단위. 최소 2개 cluster 미달은 NOT_ESTIMABLE.
- P(개선)은 bootstrap 개선 비율이며 공식 점수 향상 확률이 아니다. 0.8을 자동 탈락/승격 기준으로 사용하지 않는다.
- 주 pooled 지표의 엄격한 평균 개선은 후보 보존 조건이다. worst-block/정점·층·리드 악화, CI, 재생성 적격성은 별도로 보고한다. 자동 제출하지 않는다.
- 새 기준선의 동일 평가 키 OOF를 먼저 완성·봉인하고 후보 지표를 확인한다. 기준 수치/결과가 없는 실행은 설계 완료와 성능 개선을 구분한다.
- 누적본과 fallback은 개별 Δ를 합산해 성능을 추정하지 않고 같은 계약에서 다시 비교해야 한다.
- 이번 단계 공식 입력 접근·제출 CSV·업로드·commit·push는 0이다. 최종 패키지 빈 모델 폴더 학습→추론 및 6시간 재현은 별도 미완료 항목이다.

표 형식: `변경 | 주 pooled Δ | CI90 | P(개선, 참고) | worst-block Δ | 정점/층/리드 최대 악화 | 보조 진단 Δ | 소요시간 | 재생성 적격/후보 보존`.

## 추가 피드백 이후: P1 차기 양측 계약의 지원 검사

아래는 **현 v5를 대체하지 않는 별도 다음 계약 후보**다. [0-fit 지원 검사](../../reports/p1_twosided_support_20260906_v1/result.json)는 temp 특징/예측 없이 배포 train의 키·시간·label-run만 읽어 확인했다.

| 검증 반기 | 양측 train | 검증 행(기존과 동일) | 기존 forward 미지원 → 양측 미지원 | whole-run 추가 제외 |
|---|---:|---:|---:|---:|
| H2_2024 | 565,539 | 172,638 | 0 → 0 | 126 |
| H1_2025 | 514,367 | 208,093 | 107,125 → **0** | 13 |
| H2_2025 | 444,081 | 287,862 | 0 → 0 | 0 |

따라서 미지원 행을 삭제하지 않고도 양측21일 purge와 양성 run 격리로 정점·층 지원 공백을 없앨 수 있다. 이 방향을 차기 설계로 택한다. 다만 같은 정점·층이 존재한다는 것은 충분한 학습량·분포 일치·높은 점수를 보장하지 않는다. 미래 시기 배포 학습 label도 쓰는 retrospective 평가이며 forward 미래 예측 질문과 다르다.

이 지원 검사는 새 evaluator 전체 PASS가 아니다. inner 정책 선택 구간, outer/inner 중첩 purge, 양측 train의 분리된 시간 구간과 특징/decoder 격리 검사를 별도로 고정해야 한다. 그 전 새 B 수치·후보 수치를 산출하거나 현재 v5 결과를 새 계약의 성과로 바꾸지 않는다. 현 P2/P3 계약은 변경하지 않는다.
