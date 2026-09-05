# 기준 코어 재생성 검사 — 2026-09-06 현재 증거

## 결론

P1/P2/P3 모두 새 빈 모델 폴더에서 배포 데이터로 학습하고 새 답안을 만들었다. P2/P3는 과거 clean 답안 SHA도 같고 P1은 다르다. **동일 최종 학습 진입점을 독립 환경에서 두 번 실행한 결정론 검사 및 portable 패키지는 미완료**다. 저장 모델 replay를 재학습 결정론 PASS로 표시하지 않는다.

| 문제/검증 ID | 실제 코어 | 소모시간 | 새 학습→답안/별도 PID replay | 과거 clean SHA | 남은 쟁점 |
|---|---|---|---|---|---|
| P1 `p1_clean_regeneration_20260905_v5` | XGBoost O + LightGBM B, inner 선택 decoder | 203.343s 전체 | PASS / 169,011행 | FAIL, 최소 110 label 차이 | full O 재학습 차이 원인; 동일 full 실행 두 번 대조 |
| P2 `p2_clean_regeneration_20260905_v6` | v23 blockmask DeepSets 3-seed C3 | train 69.094s / infer 10.844s / replay 8.734s | PASS / 26,061행 | PASS | CPU-only 별도 시간·결정론 및 portable 통합 입구 |
| P3 `p3_clean_regeneration_20260905_v4` | single/multi CatBoost + 새 OOF에서 학습 router, clean recipe | prepare부터 infer 1,654.002s / answer replay 3.862s | PASS / 1,200행 | PASS | multi GPU 포함 실행; CPU-only 등가 비교는 별도 모델/환경 계약 |

## 제외 계보와 허용값

과거 `router_anchor.csv`, `gi_spike2_patch.json`, `bin17_anchor.csv`, Public 역산 alpha=-10.217/axis, 외부 관측·재분석·실관측 사전학습 가중치는 위 새 실행의 예측 입력에서 제외했다. 과거 파일 자체는 감사 증거로 보존했다. 파일 이름에 alpha가 들어 있다는 이유로 모든 정규화를 금지하지 않는다.

현재 clean P3의 Ridge alpha=10, 사전등록 shrink=0.2는 위 역산 alpha와 다른 출처로 기록되어 있다. Fable의 shrink 없는 CatBoost 등가 평균과 **동일한 모델이 아니다**. P1 MS-TCN 추가와 P2 v52/PAVA 전환도 현재 코어 재생성 검사의 결과로 주장하지 않는다. 필요하면 독립 변경 후보로 비교한다.

JSON은 의무 저장 형식이 아니다. 적합값은 새 학습이 생성한 모델 파일 및 metadata에서 로드하고, seed/구조/학습률 등 선고정 설정은 별도 출처 원장에 둔다. 같은 디렉터리에서 새 학습이 만든 OOF를 router 학습에 사용하는 것은 과거 답안 복사와 구분한다. 최종 추론이 과거 OOF를 요구하지 않게 한다.

## 단일 근거와 실패 보존

- [P1 v5 실행·21-check QA](../../reports/p1_clean_regeneration_20260905_v5/report-source.md), [P1 v4 정책 불일치 실패](../../reports/p1_clean_regeneration_20260905_v4/report-source.md).
- [P2 v6 실행·27-check QA](../../reports/p2_clean_regeneration_20260905_v6/report-source.md). v4/v5 native writer 감사 가드 실패는 품질 실패가 아니며 첫 seed 학습 각각을 보존했다.
- [P3 실행·133-check QA](../../reports/p3_clean_regeneration_20260905_v4/report-source.md).
- [상수 및 모델 출처 원장](../../reports/parallel_isolated_and_regeneration_20260905_v4/constant-lineage-ledger.md).

이 문서는 기존 실행 영수증을 연결한 요약이다. 이 문서 작성으로 모델을 다시 학습하거나 과거 점수를 새 P1 답안에 승계하지 않았다. 최종 README에는 확인하지 않은 CPU 결정론·독립 차단망·운영진 검증 PASS를 적지 않는다.

## 09-06 추가: P1 full O 구성 요소 결정론

`p1_bracket_forward_20260906_v1`에서 CPU2·동일 canonical full O 경로를 새 폴더/서로 다른 PID33936·5768로 두 번 실행했다. 각776,706행/80features, 74.938초·79.359초. 모델 SHA `58430c8327b674e40c16a09e9bd19fd77cbef691bef6f3282c9a8140bb2aa696`, 전체 특징 행렬 및 고정 학습 특징 probe 예측이 모두 exact 일치했다.

이는 **O 구성 요소의 현재 CPU2 환경 결정론만 PASS**다. probe는 학습 행으로 품질 점수를 계산하지 않았다. 기존 CPU4 `5971…` 답안 전체 파이프라인의 두 번 scratch 재생성, 과거 `064ef…` 답안 복원, 다른 머신/portable cleanroom 검증은 여전히 미완료다. O+B 및 inner 선택까지 포함한 전체 최종 진입점 검사로 확대 해석하지 않는다.

근거: [이번 실험 보고](../../reports/p1_bracket_forward_20260906_v1/report-source.md), [60-check 별도 PID QA](../../reports/p1_bracket_forward_20260906_v1/independent-qa.json). 실제 채점 SHA 대조는 [UPLOAD_SET_1.md](UPLOAD_SET_1.md), portable 상태는 [CLEANROOM_RESULT.md](CLEANROOM_RESULT.md) 참조.
