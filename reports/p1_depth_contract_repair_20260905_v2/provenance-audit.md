# P1-B 기존 최고점 부품 감사

## 결론

**현재 조건에서 기존 e150/router OOF와 새 tree의 0-fit 결합은 실행하지 않았다.** 모델 성능 실패가 아니라 exact split/purge/key 계약 불일치다. 옛 MS-TCN을 외부자료 모델로 일괄 폐기하지 않는다. 새 GPU 학습은 이번 시간 예산에서 시작하지 말라는 root 지시를 따랐다.

## 입증한 불일치

현재와 옛 OOF의 전체 키는 모두 421,032개이며 집합도 같다. 그러나 119행의 Q3/Q4 소속이 다르다. 원시 관측이나 행별 정답을 열지 않고 KEYS+fold만 읽어 1:1 merge로 확인했다.

| 분할 | 옛 event-protected | 현재 calendar |
|---|---:|---:|
| Q2 | 133,170 | 133,170 |
| Q3 | 176,738 | 176,619 |
| Q4 | 111,124 | 111,243 |

옛 e150의 Q3 holdout은 일부 series-local 사건 때문에 2025-10-01 19:40 KST까지 포함한다. 현재 Q3는10월1일00:00 직전까지다. 동일 전체 행 수만 보고 확률을 복사하면 서로 다른 학습 prefix의 예측을 섞게 된다.

- e150 Q3/Q4 학습 마지막 시각: 6월9일/9월9일23:50 KST, separation504시간10분. 21일 purge와165특징 명세가 남아 있다.
- router O/B source는 `p1_meaningful_learning_curve_generation_v1/prediction_parts/*_p100.json`의 실제 cutoff가6월23일/9월23일23:50 KST다. 7일 purge 계보로 새 tree의21일과 다르다. 설정뿐 아니라 당시 fit receipt에서 확인했다.
- 보존된 e150의 Q3/Q4 학습 receipt는6fits에157분18초를 기록했다. 동일계약9 historical+3 full GPU 재학습을 오늘 짧은 잔여시간에 자동 시작하지 않았다. 이는 새 실행 시간의 보장은 아니다.

## 부품별 출처와 재사용 조건

1. **O/B router:** `build_p1_current_router_oof_anchor_v1.py`는 O/B 불일치에 일반 station/layer 조건을 적용한다. `package_preregistered_submission_20260826.py`에 router/intersection/union 세 완성 후보가 있으며 로컬 시간 검증 기반이라는 설명이 남아 있다. 완성 후보의 공식 비교와 Public 정답/계수 역산은 구분한다. 그러나 기존 official CSV를 `router_anchor.csv`로 가져오는 것은 source→training→prediction 재현을 대체하지 못한다.
2. **e150:** `p1_mstcn_checkpoint_diagnostic_20260827_v2`는 width512/epoch150/threshold0.8/3seeds의 별도 계보다. epoch125 main v2와 trial18 frozen confirmation은 동일 모델이 아니다. cached full-year nominal depth/plateau totals를 제외하고 current-row depth를 phase train에만 적합한 분위수로 변환한다는 명세가 있다. 배포 train-only/누출 방지 계약은 확인했으나 이번 감사에서 모든 옛 checkpoint의 새 추론까지 재현한 것은 아니다.
3. **GI spike2:** `build_deadline_probe_set_20260828.py`의 원래 생성 조건은 novel AND predicted-type-spike라는 일반식이다. 그러나 GI의 입력은 고정 official CSV 계보이므로 새 후보에 특정 행 patch나 완성 CSV를 복사하지 않았다. 일반 spike 규칙의 source→fit→predict 계보를 새로 완결하기 전에는 제외한다.

## 실제로 남아 있는 성능 단서

옛 e150 고정 실험은 원래 Q3+Q4 287,862행에서 router F1 0.902917→0.906804(+0.003887)였다. Q3 +0.017209, Q4 −0.015441이고21일 블록90% CI는 [−0.013148,+0.021144]였다. 개선 평균과 계절 위험이 함께 존재했다. 이는 현재 calendar 평가에서 재검증된 수치가 아니며 공식 예상 상승폭이 아니다. `fixed_epoch_150_metrics.json`에서 읽은 과거 집계다.

## 재개 조건

새 exact-contract OOF와 재현 가능한 full model을 별도 GPU/시간 승인으로 확보한 뒤 no-op 포함 최대3개 일반 결합만 비교한다. 이번 A의 임계값/분할을 결과에 맞춰 바꾸거나 Q3 우승과 Q4 fallback을 사후 붙이지 않는다. [기계 판독 영수증](provenance-audit.json)에 source hash와 key 비교를 남겼다.
