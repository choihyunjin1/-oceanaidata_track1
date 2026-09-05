# 결론: 바람만 가려도 개선되지 않아 clean 기준선을 보존

6회 고정 학습을 완료했다. 바람만 결측 증강한 LightGBM은 matched control보다 RMSE **+0.01111106m 악화**했고, 더 강한 clean 기준선 대비로도 **+0.03972393m 악화**했다. 사전등록한 50% 혼합도 개선하지 못했다. 이번 결과는 안정성 gate로 후보를 배제한 것이 아니라 **주 평가량인 pooled SSE/RMSE 자체에서 개선이 없었던 것**이다. No-op을 보존하며 이 증강 가지의 full fit/제출은 하지 않는다.

## 내부 실측: 동일 181사례·1,086행

| 완성 정책 | SSE(m²) | RMSE(m) | clean 대비 ΔRMSE(m) |
|---|---:|---:|---:|
| clean no-op | 659.20672592 | 0.77910484 | 0 |
| matched LGBM control | 708.51498545 | 0.80771771 | +0.02861287 |
| control50 + clean50 | 675.55546767 | 0.78870682 | +0.00960198 |
| wind-only LGBM | 728.14188373 | 0.81882877 | +0.03972393 |
| wind-only50 + clean50 | 683.56626057 | 0.79336931 | +0.01426447 |

새 공식 점수 실측은 없다. 위 ΔRMSE를 Public 점수 향상으로 단정하거나 새 혼합비를 역산하지 않았다. 기존 내부 표면을 다시 사용한 탐색이며 독립 확증도 아니다. matched control은 이번 단일 seed 결과이고, 이전 v1 보고서의 2seed control 평균과 동일한 통계량이라고 혼동하면 안 된다.

wind-only는 clean 대비 73사례 개선/108사례 악화했고 case ΔRMSE 중앙값은 +0.01979780m, 90백분위 +0.18689487m였다. 이 안정성 진단은 별도 위험 기록이며 자동 탈락 조건으로 쓰지 않았다. 모든 fold/station/lead, wind관측 여부, 비바람 기상관측 여부, 과거3h onset proxy의 행수/SSE/RMSE는 [result.json](result.json)의 단일 집계 근거를 따른다. 원시 관측/target/개별 예측 행은 Git 제외 artifact에만 있다.

## 무엇을 새로 했는가

이전 full-weather330특징 가림과 달리 wind-only229특징만 함께 가리고 기압·기온·습도101특징을 보존했다. synthetic raw 289행의 `wspd/gust/wdir`를 제거하고 재계산한 1,275개 요약과 feature-level 가림이 정확히 일치했다. 따라서 gust excess·방향 sin/cos·바람-파랑 alignment·풍속 변화량 등의 파생값을 남기는 결측 누출은 없다.

기존 train-derived cache/OOF/key 및82개 입력·의존 hash를 봉인했다. 동일 corrected3fold의49/79/53사례, 78h 정점별 간격과 episode 분리, 완료된 이전 fold만 쓰는 기준선 router의 target-ready 시각을 대조했다. 새 router는0fit이다. 훈련 시 원본/바람 결측 복제본 중량은 각각0.5, 이미 바람이 없는 행은 원래 중량을 유지해 target와 원본 행별 총 중량을 보존했다.

## 실행·검증 범위

| 구분 | 결과 |
|---|---|
| historical 실제 학습 | 3fold × 2arm × 1seed = 6fit, 274.203초 |
| 자원 | CPU2threads, GPU0, 30분 cap 내 종료 |
| focused synthetic / Ruff | 9 PASS / PASS |
| 독립 산식 QA | 222 checks PASS, failed0; pooled·모든 사전지정 slice 재계산 |
| 저장 모델 fresh-process replay | PID8500→10224, 6모델/2,172예측 최대오차0 |
| 공식/test/sample/hidden/CSV/upload/외부자료 | 모두0 |
| 이번 기준 모델 **빈 모델 폴더 재생성 검사** | **미수행** — historical saved-model replay와 다름 |
| 최종 제출 준비 | 이 증강 후보는 개선 없음. 기존 기준선 보존. 전체 최종 번들 검증 완료 아님 |

[독립 QA](independent-qa.json), [fresh-process replay](fresh-process-replay.json), [사전등록](preregistration.md), [봉인 preflight](preflight.json). 실행 중 runner/config/seed/분할/가림 강도/혼합비 변경, 추가 재시도는 없었다. 합성 unknown-category 방어 테스트의 Pandas 미래 변경 경고1건은 예상된 invalid-input 검사에서만 발생했다.

## 다음 판단

새 바람 증강 full fit은 하지 않는다. 별도 명시 승인된 `p3_clean_regeneration_20260905_v4`에서 현재 clean 기준선만 **새 폴더의 빈03_model→배포 source 재처리→6historical+2full CatBoost/3router→별도 프로세스 local05_answer**로 검증한다. 이는 실패 후보의 재튜닝이 아니라 기존 기준선의 독립 재생성 점검이며 이 보고서와 완료 상태를 구분한다. 이전 실행의 model/cache/CSV를 삭제하거나 runtime 입력으로 복사하지 않는다.
