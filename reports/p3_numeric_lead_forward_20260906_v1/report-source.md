# P3 수치형 lead / v5 forward — 자원 제한 종료

## 결론

**`NOT_EVALUATED_RESOURCE_STOP`**. 첫 CPU baseline single/multi 2 fits의 실제 시간으로 고정된 전체 15-fit 예상이 **349.059분**이 되어, 사전등록한 90분 한도를 초과했다. 첫 fold 성능을 열람하기 전에 `RESOURCE_STOP_NO_RESTART` 분기로 종료했다. 수치형 lead 후보는 0 fit이며, 후보가 나쁘다거나 기존 모델보다 개선되지 않았다는 과학적 결론은 없다. 학습 재시작·예산/CPU/GPU/iteration/fold 변경은 하지 않았다.

종료 및 자원 산술 독립 QA **60/60 PASS**, focused synthetic **12+4 PASS**, Ruff PASS다. 이는 완성된 5-fold baseline/candidate 평가, fresh-process 예측 재생, 빈 full-model 폴더 재생성 검사가 통과했다는 뜻이 아니다. 그 단계들은 실행하지 않았다.

## 실제 수행량과 시간

| 항목 | 실측/상태 |
|---|---:|
| Q2_2024 categorical single, CPU 2 threads, 700 trees | 149.738281초 |
| Q2_2024 multi, CPU 2 threads, 1,200 trees | 1,612.555903초 |
| 두 fit/predict/save receipt 합 | 1,762.294185초 |
| attempt lock → resource terminal | 1,763.654350초 = 29.394분 |
| 완료 baseline backbone / 예정 baseline | 2 / 10 |
| 완료 numeric 후보 / 예정 후보 | 0 / 5 |
| 완료 router / 최대 router | 0 / 8 |
| full fits / GPU 사용 / 공식 입력 / 제출 CSV / upload | 모두 0 |

실행은 2026-09-06 KST 01:17:52.714982에 attempt lock을 소비했고 01:47:16.369332에 종료됐다. PID 38096 소멸을 확인했다. `progress.json`은 두 번째 fit 시작 당시의 완료 1개를 표시할 수 있다. 최종 `fit-receipts.json`의 2개와 `FAILURE.json`이 종료 상태의 근거다. `FAILURE.json` 및 nonzero exit의 TimeoutError는 사전등록된 **자원 추정 중단 분기**이며 모델 학습 오류나 성능 NO_GO로 바꾸어 해석하지 않는다.

고정 산식은 다음과 같다.

```text
train_anchor_scale = (7057 + 7912 + 10665 + 15974 + 22808) / 7057
                   = 9.12795805583109
15-fit forecast = 1.2 * train_anchor_scale * (2 * first_single + first_multi)
                = 20,943.5425779563 seconds = 349.059 minutes
cap             = 5,400 seconds
```

이는 train-anchor 규모에 선형 비례한다고 가정하고 20% 여유를 더한 **예상**이지 15 fits를 완료한 실측이 아니다. 각 receipt의 시간은 학습뿐 아니라 validation prediction과 model serialization을 포함한다. router와 나머지 전체 파이프라인 시간의 직접 실측을 대신하지 않는다. 첫 두 fit은 예정된 실제 baseline fit이었고 별도 재학습하지 않았다. 종료 전 생성된 메모리 내 첫-fold component predictions에 대한 점수 계산/비교를 수행하지 않았고 완성 OOF를 저장하지 않았다.

CPU-only full 학습→답안 전체의 6시간 충족 여부는 **미검증**이다. 과거 `p3_clean_regeneration_20260905_v4`의 약 27.57분은 multi GPU를 포함한 다른 평가/재생성 경로이므로 이 CPU-only v5 시간 또는 점수의 보증으로 승계하지 않는다.

## 고정한 가설과 확인된 입력 지원

가설은 clean single CatBoost의 `lead_h`만 문자열 범주형에서 실제 시간 float 수치형으로 바꾸는 것이다. 591개 특징, residual target, 기존 threshold case weights, clean single/multi 결합 및 사전고정 long-lead shrink 0.2, Ridge10 router 절차를 유지한다. 같은 fold의 새 CPU multi는 양 arm이 공유하고 router는 arm별로 이전 OOF 중 78h/episode 안전 조건을 만족하는 행만 재학습하도록 했다. 현재 후보 단계 전 중단되어 이 정책의 수치 비교는 없다.

v5 지원 검사는 Q1_2024 warmup 이후 정확히 5개 forward folds, 총 24,360 source-derived anchors 중 validation 17,267 anchors/103,602행을 확인했다. 주지표 예정은 **미가중 pooled SSE / 실제 행수의 제곱근**이었다. 과거 sparse 181-case 평가 또는 공식 query 분포와 동일한 평가라고 하지 않는다. raw 20분 hs>=1.5 연속 station-run이 episode이고 low/missing/gap에서 끊되 미래 target eligibility로 episode를 자르지 않는다. onset 및 greedy는 `NOT_ENABLED`다.

`preflight.json`은 정확한 source/cache/provenance hash를 연결하고 모든 anchor의 current 및 여섯 target을 배포 source에서 직접 대조했다. fold×station의 첫/중간/끝 **45 contexts**에서 591개 특징을 재계산한 최대 오차는 0이며 미래/48h 밖 관측 교란에 변화가 없었다. 전체 feature 행을 별도 구현으로 재생성했다는 주장은 아니다. meta 지원은 각 fold에서 0/819/3572/8881/15715 prior anchors이며 경계 232/107/186/60개를 추가로 제외한다. 이 메타 검증 지원 계산과 router 실제 학습 완료를 혼동하지 않는다.

## QA 범위와 미완료 단계

`resource-independent-qa.json`은 독립 PowerShell 검산으로 다음을 확인했다: 16개 입력 pins 및 runner/config/preflight seal, own prepared-anchor 및 저장 모델 2개 hash, fixed seed/tree/thread/CPU, 실제 fit count와 행수, 시간 산술·cap·종료 순서·PID 소멸, zero-access receipt, 완성 baseline/candidate/result/replay 부재. 모델 역직렬화·예측 재평가·원시 값 출력·새 fit은 수행하지 않았다. 접근 0은 execution guard/receipt와 정적 경로 검토의 증거이며 독립 OS 접근 감사라고 과장하지 않는다.

실행 전 main synthetic 12개와 Ruff가 통과했다. 독립 full-result QA helper의 synthetic 4개/Ruff는 첫 실제 학습 쌍이 실행 중일 때 통과했고 결과를 보지 않았다. main suite는 dtype 테스트 단언을 봉인 전에 정정하여 2번 실행했으므로 native CPU 3-tree synthetic 모델은 총 6 fits다. historical 2 fits 및 후보 0 fits와 별도 계상한다. 자세한 시점과 테스트 범위는 `code-qa.json`에 있다.

| 검사/산출물 | 최종 상태 |
|---|---|
| raw-derived source 지원·해시/45-context 특징 대조 | PASS |
| 최초 실제 CPU resource pilot | COMPLETE; 예산 초과 예상 |
| resource terminal 독립 산술/해시 QA | 60 PASS / 0 FAIL |
| 완성 CPU baseline 및 numeric 비교 RMSE/CI90/slices | NOT RUN |
| 완성 정책 saved-model fresh-process replay | NOT RUN |
| 기준 모델 빈 폴더 재생성 검사 | NOT RUN; 이번 scope 밖 |
| 공식 입력/답안 생성/업로드/기존 모델·예측 입력 | 0 |

현재 중단된 실행의 sealed runner/config/lock/model을 보존한다. 평가 결과가 없는 상태에서 새로운 arm, hard gate, 후처리 비율 또는 자동 제출을 선택하지 않는다. 후속 자원 정책이 별도 승인되기 전 신규 학습은 시작하지 않는다.

## 재현·탐색 안내

- 고정 계약: `configs/experiments/p3_numeric_lead_forward_20260906_v1.json`, 본 폴더 `preregistration.md`, artifact `seal.json`.
- 실행 경로: `scripts/run_p3_numeric_lead_forward_20260906_v1.py`. 이미 attempt를 소비했으므로 `--execute` 재실행 금지.
- 자원 판단: 본 폴더 `resource-pilot.json`, artifact `fit-receipts.json` 및 `FAILURE.json`.
- 최종 QA: 본 폴더 `resource-independent-qa.json`; 모든 파일 hash와 60개의 검사 이름/boolean을 포함한다.
- 전체 비교가 끝났을 경우를 위한 QA 코드: `scripts/qa_p3_numeric_lead_forward_20260906_v1.py`. 현재는 필요한 완성 OOF/result가 없으므로 실행하지 않았다.
- 연구 모델/anchors/attempt locks는 ignored artifact에 보존하며 최종 portable 제출 package나 Git 게시 대상으로 표시하지 않는다. 이번 lane에서 Git 쓰기는 0이다.

자원 판단 SHA-256: `4693f9283d90761d4e88e9b440e26eeedb5f96224b8780504ed0d329abe7a4f4`.
종료 receipt SHA-256: `3cef5ee4437855a0536e4d7ab1a0cf481740e681bc1cc5270664d97a338906fe`.
