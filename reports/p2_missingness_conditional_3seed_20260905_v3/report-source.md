# P2 3-seed 결측 조건부 보완 — 첫 seed의 개선이 평균에서는 재현되지 않음

## 결론

**이번 조건부 후보는 공식 제출하지 않는다.** 봉인한 동일 규칙을 3-seed 성분 평균으로 검증하자 가을 intact와 추가 가을 outage가 모두 C보다 나빠졌다. 결과는 `INTERNAL_VALIDATION_COMPLETE / NO_INTERNAL_SUPPORT_FOR_CONDITIONAL`이며, 수치·해시 QA PASS와 과학적 개선은 구분한다. 추가 규칙·seed·epoch 탐색, 유리한 첫 seed 선택, 공식 입력, CSV, 업로드는 모두 하지 않았다.

| 동일행 비교 | 행 수 | C 3-seed RMSE ℃ | conditional 3-seed RMSE ℃ | 차이(후보−C) ℃ |
|---|---:|---:|---:|---:|
| 가을 intact primary | 26,273 | 0.488284326 | 0.489067844 | +0.000783517 |
| 전체 intact | 69,850 | 0.859249914 | 0.860501412 | +0.001251499 |
| 자연 결측 trigger | 17,165 | 0.277258754 | 0.292627227 | +0.015368473 |
| 새 가을 7/14일 지원완료 outage | 9,035 | 0.772421618 | 0.803223321 | +0.030801703 |
| 새 여름 3/7/14일 outage | 10,339 | 1.662164583 | 1.655276813 | −0.006887771 |
| 겨울 지정구간(추가 결측 없음) | 8,250 | 0.251083203 | 0.265528478 | +0.014445275 |
| 점수 산정 가능한 추가 지정구간 전체 | 27,624 | 1.117146442 | 1.121412659 | +0.004266217 |

지표는 `sqrt(sum(SSE)/sum(n))`이며 fold RMSE의 단순 평균이 아니다. 숫자의 단일 근거는 [result.json](result.json), 독립 재계산은 [independent-qa.json](independent-qa.json)이다. 현재 공식 C의 `0.455143℃ / 27.622418점`은 다른 공식 평가면의 기록이며 위 내부 수치를 공식 예상점수로 변환하지 않는다. 이 후보의 공식 기대점수는 미측정이다.

## 왜 첫 seed와 달라졌는가

[이전 0-fit 검증](../p2_missingness_conditional_validation_20260905_v3/report-source.md)의 C/R 첫 seed 비교에서는 가을 intact가 `0.465330→0.464325℃`, 추가 가을 outage가 `0.971357→0.787438℃`였다. 그러나 배포 비교 대상은 이미 3-seed C다. 같은 9,035행에서 C의 3-seed 평균은 `0.772422℃`로 첫 seed C보다 훨씬 강했고, R 평균은 `0.803223℃`였다. 따라서 첫 seed 결과만 선택하면 현재 배포 기준선 대비 효과를 잘못 판단할 수 있다.

이는 세 seed가 모든 무작위성에 대해 충분하다는 증명이나 유의성 검정은 아니다. 반복 노출된 historical labels에서 이번 정확한 평균/규칙 조합의 개선 근거가 재현되지 않았다는 결론이다. R standalone은 pooled intact `0.821974℃`로 C보다 좋지만, 배포관련 가을은 `0.498233℃`로 C보다 나쁘다. pooled가 좋아 보인다는 이유로 승인된 conditional을 R 전체로 바꾸거나 가을 primary를 교체하지 않았다.

## 봉인 계약과 실제 비용

- 규칙: **공개 layer5의 temp 또는 psal이 nonfinite이면 R, 그 외에는 C**. 목표 layer2/3/4의 결측·값은 trigger에 들어가지 않는다. 각 arm의 `20260901/20260902/20260903` 세 성분을 산술 평균한 뒤 같은 trigger로 결합한다.
- 특징/학습: 기존 target-free v23 DeepSets, normalized Huber와 원본별 총량 보존 uniform data weights. 기존 domain-weighted normalized-Huber input-gradient penalty와 계수 `0.01`, blockmask, 60 epochs, lr/weight decay/배치/seed를 바꾸지 않았다. DataLoader workers 0, CPU 1 thread, GPU 단독 배정.
- 분할: 기존 가을/여름/겨울 3 fold와 양측 7일 purge를 그대로 재사용. 복원 문제의 full-history 방식이며 causal prefix 예측으로 부르지 않는다. 각 fold의 train-row/augmentation/weight 영수증이 기존 R 첫 seed와 정확히 같다.
- 재사용: **C historical 9 + R historical 첫 seed 3 + C full 3 = 15 저장 모델**, 원래 학습 runner/config/dependency와 모델 해시 대조.
- 신규: **R 나머지 두 seed × 세 fold = 6 historical fits + R full 세 seed = 3 fits**, 합계 **9 fits**. full training은 허용된 배포 관측의 동일 eligible 166,268행이다.
- wall runtime **365.671초**, fit runtime 합 **277.984초**. 15분은 사전 계획·운영 예산이며 runner의 hard timeout은 아니다. 이번 학습·평가는 그 계획 안에 종료했다.
- 이전 0-fit 검증은 16.297초·신규 backbone 0이었다. 기존 A/B 및 이 검증을 재실행하거나 결과를 덮어쓰지 않았다. A/B의 9+3 fits는 이번 신규 9 fits에 중복 계상하지 않는다.

실행 계약: [config](../../configs/experiments/p2_missingness_conditional_3seed_20260905_v3.json), [sealed runner](../../scripts/run_p2_missingness_conditional_3seed_20260905_v3.py), [preregistration-seal.json](preregistration-seal.json). 완료한 ID의 `--execute`를 재실행하지 않는다.

## 지원제약·위험을 숨기지 않는 평가

이전과 정확히 같은 날짜의 10개 episode를 사용했다. 하나는 기존 14일 development이고 나머지는 달력으로 사전 지정한 3/7/14일 × 3계절이다. 새 masking이지 새 정답·독립 확증이 아니다.

가을 3일 artificial mask의 1,294행 중 **4행**은 공개 수온 지원이 두 개 미만이 되었다. 사전계약대로 전체 scenario를 `SUPPORT_BLOCKED`로 남기고 그 scenario의 점수는 산정하지 않았다. **4행을 삭제해 나머지 점수를 보고하지 않았으며, 전체 69,850개의 평가 key는 모든 scenario에서 보존했다.** 이것은 인위적 입력의 지원제약이지 실제 공식 입력 결함이나 과학적 성능 실패가 아니다. 따라서 지원완료 가을 outage 9,035행에는 이 미산정 1,294행이 포함되지 않는다.

겨울 intact 16,884행은 공개 T5/S5가 자연적으로 모두 결측이라 이미 R로 route된다. 겨울 세 지정구간은 추가적인 결측을 만들지 않는 **no-op**이다. 이를 세 개의 새로운 결측 intervention 성공/실패 증거로 세지 않는다. 자연 결측의 temp-only/psal-only/both별 분모와 오차는 독립 QA JSON에 따로 기록했다. nontrigger 52,685행은 C와 exact 동일하고 각 지원완료 scenario의 episode 밖 예측은 intact와 exact 동일하다.

## 독립 QA와 재현 범위

- 새 stage 6 synthetic + 기존 target-mask/onset/offset/lag-dependency 11 synthetic: **17 PASS**. 별도 배포 adapter synthetic **3 PASS**. Ruff PASS.
- [독립 QA](independent-qa.json): **845 checks PASS**. n/SSE/RMSE/bias/Δ, 3-seed 평균, OR route, old C/R exact replay, 지원제약, seed/fold/epoch/fit count, 원래 코드·모델 해시, zero-access 영수증을 대조했다.
- [fresh-process-replay.json](fresh-process-replay.json): 학습 PID와 다른 PID에서 C full 3/R full 3 모델을 로드해 **배포 관측으로부터 만든 공개 특징 128행**의 각 성분과 conditional을 exact 재현했다. 신규 fit 0.
- 이 128행 검사는 저장 모델의 독립 프로세스 수치 재현 증거다. **전체 공식 CSV의 재현 검사가 아니며, 처음부터 다시 학습했을 때의 bitwise 재현 검증도 아니다.** 공식 입력을 아직 열지 않았으므로 실제 공식 changed rows/score는 알 수 없다.
- 접근 증거는 runner의 source allowlist audit hook, 불변 해시, 입력/출력 영수증이다. 운영체제 전체 파일접근을 완전 계측했다고 주장하지 않는다.

| 산출물 | SHA-256 |
|---|---|
| runner | `232ce3d3ca602ac2a860476639fce89792394a59c071d223aaca74dbeb6bcf4f` |
| config | `075a294b924eeaed834af848a9f3fd51960f2458e134106d644c1a6ddfabe5cb` |
| result.json | `775d153d005fdc6833c2e8ebc525556b2eca48064a36e63edda30f911eab1305` |
| private historical predictions | `c7d68e1285dbace2e79a61258d5d04173fd5e01b7abfdb228e9bded2ed579023` |

## 배포 준비와 종료 판단

시간을 절약하려고 별도 [deploy adapter](../../scripts/run_p2_missingness_conditional_deploy_20260905_v3.py)와 [synthetic tests](../../tests/test_p2_missingness_conditional_deploy_20260905_v3.py)를 준비했지만 **seal/materialize/replay 모두 미실행**이다. 코드에는 root의 별도 공식 접근 승인, 정보조건 PASS, 독립 QA/replay PASS 뒤에만 공식 key를 읽도록 제한했다. C full 3으로 오늘 clean C CSV를 새로 재생성해 SHA가 동일한지 확인하고, nontrigger exact 보존/전체 26,061행/중복 SHA 배제/별도 PID 전체 CSV replay를 수행하도록 준비했다. 그러나 이번 정보조건이 충족되지 않아 그 단계로 진행하지 않았다.

root는 이번 결과를 확인하고 **추가 학습·규칙 조정 없이 종료, P2 공식 materialization/upload 0 유지**를 결정했다. 원래 0-fit 상태, A/B 결과, frozen models, lock은 모두 보존했다. 공식 기회를 소모하지 않았고, 현재 허용된 clean C를 이 후보로 교체하지 않았다. Git stage/commit/push도 실행하지 않았다.
