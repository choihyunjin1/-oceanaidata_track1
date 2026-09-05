# 추가 검증 및 정보 확보용 공식 비교

## 최종 결론

P1은 답안 재현 검증 뒤 실제1회 제출했으나 **Public F1 0.767370 /27.150461점**으로 당일 control보다 **F1 −0.023363 /−0.620939점**이다. 새 최고점이나 교체 후보가 아니다. 기존 제출·모델·데이터는 보존했다. 수심 계약뿐 아니라 선택된 완성 정책이 함께 바뀐 비교이므로, 수심 특징 자체가 나쁘다는 단독 인과 결론은 내리지 않는다.

P2는 첫0-fit 단서에 대해 같은 당일3-seed control과 비교하는 **별도9-fit 학습·내부테스트·재현까지 완료**했다. 가을 intact와 추가 가을결측에서 모두 악화하여, 첫seed의 이득이3-seed 평균에서는 재현되지 않았다. **P2 추가 제출0**이다. P3도 새로 검증해 배포할 비-no-op 후보가 없어 추가 제출하지 않았다. 이 턴의 실제답안 제출은P1 1회이며 새최고점은 확보하지 못했다.

## 실행 근거와 제출 영수증

- [실행 전 판단과 P2 후속 단계 승인](decision-before-submission.md)
- [P1 공식 영수증](official-receipt.json): 2026-09-05 22:10 KST 접수(화면은 분 단위), 문제 카드와 갱신된 제출관리의 지표·점수 일치. P1 오늘 남은1/3.
- [P1 후보·학습계보·재현 보고서](../p1_depth_information_submission_20260905_v3/report-source.md): 169,011행/schema/키/순서/이진값/hash PASS, 별도PID 전체CSV byte-exact PASS, synthetic12/Ruff PASS.
- [P2 첫0-fit 검증](../p2_missingness_conditional_validation_20260905_v3/report-source.md): 기존6모델 재사용, 69,850개 동일 키, 가을26,273행. 11 synthetic/142-check QA PASS. Root도 집계156개에 대해 `sqrt(SSE/n)`을 재계산하고 runner/config SHA 일치를 확인했다.
- [P2 추가3-seed 검증](../p2_missingness_conditional_3seed_20260905_v3/report-source.md): 새 historical6+full3=9fit, 전체365.671초/실제fit합277.984초. 17 synthetic/Ruff PASS,845-check QA PASS. 별도독립reviewer는 집계·분모·decision373검사PASS. 학습PID26920과다른PID27144에서full6모델의128개 공개학습입력 재생 exact PASS. 이는 전체공식CSV나clean-machine재학습검증이 아니다.

## P2 첫 검증의 뜻

고정 규칙은 공개 layer5 수온 또는 염분이 비유한이면 R, 아니면 C다. 목표layer2/3/4 가림은 trigger가 아니다. 첫seed 기준 가을 intact RMSE는0.465330→0.464325℃로 작게 좋아지지만 전체는0.896731→0.900894℃로 나빠진다. 가을7/14일 결측 스트레스에는 개선이 있지만 여름/겨울 위험도 함께 남는다.

인위적 가을3일 결측에서 공개 수온지원2개 미만인4행이 생겨 전체scenario를 미채점했다. 몰래4행을 버려 점수를 좋게 만들지 않았으며, 이를 실제공식 입력의 결함으로 주장하지 않는다. 겨울지정구간은 원래layer5가 결측이므로 새독립 outage 실험이 아니다. 과거 관측·라벨 재사용이므로 새확증적 holdout도 아니다.

원래 `RESEARCH_ONLY_NOT_READY`를 바꾸지 않고, 추가2seed×3fold와R full3fit을 별도ID로 승인·실행했다. 결과는 다음과 같다.

| 3-seed 고정 비교 | 평가행 | C RMSE ℃ | conditional RMSE ℃ | 변화(낮을수록 좋음) |
|---|---:|---:|---:|---:|
| intact 가을 primary | 26,273 | 0.488284326 | 0.489067844 | +0.000783517 |
| intact 전체 | 69,850 | 0.859249914 | 0.860501412 | +0.001251499 |
| 새가을7/14일 지원완료episode | 9,035 | 0.772421618 | 0.803223321 | +0.030801703 |

사전고정한 정보가치조건이 false이므로 `NO_INTERNAL_SUPPORT_FOR_CONDITIONAL`로 종료했다. 이는 임의의작은허용치 때문에 단서를 버린 것이 아니라, 두 주요근거의 개선방향 자체가3-seed에서반전된 결과다. R단독 전체RMSE0.821973617은C보다좋지만 가을0.498233450은C보다나쁘며, 이결과를보고새로운R전체배포정책을선택하지않았다. 공식점수 개선량은 미산정이며 새로운P2공식성적을주장하지않는다.

P2 배포adapter는 준비만하고 공식입력/CSV/업로드0으로남겼다. 15분은계획예산으로hard-cap구현은아니며 실제약6분6초로예산내완료했다. 기존모델·봉인·실패기록은변경하지않았다.

## 범위와 한계

Data Analytics의 validate-data/analyze-data-quality 기준으로 기술QA, 내부성능, 공식점수, 전체재학습재현을 분리했다. 현재 P1의 별도프로세스 추론은 완전재학습/인터넷차단 clean-machine 검증을 대신하지 않는다. 공식Public 성적은 고정완성후보의 비교만이며 정답·분할소속·계수·threshold 역산은 하지 않는다.

공식 공지는 최종 마감일2026-09-07을 명시하나 정확한시각은 이번조회에서 확인하지 못했다. 일일3회 제한과 전체마감을 혼동하지 않는다. 이 턴의 최종모델잠금, Git stage/commit/push는0이다.
