# P3 numeric lead / v5 forward — 성능을 보기 전 고정한 계약

실행 계약의 시점 근거는 첫 fit 전 생성한 config와 `seal.json`이다. 이 설명 문서는 첫 학습 쌍이 진행 중인 동안 봉인 계약을 풀어 쓴 것이며, 사후 성능을 보고 사전등록 시각을 소급한 기록이 아니다. 작성 시점까지 성능값은 미열람이다.

가설은 single CatBoost에서 `lead_h`를 문자열 범주형 대신 실제 시간 수치형 float로 입력하는 것이다. station은 범주형으로 유지한다. 기존 clean single/multi residual target, threshold case weight, 591개 특징, single/multi0.5 등가 출발값, fixed loss-router(Ridge10/temperature2/strength0.5), 12/18/24h persistence shrink0.2를 유지한다. 새 phase/onset 특징이나 결과 기반 후처리는 추가하지 않는다.

## 평가 및 자원

평가 계약은 `configs/evaluation/ocean_forward_v5.json`의 P3다. Q1_2024는 warmup이며 OOF를 만들지 않는다. Q2_2024부터 Q2_2025까지 정확히5개 forward fold, 미가중 pooled SSE/행수의 제곱근이 주지표다. validation은 dense17,267 anchors/103,602행이다. 과거 sparse181case 실험 및 공식200case와 같은 평가 분포라고 하지 않는다. raw20min hs>=1.5 연속 run이 station-episode이며, gap/low/missing에서 끊고 미래 target 결측 여부로 episode를 자르지 않는다.

| fold | base train anchors | validation anchors | meta-fit 허용 prior OOF anchors |
|---|---:|---:|---:|
| Q2_2024 | 7,057 | 1,051 | 0 |
| Q3_2024 | 7,912 | 2,628 | 819 |
| Q4_2024 | 10,665 | 5,388 | 3,572 |
| Q1_2025 | 15,974 | 6,708 | 8,881 |
| Q2_2025 | 22,808 | 1,492 | 15,715 |

base는 fold 시작78h 이전 anchor만 쓴다. meta도 이전 fold 소속이라는 이유만으로 전부 쓰지 않는다. 같은78h 시각/동일 station-episode 제외 규칙을 적용해 현재 context 시작 이전에 모든 과거 target이 준비되게 한다. 이전 OOF의 경계232/107/186/60 anchors는 각각 제외한다. 각 arm의 새 single OOF와 공유된 새 multi OOF로 router를 별도 재적합하며 옛 router 계수를 읽지 않는다.

CPU2threads/GPU0, 정확한 기본 실행 최대15 CatBoost fits(categorical single5+multi5+numeric single5)와 router8fits다. categorical single+CPU multi를5fold 모두 완료해 새 CPU 기준선을 만든 다음 numeric single5fits를 실행한다. GPU clean 성적을 CPU 기준선에 승계하지 않는다. 새 CPU multi는 `devices` 제거와 task_type CPU로 정하고 나머지 clean hyperparameter를 유지한다. seed는 clean recipe 첫3개20260816/17/18과 미리 정한 연속 확장20260819/20이다. 양 arm의 같은 fold seed가 같다.

첫 fold categorical single와 multi는 예정된 실제 fit이자 resource pilot이다. 이2fits는 그대로 재사용한다. 전체 예정15fits 시간은 첫single/firstmulti 실측에 train-anchor 규모비 `sum(train_counts)/7057`와20%여유를 적용한다. 추정이90분을 초과하면 성능을 보기 전에 `RESOURCE_STOP_NO_RESTART`로 종료하고 root에 보고한다. 각 fit의 CPU iteration callback도 같은90분 wall deadline을 준수한다. 숫자형 arm 성적이나 첫fold 이득을 보고 epoch/seed/fold/CPU/GPU/예산을 바꾸지 않는다.

## 누출·지원 검사와 중복 지문

새 feature prepare를 수행한 `p3_clean_regeneration_20260905_v4`의 exact-hash train 특징/anchor/열목록3개만 재사용한다. 과거 model/OOF/예측/답안은 입력하지 않는다. source train_wave/train_atmos SHA와 prepare provenance를 다시 연결했다. 배포 wave118,152행/atmos130,896행을 이용해24,360anchors의 current와6targets를 원본에서 직접 대조했고, fold×station 첫/중간/끝45context의591개 특징을 다시 계산해 최대오차0을 확인했다. 미래/48h밖 관측 교란은 context 특징에 영향0이었다. 전체 cache를 독립 재구현한 증명은 아니며45context 수치 대조+고정 builder SHA+전체target/key 검증이다.

이 가설은 이전 wind-only LGBM dropout, event_phase_probe, 사후 lead-continuous residual calibration과 다르다. 과거 외부 ERA5 transfer 설정에 numeric lead가 존재하지만 비적격 외부 계보이므로 기준/가중치/계수/예측을 이어받지 않는다. 검색으로 찾지 못한 다른 동등한 실험이 절대로 없다는 주장까지 하지는 않는다.

## 판정·검사·범위

미가중 pooled RMSE가 평균 개선하면 후보를 보존한다. CI90/개선 resample 비율은 서술적이며0.8 hard gate나 자동 제출 승인으로 쓰지 않는다. worst-quarter/정점/lead/wind-observed slice 악화는 따로 보고한다. station high-run cluster는 같은 기상 폭풍/정점 간 상관을 완전히 제거하지 못하며 독립 폭풍 수로 부풀리지 않는다. onset/greedy는 `NOT_ENABLED`다.

실행 전 synthetic12개와 Ruff PASS. categorical/numeric single 및 CPU MultiRMSE 각각3iteration native 학습·저장·재생3fits를 포함하며 실제 historical fit과 분리 계상한다. pandas 새 StringDtype 때문에 `dtype == object`였던 테스트 단언을 실제 문자열 dtype 검사로 고쳤다. 이는 최초 봉인 전 synthetic-only 정정이며 학습 코드/성능 계약 변경은 아니었다. 독립 QA용 추가 synthetic4개/Ruff도 PASS다.

실제 완료 시 저장한15backbone/8router를 별도 process에서 다시 읽어 같은 OOF를 exact 재생하고, 별도 QA runner가 source target·SSE/분모·fold/78h/episode·model hash·lead cat indices를 검산한다. canonical result는 `result.json`, replay는 `fresh-process-replay.json`, QA는 `independent-qa.json`이다. 존재하지 않는 단계를 PASS로 해석하지 않는다.

공식 입력/hidden/CSV/upload/full-fit/Git 작업은0이다. 이번 historical 비교의 save/reload PASS는 빈 full-model 폴더에서 공식답안까지의 재생성 PASS나 독립 제출 패키지 PASS가 아니다. 평균 개선하더라도 full materialization은 이 작업 범위 밖이다.
