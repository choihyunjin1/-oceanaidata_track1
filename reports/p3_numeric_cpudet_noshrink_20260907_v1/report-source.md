# P3 CPU 3-seed no-shrink: 내부 평균 개선 후보 준비 완료

결론: 현재 CPU 모델에서 사전등록한 장기 리드 shrink 제거만 적용했을 때 내부 pooled RMSE가 **0.681321563 → 0.677450331 m (−0.003871233 m)**로 개선됐다. 별도 PID QA 및 실제 ZIP 추출 후 노트북 추론/재생은 PASS이다. 공식 점수는 이 후보에 아직 없으며, 최종 지정도 하지 않았다. 기존 base fresh_cold_2는 그대로 실행 중이고 그 완료 증거를 이 결과로 대신하지 않는다.

## 설정·대조군·출처

- 새 fit 0, 신규 데이터 학습 0. CPU 3-seed numeric single/multi 및 train-OOF loss router 모두 기존 completion_1의 동결 모델을 사용한다.
- 변경은 12/18/24h 최종 0.2 지속성 혼합을 제거(w=0)한 것 하나이다. 3/6/9h·모델·feature·clip·router·split·seed·학습량은 그대로다.
- config: `configs/experiments/p3_numeric_cpudet_noshrink_20260907_v1.json`, SHA `31b02bae8639b0b4a0bef7b0eeefbd015a907f53d0691ed258bfd3dcf8d2684f`.
- current CPU OOF SHA `bede4edff8ce229d66c6209bce4227f85d63b1c065113a39100b12a0ea40ec16`, 103,602행. 과거 GPU OOF 결과를 전용하지 않았다.
- 기준 답안은 모델에서 다시 계산했을 때 SHA `2015b38750d357630d5b2e9eee32807d2961ce752e35eb2b454ee16e579dda56`와 일치했다. 과거 답안 값은 읽지 않았다.
- 사전등록 후 내부 평가 PID 34592, 독립 QA는 저장된 historical router에서 raw routed 예측을 다시 계산했다. 역변환과 최대차 <1e-12, 원래 shrink를 재적용하면 원본 OOF와 exact 일치한다. 행의 6리드를 하나의 독립 표본으로 취급하지 않았다.

## 내부 결과와 위험

RMSE Δ는 후보−기준이며 음수가 개선이다. 기준 SSE 48,091.9523451907, 후보 SSE 47,546.9931632454 / N=103,602.

| 표면 | 기준 RMSE m | 후보 RMSE m | Δ m |
|---|---:|---:|---:|
| 전체 | 0.681321563 | 0.677450331 | −0.003871233 |
| Q2 2024 | 0.721701005 | 0.734290768 | +0.012589762 |
| Q3 2024 | 0.684257830 | 0.674326393 | −0.009931436 |
| Q4 2024 | 0.672256505 | 0.683426318 | +0.011169814 |
| Q1 2025 | 0.676527787 | 0.661408893 | −0.015118894 |
| Q2 2025 | 0.700574738 | 0.690683437 | −0.009891301 |
| 낮은 초기 파고 hs0<1.7 | 0.617122930 | 0.627044834 | +0.009921904 |

station×episode 951블록, paired bootstrap 4,000회, seed 20260907: **P(Δ<0)=0.914, CI90 [−0.008524918,+0.000733607] m**. 3정점 모두 pooled 개선, 장기 리드별 Δ는 12h −0.001823109 /18h −0.006429569 /24h −0.011165627. 최악 단일 episode Δ +0.177335085m(24행)도 숨기지 않는다. 5fold 중 2fold와 저파고는 악화한다. 평균 점수 개선 목표에 따라 위험을 기록하되 자동 탈락시키지 않았다.

이미 노출된 retrospective OOF에서의 개선이지 새 holdout 증명이나 예상 Public 점수가 아니다. 공개 점수는 어떤 계수·후보 선택식도 유도하지 않았다. CI90은 0을 포함한다.

## 산출물·재생

정본 폴더: `C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_numeric_cpudet_noshrink_v2/`.

- 답안 `ANSWER/submission_p3_numeric_cpudet_noshrink.csv`: 1,200행, 39,869 bytes, SHA `70761affca4d3fc6f1d24ae53467e5b185b23465926ebb4b851f0300872cddbd`.
- SOURCE_ONLY.zip: 107,017 bytes, SHA `c3aed055873bed601397073be4610fb06fa085021122f3656d65c1939faa6e78`.
- SAVED_MODELS.zip: 17,125,495 bytes, SHA `f0451b9cfe7a3042b9848679d9cfd654e7800e958d6a6e3374bd56332ecfdc7b`.
- local infer PID 16792, 별도 replay PID 8060: exact. 실제 ZIP 추출한 독립 커널/자식 PID 31316, 41960의 RUN_INFERENCE/REPLAY_INFERENCE도 exact이다. 단기 600행 그대로, 장기 600행만 변경했다. schema/key/order/unique/finite 0..30m 및 입력·모델 before/after 해시 PASS.
- source에는 빈 모델부터 학습→기존 QA→새로 학습한 7 deployment model 해시 등록→추론 절차가 포함된다. 이번 addon에서는 그 전체 학습을 실행하지 않았으며 새 fit은 0이다. SAVED_MODELS는 동결 7 deployment 모델과 열 이름·계보만 포함한다.
- 원본 README·노트북도 unbounded source manifest에 포함되어 있으므로 byte 그대로 보존했다. addon 지침은 상위 README.md, SOURCE_ONLY/START_HERE_NOSHRINK.md, RUN_TRAINING.ipynb, RUN_INFERENCE.ipynb이다. 예전 TRAIN/PREDICT 파일을 addon 진입점으로 선택하지 않는다.
- 최초 v1 패키징 검사에서 manifest가 요구하는 root 문서가 빠져 inference 전에 실패했다. 기존 v1을 보존하고 v2에 원문 root pins를 추가했다. 모델·계수 변경이나 추가 fit 없이 고쳤으며 최종 v2를 추출 재검증했다.

## 미검증·금지 작업

Base 완료는 29개 검증 prefix+7 backbone+5 router이며 uninterrupted fresh cold 자체가 아니다. 별도 fresh_cold_2 완료/동일성은 부모 작업에서 확인 대기. 새 프로세스 replay와 새 학습 재현을 혼동하지 않는다. 공식 6시간 규칙의 일반 모델 적용 범위, 제출 마감·첨부 제한은 미확인이고 실측은 우리 PC의 시간일 뿐 준수 증명은 아니다. hidden 접근·외부 데이터·공개 점수 역산·추가 학습·업로드·삭제·최종 지정·commit/push 모두 0이다.

기계 판독 정본은 이 폴더의 `result.json`과 `independent-qa.json`; 각 단계 상세 시각/소요시간/경로는 local root의 receipts에 있다.
