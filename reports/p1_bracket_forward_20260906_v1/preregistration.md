# P1 bracket forward — 사전 계약 (2026-09-06 KST)

가설은 clean B의 기존 80열에 공개 수온의 구간 양쪽 경계·내부·복귀 특징 27열만 추가하는 것이다. O, B 학습 recipe, earlier-inner threshold/policy 선택 알고리즘은 보존한다. v4 T–S·먼 flank·depth 변경을 결합하지 않는다. 결과를 보기 전 고정한 창은 6/24/72시간, 외부 flank 각 1시간이며 입력 의존성 최대 ±37시간이다. 시간 gap과 정점/층 경계는 넘지 않는다.

## 실제 실행 전에 확정한 경계

- v5 KST H1_2024 warm-up; H2_2024/H1_2025/H2_2025 세 forward fold. 21일 purge와 양성 run 시작 fold 귀속·경계 run train 전체 제외를 그대로 사용한다. inner는 outer 시작−21일을 끝으로 직전 60일, inner train은 다시 21일 앞에서 끝난다. 표본 부족 시 날짜를 바꾸지 않고 중단한다.
- primary는 calendar H1_2025 pooled F1, 전체 forward F1·known/unseen station-layer·정점/층·유형 TP/FN·양성 run 경계/내부·정상 FP는 부표다. 기존 source audit의 107,125/208,093 unseen 행(51.48%)은 삭제하지 않는다. 정확한 inner 지원은 fit 전에 새 집계한다.
- 학습 자료는 `P1_DATA_DIR/train.csv` 하나이며 SHA `20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2`. 공식 입력·이전 모델/답안·외부 관측·Public 역산값 접근 0. 배포 train 중 이미 노출된 기간의 retrospective development 평가이지 fresh confirmation이 아니다.
- 새 2개 폴더/독립 PID에서 canonical all-train O 경로를 두 번 실행한다. CPU2는 이번 자원 계약이며 과거 CPU4와 동일 수치라고 가정하지 않는다. 첫 4,096개 정렬 train-feature probe는 **in-sample 결정론 진단 전용**이며 F1/quality를 계산하지 않는다. 모델 hash·feature matrix hash·probe 예측이 일치해야 OOF로 진행한다. 이 full 모델은 OOF에 쓰지 않는다.
- 그다음 control O/B의 inner/outer 12fits를 모두 봉인한 후 B bracket inner/outer 6fits를 수행한다. 총 최대20fits, CPU2/GPU0, 실행 전체 wall cap90분. 첫 full-fit 시간은 별도 pilot 비용 없이 실제 시간 근거로 쓴다. 예측/모델이 봉인되기 전 후보 성능을 공개하지 않는다. 성능 기반 조기 중단·추가 seed·threshold 강제·재시작은 없다.
- 평균 primary F1 개선이면 후보를 보존한다. paired KST-day bootstrap 2,000회/seed20260906, CI90·개선 비율은 불확실성 표이며 0.8 hard gate 또는 공식 점수 향상 확률이 아니다. 자동 제출/승격은 없다.

## full O 차이의 읽기 전용 원인 분리

과거 v5와 canonical의 CPU4/seed/package/code/train SHA는 같았고 full B package는 정확히 같았지만 O package는 달랐다. canonical은 새 프로세스에서 full O→B, encoder 1회/공유인 반면 v5는 inner B→O 후 full O→B를 같은 프로세스에서 실행하고 encoder를 재적합했다. 이는 가능한 실행 상태 차이이지 원인 확정이 아니다. 이번 두 fresh full-O 실행은 **같은 신규 명령의 결정론**만 분리하며 과거 답안 exact 복원·별도 머신/패키지 재현을 주장하지 않는다. 이전 checkpoint나 공식 답안 값은 읽지 않는다.

## 입력 의존성 및 합성 검증

기존 feature builder의 partition 전역 depth median 출력은 clean wrapper가 train-only 통계로 덮어쓰고 plateau 세 열은 이미168h로 cap된다. 실제 모델 입력80열은 canonical과 합성 byte/value 동등성을 확인했다. conservative ±15일 바깥 관측 변형은 고정 train stats 아래 중심행 feature를 바꾸지 않았다. bracket bank는 ±37h sentinel과 실제 gap/다른 정점 변형으로 검사했다.

**Hysteresis는 low-probability run 전체 의존이므로 유한 21일 상한이라고 주장하지 않는다.** Root가 수락한 계약은 metadata에서 train/inner/outer 행을 먼저 고르고 각 허용 partition 내부 전체 context로 stats/encoder/features/rules/decoder를 실행하는 것이다. 배제 partition sentinel이 어떤 stage에도 유입되지 않고 outer 관측·label 변형이 train stats/encoder/matrix를 바꾸지 않는지 검사했다. 모델 자체를 함께 바꾸는 bounded decoder는 추가하지 않는다.

focused pytest 12 PASS(3.99s), Ruff PASS. 초기 Ruff B023는 즉시 실행되는 diagnostic lambda의 loop variable을 명시적으로 bind하여 정정했다. 성능 fit 전 코드 변경이며 이후 변경하지 않는다.

| 봉인 파일 | SHA-256 |
|---|---|
| runner | `dc111189f3f58a8f01ec59634ee6bb51ed63928399efe592fc032d5317828e93` |
| config | `cfdb92b798a9ec36095b62bb1ddb58fb3c293ec0ab02c53fbea7d4579f797c04` |
| tests | `e5062c5bcf0a69f9f689e8056d354a43a1b1c206a0d328836104a703f0d4b97d` |

```powershell
.\.venv-p1\Scripts\python.exe scripts/run_p1_bracket_forward_20260906_v1.py --execute
.\.venv-p1\Scripts\python.exe scripts/run_p1_bracket_forward_20260906_v1.py --qa
```

기존 디렉터리·lock·src를 바꾸지 않으며 local aggregate result와 별도 PID replay QA를 남긴다. 모델/OOF/probe는 artifact에만 저장하고 Git·공식 CSV·upload는 하지 않는다.
