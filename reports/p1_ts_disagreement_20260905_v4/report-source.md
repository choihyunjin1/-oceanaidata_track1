# P1 T–S 불일치 특징 분리 실험

결론: **NO_INTERNAL_GAIN**. 새 T–S 특징 19개를 추가해도 기존 clean control보다 pooled F1이 낮았다. 이 후보의 full 학습·공식 CSV 생성은 진행하지 않는다. 실패는 이 고정 특징 묶음과 현재 반복 사용한 historical 평가에 한정하며, 모든 염분 특징이 무효라는 뜻은 아니다.

| 표면 | 행 수 | 기준 F1 | 후보 F1 | 후보−기준 |
|---|---:|---:|---:|---:|
| intact Q2/Q3/Q4 pooled | 421,032 | 0.851174240 | 0.842101756 | −0.009072484 |
| fragmented | 408,789 | 0.847734942 | 0.847662346 | −0.000072596 |
| 기준 모델 재생성 검사 통과 여부 | 별도 검사 | v5 새 학습→답안 PASS | 과거 답안 exact 복원 FAIL | 아래 별도 보고서 참조 |

## 설계와 검증

- 배포 train의 776,706행만 사용했다. 이전 clean control의 exact-hash O/B 모델과 비교하며 모델 계보·입력 특징·split·calibration의 일치를 확인했다.
- B 분류기에만 T/S 변화 비율, 염분 변동, rolling 상관·회귀잔차 등 19개 특징을 더했다. 기존 80개 특징, depth 계약, 원래 O 분류기와 decoder 후보군은 유지했다. epsilon은 각 train에서만 산출했다.
- Q2/Q3/Q4에서 inner/outer 각 1 fit, 총 6 CPU fits. inner에서만 사전고정 정책을 선택했다. 총 실행 576.657초. 추가 full fits 0, 공식 입력 접근 0, CSV 0, upload 0.
- [고정 설정](../../configs/experiments/p1_ts_disagreement_20260905_v4.json), [사전등록](preregistration.md), [결과 원장](result.json).
- 합성 테스트 12 PASS, Ruff PASS. [별도 프로세스 QA](policy-replay-qa.json) 47/47 PASS: 새 inner/outer 모델 6개 해시, 6개 평가 파일 해시, 키·라벨, 후보 확률, O와 후보를 조합한 최종 decoder bits, pooled confusion count를 재계산했다. 학습 PID 35268, QA PID 38312.

## 재생성과 한계

[기준선 빈 모델 폴더 재생성 v4](../p1_clean_regeneration_20260905_v4/report-source.md)는 새 Q4 inner 학습에서 원래 정책과 다른 정책을 골라 중단했다. 이 실험의 model replay PASS는 기준선을 처음부터 다시 학습하는 검사나 독립된 최종 패키지 PASS를 대신하지 않는다. 기존 sealed result의 당시 pending 필드는 덮어쓰지 않고 이 보고서에서 후속 검사로 연결한다.

[v5 기술정정](../p1_clean_regeneration_20260905_v5/report-source.md)은 canonical CPU4/순서를 맞춰 Q4 정책을 다시 도출하고 4 fits→169,011행 새 답안→별도 PID exact replay를 완료했다. 다만 과거 답안과 양성 개수가 6,505→6,395로 달라 과거 공식 점수나 historical control의 full 산출물과 완전히 같은 모델이라고 표시하지 않는다. 독립 QA 21/21은 이 분리된 주장을 검증한 것이지 과거 SHA 복원 PASS가 아니다.

반복 사용한 historical 개발 표면이며 fresh confirmation이 아니다. 공식 점수 예상치는 산출하지 않았다. 정상 행에서 유형별 F1=0은 양성 분모가 없는 특성이지 정상행 분류 성능 0이라는 뜻이 아니므로, 유형별 진단은 recall과 FP를 구분해서 읽어야 한다.
