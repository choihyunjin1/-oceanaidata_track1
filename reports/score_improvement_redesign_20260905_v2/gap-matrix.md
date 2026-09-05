# 결정에 영향을 주는 잔여 불확실성

| 공백 | 현재 상태 | 결정 영향 | 해결 수단/담당 |
|---|---|---|---|
| P1 year-key가 공식 하락의 주원인인가 | 계약 불일치 확인, 인과 기여 미확정 | 해결 유망도 높지만 상승 약속 불가 | P1-A paired 내부 학습, raw missing/unseen-year 분리 |
| current-depth 반올림의 raw 이상·센서 노이즈 영향 | 미측정 | 단순 fallback으로 모든 문제 해결 불가 | 그대로 남긴 모든 평가행과 depth-quality slice 비교 |
| P1 최고 조합의 router/GI 출처 | 일부 원장 존재, 전체 재현 미완료 | 강한 합법 부품까지 잃을 위험 | 최초 선택 근거/일반 규칙 재생성 대조; 모호한 부품만 차단 |
| P1 상반기·신규 정점 대표성 | historical 반복 노출, 2024 정점 제한 | 단일 fresh 검증 주장 불가 | Q2와Q3/Q4를 분리표시, 추가Q1은 fresh라 부르지 않음 |
| P2 ℃ MSE가 일반화에 유리한가 | 손실 불일치 확인, 효과 미측정 | objective2×2가 짧고 가치 높은 실험 | 고정15fit 상한, 정규화항 유지 |
| P2 가을1fold와 실제 outage 대표성 | 다른 계절이 pooled 개선 지배 | pooled delta의 Public 전이 추정 금지 | 가을 intact primary + 가을 T5 outage stress + 전체 pooled |
| P2 normalized 입력의 절대상태 정보 손실 | 코드상 확인, 오차 원인 미확정 | 새 대형구조보다 단일 tree 대조 | absolute profile/기존 temporal feature의 bounded 9fit |
| P2 adaptive calibration용 누출 없는 inner OOF | 확보 여부 실행 전 확인 필요 | 전체OOF LOFO로 대체 금지 | 추가 nested fit 비용을 명시하거나 고정50:50만 대조 |
| P3 평균 음의 bias의 시간 안정성 | pooled 진단만 있음 | 편향값을 공식에 바로 더하면 안 됨 | 과거-only fold2/3의 1자유도 보정 |
| P3 평가 episode 표집법 | context/lead 조건만 알려짐 | first-eligible를 사실로 강제하지 않음 | train-only 사건가중 대조, 시각 역식별0 |
| GPU/6시간 최종 재현 | 이전 실측 존재, 새 recipe 미측정 | 세 heavy job 동시 실행 금지 | P2→P3→필요시P1 GPU 큐, full source replay 측정 |
| 오늘 남은 횟수/최종 마감 | 19:29 receipt2/2/1, 마감 공지 충돌 이력 | 즉시 제출/시간 여유 단정 금지 | 실행 시 현재시각·인증 홈페이지 재확인 |
| 새 후보의 예상 공식 점수 | 계산할 검증된 일반화 모델 없음 | 과장된 +점 예측 금지 | 현 baseline points 표시, 내부 metric 별도, 제출 후 실제 점수 |

## 조사 종료 판단

실행 순서를 바꿀 주요 근거는 확보했다. 남은 핵심은 문헌 검색이 아니라 고정된 작은 실험의 결과다. 새 모델 목록을 더 늘리지 않고 실행 설계로 넘긴다. 이 문서는 연구계획이며 미해결 사항을 PASS로 기록하지 않는다.
