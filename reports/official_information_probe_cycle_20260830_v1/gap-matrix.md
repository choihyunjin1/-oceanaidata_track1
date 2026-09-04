# 공식 정보 probe gap matrix

| 문제 | 오늘 공식 확인 | 유지 후보 | 남은 핵심 gap | 다음 고가치 probe |
|---|---|---|---|---|
| P1 | G 제거 `-0.004519 F1`; S 제거 표시상 0; G+S 제거는 G 제거와 동일 | 기존 champion `0.833548` | I 단독 기여와 G×I interaction, Private 운송 | I 제거 / G-only / anchor+GI2-only 3개를 동시 동결 |
| P2 | bin17-only `0.430194` 새 최고; bin18-only `0.431267` | bin17-only | bin17 공식 최적 강도와 Private 안정성 | bin17 strength 3점을 점수 전에 공동 동결해 이차곡선 확인 |
| P3 | S/I 제거 모두 약 `+0.0037–0.0039m` 악화; 세 station 기여 양수 | uniform alpha=.425 `0.575233` | station별 최적 강도와 Private 안정성 | G/I/S 각각 추가 강도점 1개씩 공동 동결 |

## 닫힌 재시도

- P1: 오늘 S 제거 결과를 본 뒤 S threshold를 세분화하는 적응형 제출.
- P2: Gaussian 방향 positive blend; bin18 nearby strength 미세탐색.
- P3: uniform alpha 인근 재탐색, 이전 lead-split/lead-continuous recipe 반복.

## 보존해야 할 미확인 사항

- 공식 Public aggregate는 row-level loss와 hidden label을 노출하지 않는다.
- 작은 P2 개선은 표시 정밀도보다 크지만 매우 미세하므로 Private 승격을 보장하지 않는다.
- 최종 모델 패키지는 선택 답안의 정확한 예측을 재현해야 하며 아직 제출하지 않았다.
