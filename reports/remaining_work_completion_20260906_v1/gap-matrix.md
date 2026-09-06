# 완료 범위와 남은 공백

| 항목 | 현재 증거 | 상태/남은 일 |
|---|---|---|
| P1 bracket 공식 채점 | 9031, 06:39 F1 .785944 | 완료 |
| P1 장기 saved inference | ZIP 새추출/새PID29.234s exact | 완료 |
| P1 빈 모델 재학습 | 4fits + QA/replay753.030s, 동일9031 | 완료; 새OS/두번째CPU2 wholecold는 미검증 |
| P1 범위/셀 분리비교 | 독립190QA, 주평가0/음수 | 완료·비승격; 고정0–35룰의 FP0 입증 아님 |
| P2 crossfit 수치정정 | 추가48fit 완료·QA1117 | 완료; crossfit 비승격 |
| P2 선택후보 full/portable | CPU4fit·QA17/73·새PID/ZIPreplay | 완료 |
| P2 공식 matched comparison | .489080→.475174℃ | 완료; 보존CUDA .455143 못넘음 |
| P3 numeric/hmax 비교 | 15+10backbone, 독립149/134QA | 완료; 둘다CI0포함 |
| P3 새 후보 wholecold | 배포 원자료부터12+5fit/QA/answer/replay1611.919s | 완료; 새 전체 GPU학습1회 |
| P3 저장모델 장기 추론 | 실제 ZIP새추출/새PID/6.088s exact | 완료; historical OOF/cache/probe 없이 full모델3개 |
| P3 새 후보 공식 제출 | 08:08 RMSE .608184 /23.680619점 | 완료; 기존 .607183보다 나빠 기존기준 유지 |
| 최종 모델 첨부/잠금 | 실행 안 함 | 별도 사용자 결정 필요 |
| 전체 새OS·인터넷차단 의존설치 | 같은 PC 환경만 검증 | 미검증; Python network hook은 OS차단 증명 아님 |
| 새 구조/HPO 전부 | 이번 범위 아님 | zt_real/context 등 모든 옛 설계가 수행됐다는 주장 금지 |

완료되거나 종료된 실험을 단순히 재시도하지 않는다. technical failure는 artifact 보존 후 새 명시적 설계/권한이 필요하고, 과학적 성능 미달은 결과로 남긴다.
