# Evidence gap matrix — v16 final

| 문제 | 이번 검증축 | 확인된 증거 | 결정적 공백 | 닫힌 행동 | 다음 유효 표적 |
|---|---|---|---|---|---|
| P1 | asynchronous multi-layer latent-state GP subset scan | ±10분 peer 관측 자체는 많지만 cell별 정상 calibration과 양기간 coverage가 동시에 부족 | Q2/Q3 정상 block `19/74 < 100`; 통과 cell `2/15`, `8/15` | 같은 cell threshold 완화·재실행 | station-level partial pooling을 별도 사전등록 |
| P2 | train-only regime veto | 직전 correction의 계절 운반 문제를 label 재사용 없이 차단하려 시도 | 첫 outer fold의 두 block으로 LOBO+자기참조 금지 동시 충족 불가 | held/self label 재사용, 두-block 기준 완화 | 3개 이상 prefix block 또는 label-free reference |
| P3 | Hs²-weighted directional memory | 10 value coverage 100%, source split·58-case shadow support 정상 | source pooled/3 station/18·24h가 모두 비개선 | mean-direction proxy 사후 수축·station 제외 | 실제 directional spread·partition source |
| 공통 | one-shot truth-late stopping | 공식 접근·CSV·업로드 0, hash QA PASS | historical surface의 반복 노출 | 동일 계약 결과 기반 재튜닝 | 새로운 정보원과 새 validation surface |

## 이 사이클에서 닫힌 축

- P1: 현재 cell별 84일 calibration과 최소 100 정상 block을 유지한 async latent-state GP exact 계약.
- P2: 두 training block만으로 구성한 LOBO train-only seasonal regime veto exact 계약.
- P3: 평균파향과 Hs²만으로 만든 20-feature directional-memory increment exact 계약.
