# 2026-08-28 P1/P2/P3 병렬 실행 결과

## 결론

- **P2만 기술적 공식 probe 자격이 있음:** seasonal OAS shrinkage `alpha=0.40` 후보는 생성 및 독립 QA를 통과했다. 다만 예상 개선폭이 `+3점` 승격선에 못 미치므로 전략 기본값은 `HOLD`이다.
- **P1은 공식 제출 보류:** change-point rescue와 direct interval set 두 구조 모두 fresh Q2에서 exact no-op가 선택되었다.
- **P3은 공식 제출 보류:** Chronos-2 zero-shot 및 fold별 inner-best LoRA가 기존 incumbent보다 명확히 악화되었다.
- 이번 실행에서는 공식 업로드를 수행하지 않았다.

## P1

### F1-aware change-point rescue

- 판정: `NO_GO_LOCAL_BLIND_GATE`
- Q2 proposal 540개 중 유익 proposal 19개였으나 precision 분리가 실패했다.
- 선택 arm: `ZERO_ADD_NO_OP`
- Q3/Q4 출력 변경: 0행
- pooled F1: incumbent와 동일한 `0.9068037200`
- 공식 test/sample/submission 접근·생성·업로드: 0

Q3/Q4는 이 실행에 대해서만 blind였고 과거 노출 이력이 있으므로 fresh sealed holdout으로 해석하지 않는다.

### Direct interval set Transformer

- 판정: `NO_GO_Q2_EXACT_NO_OP_SELECTED`
- CUDA feasibility smoke는 통과했으나 fresh Q2에서 과검출이 발생했다.
- anchor F1: `0.867676`
- model-union F1: `0.186985` (`-0.680691`)
- recall 변화: `+0.104097`
- 추가 TP/FP: `559 / 44,623`
- 추가 precision: `0.012372`
- 선택 arm: `ZERO_ADD_NO_OP`
- Q3/Q4 및 공식 데이터 접근: 수행하지 않음

해석: 누락 회수 능력은 있으나 정상 구간 objectness calibration이 실패했다. 다음 독립 구조는 고재현 proposal generator와 disjoint hard-negative verifier/abstention을 분리하는 2단계 모델이다. 이번 결과의 threshold 사후 튜닝은 하지 않는다.

## P2

- 기술 판정: `PASS_OFFICIAL_PROBE_ELIGIBLE_PENDING_EXPLICIT_UPLOAD_APPROVAL`
- 전략 판정: `HOLD_BY_MINIMUM_3_POINT_POLICY`
- 후보: `C:\Users\cedis\Downloads\해양 해커톤 제출용\20260828_P2_SEASONAL_OAS_TS40_PROJECTED_READY\P2_submission.csv`
- SHA-256: `6e28ddb8d78c0969e5104d7efbe28e1762f51e80d759fceb86cdef52baa29b96`
- 파일 크기: `1,284,060` bytes
- 행 수: `26,061`
- 값 범위: `16.302557–29.081289 C`
- alpha=0.20 대비 변경 행: `19,531`
- alpha=0.20 대비 RMS 변화: `0.091405 C`
- PAVA 적용 행: `9,244`; 완성본 재투영 최대 차이 `0.0`

공식 alpha=0.10/0.20 기록과 실제 제출 벡터 기하를 사용한 조건부 추정:

- 예상 RMSE 중심: `0.448627`
- 보수 구간: `[0.431042, 0.465548]`
- 현 alpha=0.20 RMSE `0.483661` 대비 중심 예상 점수 개선: 약 `+0.4396점`
- 보수 구간 상단 기준 예상 점수 개선: 약 `+0.2273점`

이는 동일한 전체행 공식 scorer라는 조건의 수학적 추정이며 점수 보장이 아니다. alpha=0.40은 특히 layer 4에서 공격적이다. 예상 개선폭이 사용자의 최소 `+3점` 기준에 미달하므로 일반 제출은 보류하고, 당일 소멸 예정 기회를 장기적인 로컬-공식 상관관계 측정에 쓰기로 명시적으로 결정할 때만 probe한다.

## P3

- 판정: `TERMINAL_NO_GO`
- Chronos-2 환경, CUDA, PEFT LoRA smoke: 통과
- exact three-way 공통집합: `179 cases / 1,074 rows`
- candidate RMSE: `0.900416 m`
- persistence RMSE: `0.865161 m` (`candidate +0.035254 m` 악화)
- incumbent RMSE: `0.782774 m` (`candidate +0.117642 m` 악화)
- fold별 incumbent 대비 변화: `+0.067676`, `+0.148410`, `+0.112698 m`
- incumbent 대비 case bootstrap CI90: `[+0.07279, +0.15928] m`
- incumbent 대비 day bootstrap CI90: `[+0.06584, +0.16487] m`
- blind prediction SHA-256: `08a750fb464bd78cf7a92c40cb734a5fd9075202c754ecfa56bf9cbde379fcba`
- prediction seal SHA-256: `f6d40d2b79785411f74be4847e32beb2ae952c0628825a4c0a9b167a45fa8b23`
- 독립 QA: `PASS`

작은 inner fold의 개선은 세 outer window에서 일반화되지 않았다. 추가 Chronos step 확대는 중단하며, 다음 구조 예산은 ERA5 masked-pretrain 계열처럼 다른 backbone에 사용한다.

## QA

- P1 관련 테스트: `6 passed`; Ruff 통과
- P2 전용 독립 QA: 통과; P2 복원 테스트 `4 passed`; Ruff 통과
- P3 독립 seal/metric/hash QA: 통과; 관련 테스트 `5 passed`; Ruff 통과
- 공식 업로드: 0
- Git commit/push: 수행하지 않음
