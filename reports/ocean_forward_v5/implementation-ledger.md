# Ocean forward v5 — 계약·합성 검증 완료, 실제 지원 감사 대기

상태는 **CONFIG_FROZEN_PENDING_TRAIN_ONLY_SUPPORT_AUDIT**다. 이 문서는 모델 실행 준비 완료나 새 성능을 인증하지 않는다. 원문 로드맵을 그대로 구현한 것이 아니라 현재 문제별 계약을 존중하여 P1/P3 chronological, P2 계절 primary로 수정한 평가안이다. 기존 v4 결과를 재채점하거나 새 계약으로 소급 재분류하지 않았다.

## 이번에 고정한 범위

- P1: KST H1_2024 warm-up, 이후 3개 반기 forward, 21일 purge. 정확 10분 양성 run은 year 경계에서 끊지 않고 시작 fold에 단일 귀속하며 purge를 가로지르는 run은 train에서 통째 제외한다. 주평가는 run 소유명이 아니라 **행 시각 H1_2025 pooled F1**이다. warm-up 소유 run의 OOF는 만들지 않는다.
- P2: KST B1 2024-05/07, B2 2024-07/09, B3 2024-09/11, B4 2024-11/2025-01, B5 2025-04/06, B6 2025-06/08, B7 2025-08/09, B8 2025-11/2026-01의 반열림 8블록과 양측 7일 purge. **B3 자연 공개 T5 조건의 seasonal pooled RMSE**를 primary로 명명한다. 전체 pooled, 각 block의 마지막 17일 T5 temp+psal joint outage 구간 및 whole-fold testmatched 성적은 별도 지표다. 문서 유래 0.29 또는 중복 composite primary는 사용하지 않는다. B3의 마지막 17일은 10월 15일부터다.
- P3: UTC 2024Q1 warm-up, 이후 5분기 forward, 78시간 purge 및 같은 station/episode train 제외. 비가중 pooled RMSE가 primary이며 6리드가 모두 필요하다. onset weighted RMSE·greedy는 **NOT_ENABLED**다. episode ID는 caller의 train-only metadata를 요구하며 새로운 episode 생성 규칙을 여기서 추정하지 않는다.
- 공통: paired cluster bootstrap 2,000회, seed 20260906, 5/95 분위 CI, 동률은 개선 아님. P1 KST 일자, P2 KST 2024-01-01 기준 7일 bin, P3 station/episode 단위다. F1은 TP/FP/FN, RMSE는 SSE/N을 cluster별 사전 집계해 재표집한다. pooled 평균 개선이면 후보를 보존하지만 자동 승격·제출은 없고 bootstrap 0.8 hard gate도 없다. 위험·worst block·slice는 별도로 보고해야 한다.

## 검증과 실행 명령

```powershell
.\.venv-p1\Scripts\python.exe scripts/ocean_evaluation_contract_v5.py
.\.venv-p1\Scripts\python.exe -m pytest tests/test_ocean_evaluation_contract_v5.py
.\.venv-p1\Scripts\python.exe -m ruff check scripts/ocean_evaluation_contract_v5.py tests/test_ocean_evaluation_contract_v5.py
```

- 2026-09-06 KST, focused pytest **45 PASS / 0.92s**, Ruff PASS. 첫 실행에서는 44 PASS와 합성 fixture의 Pandas 3 정수 열에 inf 대입 오류 1건이 있었으며, fixture만 float로 바로잡은 뒤 변경된 테스트 범위를 검증했다. 성능 결과에 따른 조정은 없다.
- 빈 입력, 중복 key/정규화 시각, 결측·비유한 값, key/order 불일치, binary F1, 불균형 fold의 pooled RMSE, warm-up, run/year/purge 경계, P2 joint mask의 원본 불변성, P3 78h·episode·6리드, bootstrap 동률/지원 부족/재현성을 합성 값으로 검사했다.
- 사전집계 bootstrap과 동일 seed 원시 행 재표집을 F1/RMSE toy에서 대조했다. CLI는 config 구조와 hash만 확인하며 실제 지원 감사 상태는 `NOT_RUN`이다.

## 봉인 파일 SHA-256

| 파일 | SHA-256 |
|---|---|
| `configs/evaluation/ocean_forward_v5.json` | `ca6f610aa087c5d2f4c3c25e7af487178c2d344b527d987f0571ddeb178b8a5b` |
| `scripts/ocean_evaluation_contract_v5.py` | `36581e8ac358463670b368e2da4ded98d5479c8a98c7c3917774c0dd68eb6e39` |
| `tests/test_ocean_evaluation_contract_v5.py` | `c5609eec1c7d65945f54db83c37926ccfffa09cdb49fb4a216b23dad8e5abf99` |

## 남은 경계

실제 train-only 지원 수·purge 후 충분성·mask 후 특징 재계산·feature context 발자국·P3 episode 생성 계보는 **미감사**다. 단순 purge만으로 무제한 ffill/bfill 또는 전체 segment 특징의 안전성을 증명하지 않는다. P2 target-layer joint mask와 전체 특징 재계산은 이후 adapter의 별도 의무이며 T5 mask helper만으로 완료되지 않는다. Bootstrap은 관측된 cluster의 기술적 재표집일 뿐 공식 향상 확률이나 선택 편향 해소를 보장하지 않는다.

이번 작업의 배포 원자료·공식 입력·hidden·기존 모델/답안 열람 0, fit 0, 공식 CSV 생성 0, upload 0, Git 변경 작업 0. 새 설정·순수 helper·합성 테스트·이 원장만 작성했다. 다음 단계는 새 성능 fit이 아니라 **train-only 0-fit 지원 감사**이며, 지원 부족을 성능 기반 날짜·gate 변경으로 덮지 않는다.
