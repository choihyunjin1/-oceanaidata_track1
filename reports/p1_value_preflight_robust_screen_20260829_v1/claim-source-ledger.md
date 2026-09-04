# Claim-source ledger

| 주장 | 근거 | 유형 | 확신 | 한계 |
|---|---|---|---|---|
| 정적 preflight가 후보를 10/10으로 통과시켰다 | `artifacts/.../preflight.json` | 로컬 기계 산출물 | 높음 | 신규성은 registry 범위 안에서만 판단 |
| 저충실도 후보는 Q3 -0.0008115, Q4 +0.0012711, pooled +0.0000426이다 | `artifacts/.../run/result.json` | 로컬 기계 산출물 | 높음 | retrospective, single seed |
| 새로 추가한 23행은 true 0이다 | `result.json`의 Q3/Q4 candidate_added_* | 로컬 기계 산출물 | 높음 | historical truth 기준 |
| full fidelity로 승격하면 안 된다 | `postrun_gate.json` | 사전 고정 규칙 적용 | 높음 | station concentration은 unavailable이며 fail-closed |
| GroupDRO는 prespecified group의 worst-case 성능을 목표로 한다 | Awasthi et al., PMLR 2024 | 1차 문헌 | 높음 | 본 실험은 full GroupDRO가 아님 |
| 조기중단과 multi-fidelity는 비싼 후보 평가를 줄이는 정당한 전략이다 | Li et al., JMLR 2018; Wu et al., PMLR 2020 | 1차 문헌 | 높음 | 낮은 fidelity와 full ranking의 상관이 필요 |
| 불변학습은 가정 실패 시 ERM을 이기지 못할 수 있다 | Rosenfeld et al., arXiv 2021 | 반증 문헌 | 중간~높음 | IRM에 직접 해당하며 모든 DRO에 동일하지 않음 |
| time-frequency alignment는 다음 구조적 연구축이 될 수 있다 | He et al., PMLR 2023 | 1차 문헌 + 전이 추론 | 중간 | P1 적용성·허용성은 아직 검증 전 |

## 외부 출처

1. https://proceedings.mlr.press/v237/awasthi24a.html
2. https://www.jmlr.org/papers/v18/16-558.html
3. https://proceedings.mlr.press/v115/wu20a.html
4. https://proceedings.mlr.press/v202/he23b.html
5. https://arxiv.org/abs/2010.05761
