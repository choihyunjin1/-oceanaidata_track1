# P1·P2·P3 병렬 로컬 실행 사이클

작성일: 2026-08-29 KST

기준 커밋: `de1392076f15e3d08b6ab361760950eba880ddad`

상태: **세 문제 terminal, 신규 공식 후보 0건**

공식 test/sample/submission 값 접근·CSV 생성·업로드: 모두 0건

## 결론

이번 사이클은 세 개의 값싼 반증을 병렬 실행했고, 세 후보 모두 다음 단계로 승격하지 않았다.

| 문제 | 실험 | 핵심 결과 | 판정 |
|---|---|---|---|
| P1 | `p1_window_phase_consistency_20260829_v1` | replay identity PASS; phase q99 `0.003728`, XOR `29`; default e150 대비 고정 평균 증분 약 `+0.000242 F1` | `NO_GO_Q2_PREFLIGHT` |
| P2 | `p2_boundary_residual_bridge_20260829_v1` | 필요한 6개 flank 중 2개가 공식 hidden 기간과 충돌; 금지값 0행 읽고 종료 | `NO_GO_CONTRACT_LEAKAGE` |
| P3 | `p3_perfect_future_wind_oracle_20260829_v1` | perfect-future-wind treatment가 pooled RMSE를 `+0.001340m` 악화; oracle gate 7/7 실패 | `CLOSE_PREDICTED_FUTURE_WIND_AND_MOS_FAMILY` |

실패를 단순 폐기하지 않고, 닫힌 정보축과 다음 연구의 제약으로 보존한다. 이번 결과로 window-phase consistency 학습, P2 two-sided target-flank bridge, P3 predicted-future-wind MOS를 동일 계약으로 다시 실행할 이유가 사라졌다.

## P1 — window phase

Q2 model state가 남아 있지 않아 RTX 5090으로 exact 3-seed e150을 약 55분 재현했다. 세 seed의 기본-view decoder probability, boundary probability, proposal은 기존 sealed Q2와 모두 bitwise 동일했다.

사전 gate 결과:

- `q99(abs(p0-p256)) = 0.0037284570`, 기준 `>=0.05` 실패
- proposal XOR `29행`, 기준 `>=50행` 실패
- fixed-average anchor-union F1 `0.8679175181`
- anchor-only F1 `0.7922755741`, 따라서 사전 문서의 anchor 대비 `Delta F1=+0.0756419440`는 통과

마지막 수치는 phase 효과로 해석하면 안 된다. 동일 Q2 grid의 기본 e150 anchor-union F1은 `0.8676757359`이므로, fixed-average의 **실제 기본 e150 대비 증분은 약 `+0.0002417822 F1`**이다. 대부분은 기존 e150이 anchor에 추가한 효과다. 이 비교는 terminal 판정을 바꾸기 위한 사후 gate가 아니라, `+0.0756`을 phase gain으로 오독하지 않기 위한 독립 해석이다.

계약대로 paired-view 5-epoch warm-start는 0회, Q3/Q4 metric 접근도 0회였다. center-weighted overlap-add가 window-position nuisance를 이미 충분히 약화했다는 쪽으로 family를 닫는다.

## P2 — boundary bridge

outside two-sided 72h target residual을 세 historical block에 동일하게 요구하면 다음 두 flank가 공식 hidden target 기간에 들어간다.

- `2025_jul_aug` 오른쪽: 2025-09-01~09-04 KST
- `2025_nov_dec` 왼쪽: 2025-10-29~11-01 KST

내부 first/last 72h로 바꾸면 다른 실험 계약이 되고 validation truth를 입력으로 쓰게 된다. 따라서 계약을 바꾸거나 금지 target을 읽지 않고 fail-closed 했다. source observation, official hidden/test/sample/submission 읽기는 모두 0행이며 bridge fit·smoothstep·projector·metric gate도 0회다.

이는 성능 실패가 아니라 **데이터 계약으로 식별 불가능한 가설**이라는 terminal 결과다.

## P3 — future-wind oracle

historical frozen KMA OOF 교집합 179 cases, 1,074 rows에서 실제 미래 `u/v`를 treatment에 넣었다. 3/6/9/12h는 exact no-op이고 18/24h만 활성화했다. 동일 folds, 78h embargo, control-only ridge alpha를 treatment에 공유했다.

| 항목 | 결과 |
|---|---:|
| control pooled RMSE | `0.782538046m` |
| treatment pooled RMSE | `0.783877901m` |
| pooled Delta RMSE | `+0.001339855m` |
| 18h Delta RMSE | `+0.005679471m` |
| 24h Delta RMSE | `+0.001391757m` |
| whole-case bootstrap CI90 | `[-0.001793608,+0.004342590]m` |
| 개선 fold / station | `1/3`, `1/3` |
| worst station×lead | `+0.011200734m` |

oracle gate 7개가 모두 실패했으므로 conditional future-wind forecast와 MOS는 0회 실행했다.

한계: 179 cases 중 미래 벡터가 여섯 lead 모두 유효한 사례는 123개였고, 결측은 사전 고정한 zero-delta persistence로 처리했다. 따라서 완전 관측 future wind 전체에 대한 물리적 상한을 증명하지는 않지만, 이 exact deployable contract를 승격할 근거는 없다.

## 자원 사용과 병렬화

- P1: RTX 5090, VRAM 약 22GB, 3-seed e150 replay 약 55분
- P2: 정적 계약 감사와 focused QA, 학습 0회
- P3: CPU historical oracle, conditional 모델 학습 0회
- P2·P3 정적/CPU 작업과 P1 GPU replay를 병렬 배치해 자원 충돌을 피했다.

## 독립 QA

- P1: Ruff PASS, focused pytest 35 PASS, seal/hash/inventory/metric replay PASS
- P2: Ruff PASS, focused pytest 4 PASS, independent QA PASS
- P3: Ruff PASS, focused pytest 4 PASS, independent aggregate replay 19/19 PASS
- 합계 focused pytest: 43 PASS
- tracked 경로의 prediction CSV·checkpoint·raw row: 0건
- 공식 업로드: 0건

## 폐쇄한 축과 다음 연구 경계

1. P1 window-phase consistency 학습은 이번 exact gate로 닫는다. `+0.0756`을 phase gain으로 재해석하지 않는다.
2. P2 outside two-sided target-flank bridge는 official hidden 기간을 요구하므로 닫는다. 내부 flank로 이름만 바꿔 재개하지 않는다.
3. P3 predicted-future-wind + frozen-KMA MOS는 oracle 실패로 닫는다. 과거 lag 추가나 future-wind predictor를 별도 이름으로 재실행하지 않는다.
4. 다음 연구는 이 세 family 밖의 독립 정보축이어야 하며, 구현 전에 같은 방식의 정적 계약 감사와 값싼 oracle을 먼저 둔다.

## 재현 명령

one-shot runner의 `--execute`는 이미 소비됐으므로 재실행하지 않는다. 안전한 검증은 다음 focused tests와 정적 QA로 제한한다.

```powershell
.\.venv-p1\Scripts\python.exe -m ruff check scripts/run_p1_window_phase_consistency_20260829_v1.py scripts/verify_p1_window_phase_consistency_20260829_v1.py scripts/run_p2_boundary_residual_bridge_20260829_v1.py scripts/qa_p2_boundary_residual_bridge_20260829_v1.py scripts/run_p3_perfect_future_wind_oracle_20260829_v1.py scripts/qa_p3_perfect_future_wind_oracle_20260829_v1.py tests/test_p1_window_phase_consistency_20260829_v1.py tests/test_p2_boundary_residual_bridge_20260829_v1.py tests/test_p3_perfect_future_wind_oracle_20260829_v1.py
.\.venv-p1\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_p1_window_phase_consistency_20260829_v1.py tests/test_run_p1_incumbent_preserving_mstcn_asrf_v2.py tests/test_p2_boundary_residual_bridge_20260829_v1.py tests/test_p3_perfect_future_wind_oracle_20260829_v1.py
```
