# P2 boundary residual bridge terminal audit

작성일: 2026-08-29 KST

기준 커밋: `de1392076f15e3d08b6ab361760950eba880ddad`

실험: `p2_boundary_residual_bridge_20260829_v1`

판정: `NO_GO_CONTRACT_LEAKAGE`

family 상태: `CLOSED_NO_RETRY`

## 결론

`P2_BOUNDARY_RESIDUAL_BRIDGE_V1`의 고정 outside-flank 계약은 공식 hidden target을 읽지 않고는 세 historical block에 적용할 수 없다. 따라서 내부 first/last 72시간을 관측 flank로 바꾸거나 block을 줄이는 사후 재해석 없이 즉시 fail-closed 했다.

bridge fit, cubic smoothstep, 기존 projector, RMSE·bootstrap·cosine·p99 gate는 실행하지 않았다. 예측 행·prediction 파일·모델·CSV도 만들지 않았다.

## 충돌한 고정 flank

| historical block | side | 필요한 outside 72h KST | 공식 hidden target과 충돌 |
|---|---|---|---|
| `2025_jul_aug` | right | `[2025-09-01 00:00, 2025-09-04 00:00)` | 전체 충돌 |
| `2025_nov_dec` | left | `[2025-10-29 00:00, 2025-11-01 00:00)` | 전체 충돌 |

공식 hidden target interval은 `[2025-09-01 00:00, 2025-11-01 00:00)` KST이며 target layer 2·3·4의 temp/psal은 읽기 금지다. 필요한 여섯 flank 중 두 개가 이 구간에 포함된다.

## 실행 영수증

- one-shot contract audit 실행 수: `1`
- bridge fit 수: `0`
- data path open: `0`
- source observation row read: `0`
- official hidden target row read: `0`
- test_index/sample/submission row read: `0`
- prediction/model/CSV 생성: `0`
- contract reinterpretation: `false`
- 내부 first/last 72h 사용: `false`
- metric gate 평가: `false`
- family retry: 금지

수치 gate는 계약에 그대로 보존했다: pooled ΔRMSE `<= -0.0020°C`, 2024 Sep-Oct `<= -0.0015°C`, 최소 2/3 block 개선, worst block `<= +0.0005°C`, worst layer `<= +0.0010°C`, KST-day bootstrap CI90 upper `< 0`, absolute correction-axis cosine `<= 0.30`, correction p99 `<= 0.15°C`. 입력 계약이 먼저 실패했으므로 모든 metric check는 `null`이다.

## 재현

```powershell
.venv-p1\Scripts\python.exe -m ruff check scripts/run_p2_boundary_residual_bridge_20260829_v1.py scripts/qa_p2_boundary_residual_bridge_20260829_v1.py tests/test_p2_boundary_residual_bridge_20260829_v1.py
.venv-p1\Scripts\python.exe -m pytest tests/test_p2_boundary_residual_bridge_20260829_v1.py
.venv-p1\Scripts\python.exe scripts/run_p2_boundary_residual_bridge_20260829_v1.py --execute
.venv-p1\Scripts\python.exe scripts/qa_p2_boundary_residual_bridge_20260829_v1.py
```

runner와 QA 산출물은 write-once다. 이미 terminal JSON이 존재하는 동일 경로에서 runner나 QA를 다시 실행하면 실패하며, 재실행이 허용되지 않는다.
