# P2 domain-invariant vertical-curvature v9 technical failure

## 결론

`p2_domain_invariant_vertical_curvature_20260901_v9`은 **0-fit technical INVALID**다. 과학적 GO/NO_GO가 아니다. 두 후보의 첫 `Ridge.fit()` 전에 전처리가 fail-closed했고, prediction commitment·metric·공식 입력 접근·CSV·upload는 모두 0이다.

원인은 2024-05-01~2024-08-24의 고정 training window에 layer 8 관측이 없어 다음 네 공개층 feature 열이 전부 NaN이었던 것이다.

- `temp_offset_l8`
- `psal_anomaly_l8`
- `depth_offset_l8`
- `nominal_offset_l8`

presence indicator 열은 이미 0이므로, 후속 `v9r1`은 후보·split·alpha·blend·gate·winsor를 유지하고 이 네 열만 deterministic zero로 처리한다. 기존 v9 artifact와 lock은 보존하며 v9를 재실행하지 않는다.

## 무결성

- config SHA-256: `6c26b704355f85415fca185482fb0ef48eb4201ce8b08326cabbed2a8bbfaffe`
- runner SHA-256: `e417c2aa0ecc8d2d5c5fdc274f9e3641017db47ac3f5ea9f3d2139723a200463`
- attempt lock SHA-256: `0d529ee3e1907fd6c3e98694b4e28bf53ecdd131ce9bae38e9280551099bb754`
- artifact files: `attempt_lock.json` 1개
- model fits: `0`
- predictions/metrics: `0/0`
- official/test/sample/baseline/score/query/hidden rows: `0`
- submission CSV/upload: `0/0`
