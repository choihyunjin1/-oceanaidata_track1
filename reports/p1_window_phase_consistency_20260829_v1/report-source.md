# P1 window-phase consistency one-shot

기준 커밋: `de1392076f15e3d08b6ab361760950eba880ddad`

상태: **NO-GO — Q2 preflight fail-closed**

## 범위

- historical train/OOF 전용이다.
- 공식 test, sample submission, 기존 submission 값은 읽지 않는다.
- prediction CSV 생성과 업로드는 금지한다.
- 원본 데이터는 변경하지 않는다.
- 결과 기반 grid와 자동 rerun은 금지한다.

## Q2 alternate-tiling preflight

고정 recipe는 width 512, epoch 150, threshold 0.8, seeds 20260827/20260839/20260863이다. 기본 창은 2048행·stride 512이고 alternate view는 각 exact segment 앞에 invalid zero context 256행을 붙인 위상이다. 두 view 모두 center-weighted overlap-add를 사용한다.

과거 Q2 e150 probability grid는 sealed 상태지만 Q2 model state는 남아 있지 않다. 따라서 같은 Q2 training prefix에서 3개 seed를 epoch 150까지 replay하고, 기본-view decoder probability·boundary probability·proposal이 기존 sealed Q2 cell과 bitwise 일치해야만 alternate view를 유효한 frozen-e150 audit로 인정한다. 이 identity gate가 실패하면 Q2 truth를 intervention 판단에 사용하지 않고 paired-view 학습도 시작하지 않는다.

두 view의 정렬 probability와 고정 proposal/candidate를 NPZ로 먼저 seal/hash한 뒤 Q2 truth를 연다. 다음을 모두 만족해야 한다.

- `q99(abs(p0 - p256)) >= 0.05`
- proposal XOR `>= 50`행
- 고정 산술평균 probability와 boundary를 threshold 0.8 decoder에 한 번 적용한 anchor-union `Delta F1 >= +0.001`
- replay identity 3개가 모두 bitwise equal

## 통과 시 단일 warm-start

Q2가 모두 통과할 때만 Q3/Q4의 기존 frozen e150 checkpoint 6개를 사용한다. 각 seed/phase는 정확히 5 epoch, AdamW, constant LR `1e-5`, weight decay `1e-4`로 한 번만 진행한다. 두 view의 기존 supervised MS-TCN++/ASRF loss 평균에 Bernoulli symmetric Jensen-Shannon loss(weight 1.0)를 더한다. micro batch 16, accumulation 4는 기존 effective batch 64를 유지하기 위한 메모리 실행 설정이다.

Q3와 Q4 blind candidate를 모두 seal/hash하기 전에는 어느 fold metric도 계산하지 않는다. 최종 gate는 Q3 delta F1 `> 0`, Q4 delta F1 `> 0`, pooled delta F1 `>= +0.001`, anchor-positive removal `= 0`이다.

## 산출물

- tracked: config, runner, independent verifier, focused tests, 이 보고서, aggregate JSON
- ignored: one-shot lock, replay history, sealed prediction NPZ/receipts, terminal JSON
- tracked 경로에 원본 행, prediction CSV, checkpoint를 쓰지 않는다.

## 실행 결과

one-shot은 2026-08-29 KST에 약 55분 27초 실행됐다. Q2 width-512 e150 3-seed replay의 기본-view decoder probability, boundary probability, threshold-0.8 proposal은 기존 sealed Q2 cell과 모두 bitwise 동일했다. 따라서 재구성 identity는 통과했다.

| 항목 | 관측 | gate | 판정 |
|---|---:|---:|---|
| `q99(abs(p0-p256))` | `0.0037284570280462503` | `>= 0.05` | FAIL |
| proposal XOR | `29`행 | `>= 50`행 | FAIL |
| fixed-average Q2 anchor-union Delta F1 | `+0.07564194399984181` | `>= +0.001` | PASS |
| default replay identity | probability/boundary/proposal 모두 bitwise equal | 모두 equal | PASS |

평균-view Q2 candidate F1은 `0.8679175181125767`, frozen anchor F1은 `0.7922755741127349`였다. 그러나 이 값은 기존 e150 효과가 대부분이며, 핵심 가설인 window-phase disagreement의 크기와 proposal 변화량이 사전 gate에 크게 못 미쳤다. 결과적으로 center-weighted overlap-add가 이 nuisance를 이미 충분히 약화한다는 반증으로 해석한다.

계약대로 paired-view symmetric-JS warm-start는 시작하지 않았고 실행 횟수는 0이다. Q3/Q4 truth/metric은 열지 않았으며 checkpoint·prediction CSV·submission 생성과 upload도 0건이다.

## 독립 QA

별도 verifier가 sealed NPZ hash와 array inventory를 검증하고, historical Q2 truth/anchor에서 q99·XOR·Delta F1·gate를 독립 재계산했다. tracked `aggregate.json`과 append-only terminal JSON이 일치했고, run namespace의 CSV는 0개였다. QA 결과는 PASS다.

재현 명령:

```powershell
.\.venv-p1\Scripts\python.exe scripts\run_p1_window_phase_consistency_20260829_v1.py --check-only
.\.venv-p1\Scripts\python.exe scripts\run_p1_window_phase_consistency_20260829_v1.py --execute
.\.venv-p1\Scripts\python.exe scripts\verify_p1_window_phase_consistency_20260829_v1.py
```

두 번째 명령은 one-shot lock 때문에 이미 실행된 namespace에서는 의도적으로 거부된다. 동일 결과를 얻기 위한 자동 rerun은 이 계약에서 허용되지 않는다.
