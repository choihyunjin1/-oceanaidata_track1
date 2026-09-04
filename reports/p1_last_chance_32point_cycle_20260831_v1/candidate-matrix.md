# P1 last-chance decision matrix

## 결론

성능 승격 기준을 통과한 후보는 없다. 현 챔피언 `P1_1_E150_PLUS_GI_SPIKE2`(Public F1 0.833548, 28.909341점)를 보존한다. v33a REMOVE_I 정보 획득 probe는 2026-09-01 공식 채점에서 Public F1 0.822795, 28.623545점으로 회귀했으므로 I-ORS 제거 방향을 폐쇄했다. 남은 제출권은 이 factorial 계열의 반복에 쓰지 않는다.

| 후보 | 상태 | 핵심 내부 결과 | CI90 / 예상 점수 | 위험·중복 판정 | 공식 접근 |
|---|---|---|---|---|---:|
| v32a Ordered CatBoost event/day | NO_GO | pooled ΔF1 -0.123955; Q3+Q4 vs E150 -0.175536 | q34 CI [-0.245140,-0.110258]; 약 24.2439점 | 독립 tree이나 recall 붕괴 | 0 |
| v32b CAUSAL_CIF_LITE32 | NO_GO | add-only 변경0; standalone pooled ΔF1 -0.799554 | standalone CI [-0.836058,-0.763922] | selector abstain, standalone 오보정 | 0 |
| v32c causal endpoint-peer inpaint | NO_GO | pooled ΔF1 -0.000285; Q3 +0.000372, Q4 -0.001234 | CI [-0.000752,+0.000194]; calibrated -0.012967점 | morphology/peer 계열 중복 위험, precision 부족 | 0 |
| v32d CIF selective bidirectional | NO_GO | pooled ΔF1 -0.068800 | CI [-0.094109,-0.047688]; calibrated -1.833968점 | 901 FP addition, 531 TP removal | 0 |
| v32e CatBoost default 0.5 | NO_GO | pooled vs tabular -0.048750; q34 vs E150 -0.090549 | pooled CI [-0.075736,-0.023130]; 약 27.613650점 | threshold 계열 폐쇄, anchor removal | 0 |
| v32f prefix-calibrated CatBoost | NO_GO | pooled vs tabular -0.115136; q34 vs E150 -0.117402 | pooled CI [-0.165454,-0.072504]; 약 25.7890점 | fold threshold 0.06/0.68/0.40로 transport 불안정 | 0 |
| v32g E150 + CatBoost p>=0.8 | NO_GO | pooled ΔF1 -0.000173; q34 -0.000239 | pooled CI [-0.000456,+0.000073]; 약 28.902994점 | 4TP/11FP, marginal precision gate 실패 | 0 |
| v32h E150 OR CatBoost 0.5 | NO_GO | pooled ΔF1 -0.008656; q34 -0.010059 | pooled CI [-0.012880,-0.005387]; 약 28.641997점 | 22TP/333FP, union 계열 폐쇄 | 0 |
| v33a REMOVE_I all layers | OFFICIAL REGRESSION / REJECT | pooled +0.001950; q34 +0.003956; Q2 -0.002176 | 내부 q34 +0.105154점 예상과 달리 공식 F1 0.822795, 28.623545점; 챔피언 대비 -0.285796점 | I-ORS E150 80행 제거는 공식적으로 유해; 기존 챔피언 유지 | 1 submission / hidden truth 0 |
| v33b nested I-layer ablation | NO_GO | pooled -0.000459; q34 -0.000681; Q3 -0.001283 | pooled CI [-0.001944,+0.000515]; -0.018096점 | prefix-selected layer2가 Q3에서 역전; 집중도 93.55% | 0 |
| v33c nested S-layer ablation | NO_GO / ABSTAIN | 선택 layer 없음; ΔF1 0 | CI [0,0], 예상 0점 | 모든 S layer marginal precision이 cutoff 상회 | 0 |

v32i bounded synthesis는 실행 가능한 비중복 신규축이 없다고 판정했으며 후보·lock·CSV를 만들지 않았다. v32a–h의 CatBoost/CIF/morphology/union 변형을 더 반복하는 것은 같은 노출된 Q2/Q3/Q4에서의 결과 맞춤 위험이 크다.

## 제출 선택

1. 기본: 챔피언 유지. 이미 검증된 Public F1과 파일 SHA가 있는 유일한 방어적 선택이다.
2. v33a 공식 결과: 2026-09-01 Public F1 `0.822795`, `28.623545점`으로 챔피언보다 `0.285796점` 낮았다. I-ORS E150 제거 방향은 폐쇄한다.
3. 금지: v32a–h, v33a 재시도, v33b/v33c 제출 및 결과 기반 threshold/layer 재시도.

이 문서 작성 중 후보 재실행, 공식 row/value 읽기, CSV 생성, 업로드, finalization은 모두 0이다.
