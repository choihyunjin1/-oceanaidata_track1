# P1 E150 + Ordered-CatBoost precision union v32g

## 결론

`NO_GO_INTERNAL_GATE`다. Frozen E150을 보존하고 Ordered-CatBoost `p>=0.8` 양성만 더한 고정 add-only 후보는 15행을 추가했지만 TP 4, FP 11로 marginal precision이 `0.266667`에 그쳤다. Pooled ΔF1은 `-0.000172982`, Q3+Q4 ΔF1은 `-0.000238805`였으므로 공식 CSV를 만들거나 업로드하지 않았다.

## 정확히 한 번 실행 결과

- fit/search/retry: `0 / 0 / 0`
- runtime: `9.634212s`
- 변경률: `15 / 421,032 = 0.003563%`
- addition precision / Wilson LCB90: `0.266667 / 0.125818`
- reference F1/2: `0.446649`
- Q2/Q3/Q4 ΔF1: `-0.000055981 / -0.000406664 / 0`
- pooled bootstrap CI90: `[-0.000455776, +0.000072676]`, P(improve)=`0.1256`
- Q3+Q4 bootstrap CI90: `[-0.000660190, +0.000106808]`, P(improve)=`0.1372`
- 단순 점수 환산 중심: `28.902994`점 (현재 `28.909341`점보다 낮음)

독립 모델 전체 precision이 높더라도 incumbent가 이미 음성으로 둔 차집합의 조건부 precision은 낮았다. 따라서 standalone precision을 add-only 효익으로 곧바로 해석할 수 없다는 반증이다. 결과를 본 뒤 threshold를 바꾸지 않았다.

## 봉인 및 접근 원장

Label column을 읽기 전에 E150 OR CatBoost action mask와 NPZ를 저장했다. proposal은 target-before-seal `0`, official read `0`을 기록한다.

- result SHA-256: `d6bcc13d0995637a91cecaa9092922485b1d4e15ac0b814255ae127a515afc78`
- proposal SHA-256: `98b2b04cdd10039b164a5cf0dea1dcd9b785359b58e5cbca1da94042b22a608b`
- sealed NPZ SHA-256: `5e6d2cf920a5c9006c09c8e71490ff74024f697529d72a43154c3e0e3172480d`
- config SHA-256: `bed5499c56daa68f280bc1f1c3a2700a6771626c24cd2dec4b16e8f6c017504c`
- runner SHA-256: `6399838d953cb2dd0fcd499c11ccd0e9b9247ec962d98ed4487ff8ab324d78de`
- official/test/sample/submission/hidden/upload: 모두 `0`
- focused pytest: `3 PASS`; Ruff 및 py_compile: PASS

Historical Q2/Q3/Q4는 반복 연구에 노출된 개발면이므로 bootstrap은 안정성 진단일 뿐 fresh-holdout 추론구간이 아니다. 이 후보는 그 제한을 감안해도 중심값과 precision gate가 모두 불리하다.
