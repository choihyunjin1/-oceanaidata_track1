# 결론: 사전등록 디코더 전략은 NO_GO, clean O/B control 보존

동일 421,032행에서 control F1 0.8511742399 대비 inner on/off 선택 전략은 0.8502914238(Δ -0.0008828161)이었다. 따라서 이 디코더는 최종 후보에 넣지 않는다. 최고점 개선을 입증한 결과가 아니며, 모델 학습 없이 기존 O/B 확률의 후처리 가능성을 한 번 확인한 결과다.

| 내부 외부 구간 | inner 선택 | control F1 | 선택 전략 F1 | always-on 진단 F1 |
|---|---|---:|---:|---:|
| 2025 Q2 | OFF | 0.7721567136 | 0.7721567136 | 0.7756441185 |
| 2025 Q3 | OFF | 0.8782268115 | 0.8782268115 | 0.8805837338 |
| 2025 Q4 | ON | 0.9145813282 | 0.9116616314 | 0.9116616314 |
| pooled | 구간별 고정 선택 | 0.8511742399 | 0.8502914238 | 0.8527915384 |

선택 전략은 TP 21행을 추가했지만 50행을 제거했다. FP는 8행 제거, 추가 0이었다. 보존 대상으로 정의한 plateau/confirmed spike는 제거 0이다. raw always-on pooled 값은 +0.0016172985로 보이지만, 이것을 보고 inner 선택을 폐기하면 사후 정책 변경이므로 사용하지 않는다.

Backbone 재학습 0, train-only 전이 추정 6, inner ON/OFF 선택 3, runtime 175.688초. lambda=1/Laplace=1, 실제 10분 인접이며 정점·층·gap·학습 경계를 넘지 않는 전이만 집계했다. high-threshold를 뺀 logit unary는 calibrated likelihood가 아닌 scoring heuristic이다. 공식 input·sample·hidden 접근, CSV·upload 0.

첫 launcher는 run_ 접두사 누락 때문에 lock 전 종료했고 평가 0이었다. preflight-repair.md에 기록한 root 승인 경로 수리 후 첫 실제 평가를 진행했으며 원 실패 로그는 보존했다. 사후 재튜닝·모델 재학습은 없다.

Synthetic pytest 10 PASS/Ruff PASS. Root 독립 QA도 key unique/finite/행수·TP·FP·FN·F1을 확인했다. 결과 SHA d6650e05026fb9bf4db264dbee0f3ad2b1aab636d27fecdd2f8f27cfeff2ce12, QA NPZ SHA f2c956411661982b57a4de03cd59394ece49755fde651ab0b71a838bfa2b32b5.

다음 단계는 이미 승인된 clean control만 fulltrain 2-fit로 보존하는 것이다. 2025 재사용 평가면이며 새로운 독립 검증이나 공식 점수 개선을 주장하지 않는다. 점수 환산 -0.023482908점은 ΔF1×26.6이 그대로 전이된다는 조건부 산술표시일 뿐 예측 공식 점수가 아니다.
