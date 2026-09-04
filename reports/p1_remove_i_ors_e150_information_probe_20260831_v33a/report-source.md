# P1 v33a 결론

`P1_REMOVE_I_ORS_E150_INFORMATION_PROBE`는 **성능 승격 후보가 아니라, 오늘 마지막 슬롯이 만료될 때만 고려할 수 있는 조건부 정보 probe**다. Historical raw E150 대비 I-ORS의 `incumbent=0, raw_E150=1` addition만 제거했을 때 Q3/Q4 pooled `ΔF1 +0.003956398`, 선형 예상 `+0.105153636점`이었다. 170개 제거 중 false positive 133개와 true positive 37개가 포함됐고, day-block CI90은 `[-0.001571471,+0.010770464]`, `P(improve)=0.856`으로 0을 가로지른다.

Q2는 `ΔF1 -0.002176446`(204개 제거, TP 100/FP 104), Q3는 `+0.006613338`(168개, TP 37/FP 131), Q4는 `+0.000217205`(2개, TP 0/FP 2)다. Q2/Q3/Q4 전체 pooled는 `+0.001949502`다. 따라서 세 fold 모두 개선이라는 성능 gate는 실패하지만, Q3/Q4 중심값은 양수이고 2026-08-30 공식 factorial에서 G-ORS removal은 유해, S-ORS removal은 표시 해상도상 중립, I-ORS removal만 미측정이어서 질문 자체의 정보가치는 남아 있다.

Action은 정답 접근 전에 봉인됐다. raw E150 SHA는 `f3da6a56...e96ebf`, candidate SHA는 `68f782c0...799a73`, removal SHA는 `b284b270...6bc093`, sealed NPZ SHA는 `e6a20bae...9fd0a0`이다. Fit 0, retry 0, action seal 전 truth read 0, official/hidden/test/sample/CSV/upload 모두 0이다.

공식 materializer는 준비만 됐고 실행하지 않았다. 실행될 경우 canonical champion+GI2와 anchor positive를 그대로 보존하며, frozen official E150 addition 333개 중 I-ORS 80개만 제거해 169,011행·positive 6,316행 후보를 만든다. Materializer preflight는 official read 0으로 `READY_NO_OFFICIAL_READS`를 통과했다. 정확한 실행 명령은 다음과 같지만 별도 선택 전 실행하면 안 된다.

```powershell
.\.venv-p1\Scripts\python.exe scripts\materialize_p1_remove_i_ors_e150_information_probe_20260831_v33a.py --execute
.\.venv-p1\Scripts\python.exe scripts\validate_submission.py "C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P1_REMOVE_I_ORS_E150_INFO_PROBE_READY_V33A\P1_1_REMOVE_I_ORS_E150\P1_submission.csv"
```

현재 판단은 `CONDITIONAL_GO_INFORMATION_PROBE_ONLY_IF_LAST_SLOT_WOULD_EXPIRE_AND_REGRESSION_RISK_IS_ACCEPTED`다. 개선을 주장할 근거는 아니며, 공식 결과가 동점·회귀·무효라면 Public F1 `0.833548`, points `28.909341`, SHA `57844ef2...53687`인 기존 champion을 유지한다.
