# P1 수심 스트레스 중단 복구 — 새 ID, 0-fit

## 결론: unknown-year에서도 bracket 상대 개선 유지 — fresh QA 67/67 PASS

실측 진단은18확률을 그대로 재사용하고 H2_2025의9확률만 새로 계산하여 **251.719초/모델fit0**에 완료했다.
아래는 동일한 과거양측OOF의반사실 비교이며 배포성능 증명이 아니다. 기존frozeninner정책과700트리를 유지했다.

| H1_2025 primary,208,093행 | clean O/B | O + B(bracket) | bracket 상대ΔF1 | paired-day CI90 |
|---|---:|---:|---:|---|
| 표준 양측 | .737031619 | .757196860 | +.020165240 | [+.005014770,+.037606590] |
| forced unknown-year | .739425357 | .752901598 | +.013476241 | [+.002593516,+.025738337] |
| training-year median fallback | .736990155 | .757152826 | +.020162672 | [+.004999837,+.037564197] |

Unknown-year primary는181calendar-day/2,000회/seed20260906,bootstrap P(Δ>0)=.9795다.
이 수치는공식성공확률이아니며 .8hardgate로사용하지않는다. 세조건 모두같은208,093행(양성9,911행)을평가한다.
Unknown primary의TP6,704→6,714(+10),FP1,518→1,210(−308),FN3,207→3,197이다.
주평가의이득은주로FP감소로설명할수있다. 전체pooled는TP20,238→20,927과FP4,037→4,141이모두증가하므로
그설명을전체기간에그대로확대하지않는다. Root의별도TP/FP/FN→F1산술대조도일치했다.

| 전체3fold,668,593행 pooled | clean O/B | O + B(bracket) | 상대ΔF1 |
|---|---:|---:|---:|
| 표준 양측 | .795140828 | .809641411 | +.014500582 |
| forced unknown-year | .775536012 | .789936585 | +.014400573 |
| training-year median fallback | .797963526 | .810178898 | +.012215372 |

**상대 개선과 절대 변동은 다르다.** 표준→unknown-year에서primary clean기준은+.002393738,
bracket후보는−.004295262다. 전체pooled에서는각각−.019604816/−.019704826이다.
따라서“수심문제를해결했다”가아니라“이두열손실스트레스에서도같은상태의기준보다후보가높다”는결론이다.
Fallback이표준을무료로복원하지도않는다. Primary는clean−.000041465/bracket−.000044033,
전체는clean+.002822698/bracket+.000537488로변동한다. 이결과에서fallback을선택·혼합하지않았다.

위험은남는다. Unknown-year의최악fold H2_2024는 .721322314→.705743043(상대Δ−.015579271),
최악정점층 H2_2024 S-ORS L2는 .922702965→.825992150(Δ−.096710815)이다.
이셀의TP949→947,FP108→346이다. 불리한fold/셀을제외하거나사후정책으로교체하지않았다.

실제encoder점검에서표준두메타의nominalmissing/범주미지원은0/0이고, forcedunknown-year는
이실험의전체668,593행모두NaN/미지원범주였다. 이는**현재과거outer모델의실측**이지공식169,011행의점검결과가아니다.
Fallback은nominalmissing0이지만평균깊이로새범주가생겨H2_2024/H1_2025/H2_2025에서각각
26,099/9,860/23,111행의depth_regime이학습encoder에없었다. 다른78열(O/B) 또는105열(bracket)은exact다.

실행결과SHA `ea8a7d097fd41e67ddc87cc1d06c3093de5d3c0fe650ad6de50bb78049e4d731`.
실행PID38612,source표준양측freshQA63/63PASS다. 별도PID40628의67/67검사에서재사용18확률을포함한
전27확률이동일frozen모델의새예측과exact일치했다. 모델/source해시,두열외matrix불변,고정decoder,
OOF키·순서·label·달력,metric/CI/slice도대조했다. 중단전에SHA원장이없던재사용부분의불확실성은이freshreplay로해소했다.
QASHA `25ed3d2f2e7bc59ce64394ecd714d661d27ca322379de8951d65d57442e9f615`.
QA299.453초 + 이어가기251.719초 = **551.172초(9.19분)**으로900초추가계산상한내완료했다.
기존v1부분산출물·lock·source·config변경0,모델fit0,공식/hidden/CSV/upload/Git0이다.
완료검증은같은환경의별도프로세스재현이며cleanroom전체훈련/공식답안/공식점수검증이아니다.
`result.json`의`COMPLETE_FRESH_QA_PENDING`은불변으로남기고최종재현판정은`independent-qa.json`으로분리한다.

주요근거: `scripts/run_p1_tuning_depth_stress_20260906_v1.py`의`replace_metadata`/`matrix_guard`,
`scripts/run_p1_tuning_depth_stress_20260906_v2.py`의`preflight`/`execute`/`qa`,
새config,`partial-manifest.json`,`result.json`,`independent-qa.json`이다.
이ID의`--execute`는이미소비됐으므로재시작하지않는다. 수치확인에는완료receipt를우선하고,
별도재검증필요가있을때만같은script의`--qa`(fit0)를사용한다.

사용자가 중지했던 작업의 확인·진행을 승인했다. Root가 승인한 범위는 별도 ID의0-fit 기술적 이어가기이며,
새학습/공식입력/CSV/upload/HPO는 없다. 기존v1 runner/config/attempt lock/부분 artifact를 수정하지 않는다.

### 종료 원인 기록

03:37 KST조회에서 v1 PID22780이 없고 unified execution은exit1이다. 마지막 로그는
H1_2025/bracket/start,275초이며, 마지막 확률/rules파일은03:12:12(약311초)에 저장됐다.
`terminal_result.json`과 `finally`결과가 전혀 없다. trace에Python예외·Access denied·600초상한 예외는 없다.
**사용자 중지와 정합적인 비정상 중단**으로 분류한다. 정확한OS종료신호가 기록되지 않았으므로 원인을 단정하지 않는다.
실험성능 실패나 권한차단으로 바꾸어 쓰지 않는다.

### 재사용·검증 계약

- H2_2024/H1_2025의 각3모델×3조건=18확률배열과2rules배열을 보존한다.
- lock의runner/configSHA일치,모델/source핀,확률길이·finite·[0,1],원표준확률6개exact를 사전확인했다.
- 중단전 개별SHA원장이 저장되지 않았으므로 `partial-manifest.json`은 **복구시점 수용해시**다.
  과거에 봉인된 해시인 것처럼 주장하지 않는다. 완료후별도PID 모델→확률 replay로 재사용배열까지exact대조한다.
- 실행은 완료18확률을 그대로 읽고 **H2_2025의9확률만 신규 추론**한다. source training/hash/context는 그대로다.
  train/validation 키·label·달력·원시depth·모델·encoder·rules·threshold·policy·700iteration 모두 고정한다.
- 변경은두메타열뿐이다. unknown문자열을 실제encoder에통과시키며, **학습에없는범주일때만 −1**이다.
  강제−1코드,median/0numericimputation,결과기반fallback선택은 없다.
- CPU2/GPU0/모델fit0/공식·hidden·CSV·upload0. 이어가기와freshreplay 계산합계15분의단계경계상한이다.
  진행중 단일 추론/decoder를 강제kill하지는 않는다. 초과·오류는새terminal로남기며자동재시작하지않는다.
- 사전 합성/경계 pytest9 PASS, 기존feature/encoder differential14checks PASS, Ruff PASS.

표준양측·forcedunknown-year·train-stats medianfallback은 별도결과로 보고한다.
반사실스트레스를미래성능증명이나fallback재학습결과로확대하지않는다.
모델은원래year-key로학습되었으며실제rawdepth누락은여전히그대로다.
정상train min/max만으로공식FP0/FN0가보장되지는않는다.

## 후속범위

이번스트레스에서unknown-year의상대개선이유지되었으므로, fallback을섞지않은bracket-only의
아래좁은후속준비는검토할가치가있다. 다만스트레스는이미본과거OOF이며미래/공식확인과다르다.
8×HPO·공식materialize를자동시작하지않는다. Root가후속실행을분리할때까지다음내용은설계제안뿐이다.
다음좁은후보준비안의4fit은기준portable과같은고정finalinner
2025-07-12~09-10(train stop06-21,21일purge) O/B2fit와fullO/B2fit이다.
이번outer의H1승자나H2threshold를옮기지않고동일select_inner로B107을평가하는별도계약이필요하다.
그workflow두번독립재생성은8fit이며4fit과혼동하지않는다. 현재이후속fit은0이다.
