# P1 official component claim-evidence ledger

| Claim | Evidence | Identification | Decision impact |
|---|---|---|---|
| 0.3점에는 약 `+0.01128748` F1, 즉 Public F1 `0.84483548`가 필요하다. | 챔피언 `0.833548/28.909341`, persisted empirical slope `26.5781209 points/F1` | Planning conversion; official formula claim 아님 | 작은 2–15행 변화로는 목표 크기를 뒷받침하지 못함 |
| Router는 B보다 `+0.024163 F1`, `+0.642207점`이다. | 2026-08-26 frozen three-cell receipt | Exact file-level contrast | 이미 챔피언 anchor이므로 재제출/재조합 금지 |
| E150 333 additions는 router보다 `+0.015375 F1`, `+0.408627점`이다. | 2026-08-27 Round F | Exact file-level contrast | 이미 챔피언에 포함 |
| I 80행의 공식 marginal은 GI2 존재 시 `+0.010753 F1`, `+0.285796점`이다. | champion vs v33a exact 80-row removal | Exact set-level contrast; row effects는 미식별 | I 제거 family 폐쇄; 이득은 이미 챔피언에 있음 |
| I sign은 GI2 부재에서도 거의 복제된다 (`+0.010760 F1`, `+0.285961점`). | raw E150 all vs GS-only | Exact set-level contrast | I 보존 신뢰 강화, 새 이득은 아님 |
| G 15행은 `+0.004519 F1`, `+0.120101점`이다. | champion vs remove-G; I-only vs remove-GS factorial replication | Exact set-level contrast at displayed resolution | G 제거 family 폐쇄; 이미 챔피언에 있음 |
| S 238행은 표시상 net-neutral이다. | champion vs remove-S; remove-G vs I-only | Aggregate set-level only, nearest-rounding 가정 시 marginal 약 `±0.000001 F1` | 유익/유해 row subset을 official aggregate로는 식별 불가 |
| S layer subset 제거도 현재 historical 근거에서 선택되지 않았다. | v33c nested causal ablation: selected layers 0, delta 0 | Historical bounded replay, official 0 | Public score를 row oracle로 쓰지 않는 한 materializer 금지 |
| sparse veto는 325/333 E150 additions를 제거해 `-0.351064점`이었다. | 2026-08-29 official receipt | Exact file-level contrast | “몇 개 고정밀 행만 남기기” 계열의 강한 recall 위험 |
| label-shift EM family는 내부 PASS가 Public `-0.923012점`으로 역전됐다. | v30 internal receipt + official receipt | Family transport failure; 모든 long-event rescue 일반화는 아님 | 같은 score/EM/calibration family 결과맞춤 재시도 금지 |
| HistGBDT 4-row add-only는 표시상 tie였다. | 2026-08-31 official receipt | Exact file-level, row truth 미식별 | exact diff 재사용 금지; 0.3점 근거 없음 |
| Orthogonal long-event rescue는 아직 닫히지 않았지만 현재 제출 후보는 아니다. | Historical FN의 98%+가 long-event interior; v27/v28/v30 transport evidence | Mechanism remains plausible; deployable rule unresolved | fresh frozen prefix validation을 통과하기 전 ABSTAIN |

## Identified vs unidentified

식별된 것은 **파일 단위 set marginal**뿐이다. I·G·router·E150 전체의 부호와 크기는 재현 가능한 공식 대비가 있고 모두 현재 챔피언에 포함된다. 미식별인 것은 S 238행 내부의 TP/FP 구성, 개별 행 유틸리티, private surface, 그리고 새로운 long-event rescue의 Public transport다. 표시 점수에서 0인 S marginal을 행별 감독신호로 쪼개는 것은 허용되지 않는다.

## Exact next decision

`ABSTAIN`. 두 남은 슬롯 중 lane B가 추천할 exact candidate는 없다. 챔피언 보존의 기대 점수는 `28.909341`; 새로운 gain interval은 데이터가 없어 수치화하지 않는다. “범위 없음”은 보수적 표현이 아니라, 공식 aggregate만으로 row subset 후보의 확률분포를 식별할 수 없다는 뜻이다.

재개 조건은 독립 train-only long-event-interior rescue가 사전봉인 Q3/Q4에서 pooled `ΔF1 > +0.01128748`, 각 fold nonnegative, dependent day-block CI90 lower `>0`, 새로운 diff/hash를 동시에 만족하는 경우뿐이다.
