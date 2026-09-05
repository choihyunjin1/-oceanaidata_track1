# P1 bracket forward — 결과와 재현 QA

## 결론

**사전등록 primary 평균 개선으로 후보를 보존한다. 공식 점수 상승·자동 제출·최종 패키지 완성 판정은 아니다.**
기존 O/B 선택 알고리즘에 B의 bracket 특징만 추가한 완성 절차가 calendar H1_2025
전체 208,093행에서 F1 **0.6927465363 → 0.7000919963, Δ +0.0073454601**을 냈다.
107,125개 unseen station-layer 행을 제거하지 않았다. 세 forward fold 모두 평균 개선했지만,
primary paired-day bootstrap CI90 **[−0.0002816849, +0.0165006380]**은 0을 포함한다.
개선 resample 비율 0.937은 공식 개선 확률이나 0.8 hard gate가 아니다.

20/20fit이 986.235초(약16분26초)에 종료됐다. 별도 PID replay **60/60 PASS**,
Root 별도 pooled 산식·키 검증 PASS다. 추가 성능 학습·공식 입력·CSV·upload·Git 작업은 하지 않았다.

원장: [result.json](result.json), [independent-qa.json](independent-qa.json),
[root-arithmetic-qa.json](root-arithmetic-qa.json), [사전등록](preregistration.md), [지원 표본](support.json).
수치의 canonical 원장은 result이며 아래 표는 읽기 편의상 핵심만 표시한다.

## 사전 고정 비교

| 평가 표면 | 행 수 | Control F1 | Bracket 절차 F1 | ΔF1 |
|---|---:|---:|---:|---:|
| **H1_2025 primary 전체** | 208,093 | 0.6927465363 | 0.7000919963 | +0.0073454601 |
| H2_2024 forward | 172,638 | 0.4434079602 | 0.4790213325 | +0.0356133723 |
| H2_2025 forward | 287,862 | 0.8570215025 | 0.8883074239 | +0.0312859214 |
| 세 forward pooled 부표 | 668,593 | 0.7078250586 | 0.7317740625 | +0.0239490039 |

원래80열과 O/B 학습 recipe는 유지하고, B에만 6/24/72시간 bracket의27열을 추가했다.
각 창 외부 flank1시간, 새 bank의 최대 물리적 입력 반경37시간이며 gap을 넘어 행을 압축하지 않는다.
threshold/policy는 fold별 earlier-inner에서만 선택했다. H1에서는 control이 O,
후보가 O와 새 B union을 선택했으므로 **동일한 최종 decoder를 고정한 단일 특징 효과**라고 주장하지 않는다.
이는 미리 고정한 같은 선택 알고리즘 아래의 두 절차 비교다. 기존 depth repair나 T–S 변경은 결합하지 않았다.

P1 v5의 3forwardfold/21일purge/양성 run 시작 귀속을 유지했다. 12control O/B inner·outerfit을 모두 봉인한 뒤
6bracketfit을 실행했고 성능에 따른 추가 seed/window/threshold 변경은 없었다.
이미 연구에 노출된 배포 train 시기를 쓰므로 retrospective development 결과이지 fresh confirmation은 아니다.
과거 공식 clean control F1과 평가 시기·정책이 다르므로 이 내부 F1을 과거 공식 점수에 더하거나 옮기지 않는다.

## Known/unseen과 실패가 남은 곳

| H1_2025 진단 | 행 수 | Control F1 | Candidate F1 | ΔF1 |
|---|---:|---:|---:|---:|
| unseen station-layer | 107,125 (51.48%) | 0.6198709083 | 0.6328798186 | +0.0130089103 |
| known station-layer | 100,968 (48.52%) | 0.7671136203 | 0.7692487168 | +0.0021350964 |

primary 전체 TP는 5,950→6,088(+138), FP는1,317→1,393(+76), FN은3,961→3,823이다.
unseen에서도 이득은 있지만 FN2,779개가 남는다. known-only로 primary를 바꾸거나 unsupported 문제를 해결했다고 하지 않는다.

- H1 G-ORS L1은 F1 0.70936→0.59504: TP72는 같고 FP59→98로 악화했다. station-layer 중 최악 Δ다.
- H1 I-ORS L7은0.08924→0.16970으로 개선했지만 FN1,321개가 남는다. H1 S-ORS L8은0.52170 그대로다.
- 전체 forward 정상행 FP는1,585→1,744(+159). 양성 run 경계 TP218→235, 내부 TP15,942→16,879로 개선했지만
  내부 누락10,678개가 남아 경계 특징만으로 내부 이상을 다 해결하지 못했다.
- 유형 부표에서 drift TP3,850→4,281, offset2,553→2,816, noise4,713→4,997,
  flatline5,316→5,316이다. 복합유형 행은 여러 부표에 포함될 수 있으므로 유형 합계를 pooled 분모로 쓰지 않는다.

이 부표는 다음 가설의 근거이지 현재 결과를 보고 정점별 계수나 예외 threshold를 만든 근거가 아니다.

## 재현 및 경계 QA

- 별도 프로세스 두 번의 canonical all-train O 경로는 CPU2/동일 명령에서 model package,
  전체 encoded matrix, train keys 및 첫4,096개 in-sample train-feature probe 예측 hash가 모두 일치했다.
  PID33936/5768, 새 O package SHA `58430c8327b674e40c16a09e9bd19fd77cbef691bef6f3282c9a8140bb2aa696`.
  probe의 품질/F1은 계산하지 않았다. 두 full O는 OOF에 사용하지 않았다.
- 이는 **새 명령의 full O 결정론**이다. 과거 CPU4 full O 차이의 원인을 확정하거나,
  전체 O/B 최종모델 두 번 재학습·공식 답안 exact 복원·clean-machine package 검증을 대신하지 않는다.
- 학습 PID34820과 다른 PID21444에서18개 inner/outer 모델의 확률을 전부 재생성했다.
  model/probability hash 및 확률 exact, 모든 OOF key/bit, inner 재선택과 aggregate/CI/slice 재계산이 **60/60 PASS**.
- focused synthetic pytest12 PASS, Ruff PASS. partition 선택을 stats/features/encoder/rules/decoder보다 먼저 적용한다.
  배제 partition 변형 및 outer 관측/label 변형이 train 입력에 들어가지 않는 sentinel을 확인했다.
- hysteresis는 허용 partition 전체 low-probability run에 의존한다. 무한 run context를21일purge만으로 안전하다고
  선언하지 않았으며, 승인된 partition 격리 계약에서 검사했다. 양측 훈련으로 바꾸려면 별도 계약·sentinel이 필요하다.
- train input 하나의 SHA `20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2`;
  과거 모델/답안 읽기0, official0, hidden0, CSV0, upload0, score inversion false.

| 산출물 | SHA-256 |
|---|---|
| frozen runner | `dc111189f3f58a8f01ec59634ee6bb51ed63928399efe592fc032d5317828e93` |
| frozen config | `cfdb92b798a9ec36095b62bb1ddb58fb3c293ec0ab02c53fbea7d4579f797c04` |
| terminal result | `d463f9181a9050f2cc550a624923bcecd70c88303a060a3d6f41af95e79daa4b` |
| local OOF parquet | `896049e265f818c57449191d55a2140ead8507295c2c6c3f6643e5d76924c247` |

## 다음 판단 — 자동 실행 없음

후보의 평균 개선을 보존하되 이 실험의 primary/분모/threshold/모델은 그대로 둔다.
별도 [양측 split 지원 점검](../p1_twosided_support_20260906_v1/preregistration.md)은
같은 검증행에서 다른 배포 train 시기의 station-layer 지원이 생기는지를 모델 없이 조사한다.
미래 시점 train label을 허용하는 보간형 질문이므로 chronological 성능 검증과 동의어가 아니다.
새 prospective 계약과 inner 선택·context 격리 검증을 먼저 확정하며, 다음 학습/튜닝/제출은 자동으로 진행하지 않는다.
