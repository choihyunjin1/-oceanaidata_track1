# 2026-08-27 구조적 challenger 실험 — canonical report source

상태: **실험·독립 QA 완료 / 제출 승격 없음 / 공식 업로드 없음**

## 결론

새 구조를 미리 탐색하고 실제로 검증할 수 있으며, 이번 사이클에서 P1·P2·P3 각각 한 가지 비중복 구조를 사전등록해 실험했다. 결론은 세 후보 모두 현재 제출 모델로 승격하지 않는 것이다.

- **P1:** 장기 이상 내부의 7~48행 구멍을 잇는 topology bridge는 전체 micro-F1을 `0.001454` 낮췄다. 이 정확한 bridge 계열은 폐기한다.
- **P2:** joint T/S continuous-depth 모델에서 density-gradient soft loss는 첫 3개 블록에서 `-0.002812 RMSE` 신호가 있었지만, 새 2개 확인 블록에서는 `+0.000678 RMSE`로 재현되지 않았다. 물리 메커니즘도 확인되지 않아 승격하지 않는다.
- **P3:** lead-continuous residual surface는 전체 `-0.003141 m`, 실제 학습이 적용된 두 fold에서 `-0.004188 m` 개선했으나 사전등록된 최소 효과와 신뢰구간 gate를 통과하지 못했다. 방향성은 유망하지만 현재는 `INCONCLUSIVE`다.

따라서 이번 연구는 “아무것도 시도하지 않은 보류”가 아니다. 세 구조를 실제로 소거·선별했고, 다음 계산은 P1 bridge와 P2 density-loss의 미세 조정이 아니라 **P3의 완전히 분리된 신규 holdout 구축**에 우선 배분하는 것이 합리적이다.

## 범위와 불변 경계

- 동결 Round E 세트의 시작 및 종료 `SET_MANIFEST.json` SHA-256은 모두 `7dd80d6288cd957192055916627b6bd31778565defb60f56c9baf078c8d487bc`다.
- 동결 상태는 `FROZEN_READY_NOT_UPLOADED`, 이번 challenger의 공식 업로드 수는 `0`이다.
- 새 파일은 `artifacts/structural_challenger_20260827_v1`와 `reports/structural_challenger_20260827_v1`에 격리했다.
- 공식 test, sample submission, submission candidate, hidden target, P3 ERA5 고정 실험은 읽거나 변경하지 않았다.
- 기존 dirty worktree와 동결 제출 파일은 수정하지 않았다.
- 결과를 본 뒤 threshold, feature, split, 모델 크기, epoch, seed 또는 gate를 바꾸는 재실행은 하지 않았다.

## 연구 근거와 실험 설계

| 문제 | 구조 가설 | 한 번의 사전등록 검증 | 결정 기준 |
|---|---|---|---|
| P1 | 점별 분류기가 놓친 장기 offset/drift 내부 구멍은 segment topology로 복구할 수 있다 | 동결 OOF의 7~48행 내부 구멍만 비파괴적으로 연결 | pooled F1 `+0.0015` 이상, 3 fold 중 2개 이상 개선, slice/integrity guard 통과 |
| P2 | T와 S를 연속수심 좌표에서 함께 복원하고 density-gradient soft loss를 주면 단순 joint decoder보다 일반화가 좋아진다 | 동일 초기값·순서·예산의 control 대 physics arm | pooled ΔRMSE, paired-day CI, block/layer 일관성, 별도 N² 메커니즘 gate |
| P3 | lead별 독립 보정보다 lead 좌표와 causal regime을 공유하는 저용량 surface가 sparse sample을 효율적으로 쓴다 | 첫 fold exact no-op, 이후 fold는 과거 fold만으로 ridge fit | 전체 및 active ΔRMSE, day bootstrap CI, 두 active fold와 slice guard |

이 선택은 기존 저장소에서 반복된 모델을 다시 이름만 바꾼 것이 아니다. P1은 구간 proposal 관점, P2는 온도·염분 공동 구조와 밀도 안정성, P3는 lead-time-continuous 계수 공유를 각각 최소 구현으로 분리했다. 구간 단위 모델링은 temporal action localization의 boundary/proposal 관점과 맞닿아 있고([ActionFormer, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf)), T/S 공동 모드는 단일 수온 단조화보다 thermohaline 구조를 직접 표현한다([Joint T/S FPCA, JPO 2019](https://journals.ametsoc.org/view/journals/phoc/49/10/jpo-d-19-0120.1.xml)). 해수 안정성은 온도만이 아니라 T/S 결합으로 결정되므로 N²를 secondary mechanism으로 사용했다([TEOS-10 N²](https://www.teos-10.org/pubs/gsw/v3_04/html/gsw_Nsquared.html)). P3의 horizon 계수 공유는 lead-time-continuous postprocessing 연구와 같은 통계적 동기를 갖는다([Wessel et al., QJRMS 2024](https://doi.org/10.1002/qj.4701)).

## P1 — long-event topology bridge

### 고정 규칙

- 동결 Round-B 3-seed binary endpoint가 모두 양성인 parent segment만 사용했다.
- parent duration은 48~519행, 내부 hole은 7~48행으로 제한했다.
- incumbent의 양성은 하나도 제거하지 않았고, 기존 close-gap `<=6` 표면과 겹치지 않게 했다.
- truth/anomaly type은 proposal 생성에 사용하지 않았다.

### 결과

| 지표 | incumbent | challenger | 변화 |
|---|---:|---:|---:|
| pooled micro-F1, 421,032행 | 0.864670 | 0.863216 | **-0.001454** |
| Q2 fold | 0.791862 | 0.787998 | -0.003864 |
| Q3 fold | 0.889630 | 0.890703 | +0.001074 |
| Q4 fold | 0.914567 | 0.912436 | -0.002131 |

4개 proposal이 126행을 추가했지만 추가분은 TP 33, FP 93으로 precision이 `0.261905`에 불과했다. 개선 fold는 1/3뿐이며 offset/drift recall 증가는 없었다. anchor 1→0 변화와 기존 gap surface 중복은 모두 0이었다. 판정은 `REJECT_FAMILY`다.

해석 범위는 정확히 이 endpoint-unanimity topology bridge에 한정한다. 모든 segment model이 무효라는 뜻은 아니지만, 같은 bridge의 gap·duration threshold를 결과에 맞춰 재조정하는 것은 중단한다.

## P2 — joint T/S continuous depth + soft density loss

### 첫 화면

control과 physics arm은 parameter `317,186`, seed `20260827`, 18 epoch, fold별 동일 초기 state와 minibatch 순서, AdamW `lr=3e-4`, weight decay `1e-3`, vertical-difference weight `0.25`를 공유했다. 유일한 차이는 physics arm의 선형화 density N²-gradient truth-matching weight `0.10`이다. validation T/S는 입력 panel에서 함께 가렸고, 양 arm의 공개 입력은 byte-semantic identity를 통과했다.

| 첫 3개 블록, 69,850행 | control | physics | 변화 |
|---|---:|---:|---:|
| pooled T RMSE | 1.018070 | 1.015259 | **-0.002812** |
| paired KST-day CI90 |  |  | `[-0.004483, -0.001213]` |

블록 변화는 `-0.014598`, `+0.004512`, `-0.005401`; layer 변화는 L2 `-0.013925`, L3 `-0.003239`, L4 `+0.000710`이었다. N² secondary는 두 adjacent pair 모두 악화했다.

첫 화면에는 외부 문제별 prereg gate와 runner 내 block veto가 서로 다른 `PROMOTE_TO_CONFIRMATION`/`INCONCLUSIVE`를 내는 P0 decision-contract 충돌이 있었다. 어느 규칙이 유리한지를 결과 후 선택하지 않았다. 대신 더 엄격한 통합 confirmation gate를 새로 고정하고 별도 기간에서 한 번 확인했다.

### 독립 기간 확인

첫 confirmation v1의 두 기간은 timestamp는 완전했지만 finite target T/S가 0이어서 **과학 지표 계산 전** 종료했다. 이 실패는 tombstone으로 보존했다. 이후 metric/prediction 값을 보지 않고 “완전한 2개월 KST coverage + L2/L3/L4 각각 joint finite T/S 6,000개 이상 + 첫 화면과 비중복 + 서로 다른 season bin”을 feasibility rule로 고정했다. 그 규칙의 시간상 첫 두 블록인 2024-05~06, 2024-07~08을 v2로 정확히 한 번 실행했다. 두 블록은 과거 내부 산출물에 등장한 적이 있어 pristine holdout은 아니며, 이 한계도 사전등록에 명시했다.

| 확인 2개 블록, 48,958행 | control | physics | 변화 |
|---|---:|---:|---:|
| pooled T RMSE | 1.662523 | 1.663201 | **+0.000678** |
| paired KST-day CI90 |  |  | `[-0.000747, +0.002019]` |

- block: May–Jun `-0.001540`, Jul–Aug `+0.003374`로 1/2 개선.
- layer: L2 `+0.001327`, L3 `+0.000665`, L4 `+0.000427`로 0/3 개선.
- N²: 2–3 `+2.7126e-05`, 3–4 `-1.7931e-08`로 두 pair 개선 조건 실패.
- 자동 판정과 종합 판정은 모두 `INCONCLUSIVE`; 실무 결정은 `NO_PROMOTE`다.

첫 화면과 확인 화면 118,808행을 사후 aggregate-only로 합치면 control `1.322247`, physics `1.321326`, Δ `-0.000920`이다. 이는 CI가 없는 진단치이며 사전등록 최소 개선 `-0.001`에도 못 미친다. 따라서 첫 화면의 유리한 부호는 새 계절에서 재현되지 않았고, decision-contract 충돌과 무관하게 density-loss component는 승격 근거가 없다. 이 결론은 joint continuous-depth decoder 전체를 폐기하는 것이 아니라 **density-gradient weight 0.10의 부가 가치가 확인되지 않았다**는 뜻이다.

## P3 — lead-continuous causal residual surface

### 고정 규칙

- 181 cases/1,086 rows의 corrected-v2 OOF와 6개 lead만 사용했다.
- basis는 lead linear/quadratic와 `lead ×` 4개 causal regime feature, ridge alpha `16`, correction clip `±0.15 m`, weight bound `[-0.25, 0.5]`로 한 번 고정했다.
- 첫 fold는 exact byte-identical no-op이고, 뒤 두 fold는 현재 fold target을 전혀 보지 않고 이전 fold만 fit했다.

### 결과

| 평가면 | incumbent | challenger | 변화 |
|---|---:|---:|---:|
| 전체 181 cases | 0.779105 m | 0.775964 m | **-0.003141 m** |
| active 132 cases | 0.801702 m | 0.797514 m | **-0.004188 m** |

두 active fold와 3/6/9/12/18/24h 여섯 lead는 모두 개선했다. 그러나 active day-bootstrap CI90은 `[-0.010129, +0.001585]`로 0을 포함했고, 사전등록한 active 최소효과 `-0.005 m`에도 못 미쳤다. I-ORS는 `+0.002158 m` 악화했지만 slice guard 범위 안이었다. 판정은 `INCONCLUSIVE`다.

확인용으로 재사용 가능한 기존 artifact surface를 result-blind audit했지만 fresh surface는 없었다. current exact는 181/181이 겹치고, 182/event188/ReVIN 계열도 179~180 cases가 직접 겹쳤다. ID상 분리된 legacy 177 cases도 같은 station 기준 전부 72시간 안에 있어 48h context + 24h target 관점에서는 실질적으로 100% 중복이다. 같은 표면을 이름만 바꿔 재검증하면 독립 증거가 아니므로 두 번째 실행을 하지 않았다.

## 교차 문제 판정

| 문제 | 과학 판정 | 제출 승격 | 다음 행동 |
|---|---|---|---|
| P1 | `REJECT_FAMILY` | NO | exact bridge family 종료; 다음에는 별도 boundary proposal scorer가 있는 비파괴 segment ranker만 검토 |
| P2 | screen signal, confirmation `INCONCLUSIVE` | NO | density loss 미세튜닝 중단; joint decoder 자체의 incumbent 대비 효과를 별도 matched test로 분리할 때만 재개 |
| P3 | 일관된 개선 방향, CI/최소효과 미달 | NO | corrected-v2를 재생성할 수 있는 신규 78h-separated holdout을 먼저 구축 |

이번 사이클의 우선순위는 P3 > P2 decoder-only decomposition > 새로운 P1 segment ranker다. 다만 P3도 신규 holdout이 없으면 추가 계산의 정보가치가 거의 0이므로, 기존 181/182/188 표면의 반복 학습은 금지한다.

## QA와 재현성

- P1 결과 SHA-256: `5d34b4e0006fd215ef03684e32265f9bf109bd23cde00d4ed9ba29add2521ec4`
- P2 첫 화면 결과 SHA-256: `aca355e9cc5ab6df5d9e04d0a7aeb61ecb785f0adcc21534a5d892064217954e`
- P2 confirmation v2 prereg SHA-256: `67c96273d55d86b88a31760f82f3064003fcc4150793a2b01889b9fb58437adf`
- P2 confirmation v2 결과 SHA-256: `28f7d417277e8a91627e6cd589c6da4350de979093c938d09385834a35887fe2`
- P2 confirmation v2 manifest SHA-256: `68a3ada851ac2cc19595c44c7da0d0b6e3386b09a75595398521ff19dea0a823`
- P3 결과 SHA-256: `5343dfd5846f2ed21bfe1b1e033e9acebf03385e98d79bf9693486cb3ba5ee8c`
- P3 confirmation-surface audit SHA-256: `ee4b55a2c437ab76a257149833272ecef5c4a33da4dcb2d032d4fa88815e651d`
- 기존 독립 교차 QA SHA-256: `1fd22e71adbf8987d0838fd1705cb78485725bde921cfd88805ee89ab171da73`
- P2 confirmation 독립 QA SHA-256: `169980f05c8db825dee5cfd4e1d053ebdef10383d69631db71e3c98e1737ec3d`

P1 저장 confusion matrix의 F1, P2 저장 SSE/RMSE, P3 저장 계수 기반 44개 scalar/metric은 독립 재계산과 일치했다. P2 confirmation은 RMSE/SSE 최대 오차 `2.22e-16`, availability exact recount, verdict 재계산을 통과했다. 독립 QA가 확인한 재현성 caveat는 다음과 같다.

- confirmation manifest의 `sanitized_command`가 wrapper가 아니라 상속 engine을 가리켜 그대로 복사하면 v2가 재현되지 않는다.
- confirmation manifest에 v2 prereg hash가 직접 기록되지 않았다.
- result limitation에 “three blocks”라는 상속 문구가 남았지만 실제 confirmation은 2 blocks다.
- aggregate-only 정책 때문에 paired-day CI는 row-level OOF 없이 독립 재생성할 수 없다.

이들은 결과 수치나 `INCONCLUSIVE` 판정을 뒤집지는 않지만, 다음 runner에서는 실행 전 manifest schema 검사를 gate로 넣어야 한다.

## 제한

1. 모두 train-derived local evidence이며 공식 hidden score의 개선을 보장하지 않는다.
2. 이번 검증은 전체 모델 공간의 완전탐색이 아니라, 중복을 제거한 세 구조 가설의 한 번씩의 소거 실험이다.
3. P2 confirmation 기간은 prior team exposure가 있어 완전히 깨끗한 holdout이 아니다.
4. P3는 fresh confirmation surface가 없어 방향성 이상의 결론을 내릴 수 없다.
5. 동결 Round E 세트의 제출 여부와 순서는 이 연구와 별도이며, 정확한 CSV를 공식 제출하려면 사용자의 새 승인이 필요하다.

## 주요 문헌

- Zhang et al. (2022), *ActionFormer*: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf
- Joint temperature–salinity FPCA (2019): https://journals.ametsoc.org/view/journals/phoc/49/10/jpo-d-19-0120.1.xml
- TEOS-10 `gsw_Nsquared`: https://www.teos-10.org/pubs/gsw/v3_04/html/gsw_Nsquared.html
- Wessel, Ferro & Kwasniok (2024), lead-time-continuous postprocessing: https://doi.org/10.1002/qj.4701
