# 다음 평가 계약 준비도 — 핵심 선택 후 코드 봉인 필요

결론: **현재 사양을 그대로 코드로 옮길 준비는 아직 안 됐다.** 세 문제의 표면·분모·분할 의미를 먼저 선택해야 한다. 아래는 실행 계약 제안이며 새 실험을 실행하거나 기존 v4 결과를 새 기준으로 재채점한 문서가 아니다. 문서와 기존 집계 metadata만 읽었다. 새 원본 관측/공식 입력/hidden/모델/답안 값 접근 0, 학습 0, 기존 코드 수정 0, Git 변경 0.

## 공통으로 먼저 확정할 것

- [root 실행 해석](roadmap-review.md) 8–19행을 우선한다. 로드맵 §2의 `P(개선)≥0.8`, P1_SPEC의 다중 하방 gate, P2_SPEC의 블록 악화 제한, P3_SPEC의 `≥0.9/4개 블록 개선`은 서로 다르다. **기존 결과에 어느 것도 소급 적용하지 않는다.** 다음 계약에서도 평균 변화·CI·worst/slice를 분리하고, 새 자동 승격 수치를 쓸지는 결과 확인 전에 한 번 선택한다. 추천은 평균 개선을 후보 보존 근거로 삼고 bootstrap 비율은 불확실성 표로 제공하는 것이다. 별도 0.8 hard gate를 암묵적으로 추가하지 않는다.
- `Δ`는 후보−기준으로 통일한다(F1 양수/RMSE 음수가 개선). “평균 Δ”는 fold 지표의 산술평균이 아니라 **같은 scoring keys에서 재계산한 pooled 지표 차이**를 뜻하도록 명시한다. 모델별 결측/실패 행을 조용히 제외하지 않는다. 공식 지표와 새 stress/composite 지표는 이름을 분리한다.
- [COMMON_SPEC](../../docs/ocean_v2_codex/COMMON_SPEC.md) 35–37행의 bootstrap은 paired cluster 재표집, 동일 복제 가중, 고정 seed·반복 수·동률 처리·빈 cluster 처리까지 고정해야 한다. `p_improve`는 재표집 개선 비율이지 공식 점수 개선 확률이 아니다. 평균 위험·선택 편향·지원되지 않은 계절을 없애지 못한다.
- COMMON_SPEC 15–19행의 seed fan-out·정렬·threads를 기존 모델에 적용하면 단순 평가 리팩터링이 아니라 재생성 계약 변경이 될 수 있다. 모델/config/환경/입력 순서와 새 subprocess 경계를 봉인한다. 실제 [P1 v5](../p1_clean_regeneration_20260905_v5/report-source.md)는 새 학습→답안 생성/replay를 통과했지만 기존 답안 exact 복원은 실패했다. “재생성 PASS” 하나로 둘을 합치지 않는다.
- COMMON_SPEC 31–33행의 “물리 상수 외 모든 실수 금지”는 운영진 규정보다 강하다. 숫자를 **물리·구조/사전등록 hyperparameter/train-fit/금지 계보**로 분류하고 출처를 감사한다. JSON 위치만으로 적격성 판정하지 않는다. 모델 교체·최종 잠금·업로드/삭제·Git은 evaluator의 자동 부수효과로 넣지 않는다.

## P1 — 반기 4블록과 시간 방향은 같은 요구가 아니다

근거: [P1_SPEC](../../docs/ocean_v2_codex/P1_SPEC.md) 51–60, 75, 99행; [현재 P1 계약](../../00_MUST_READ_FIRST.md) 6–9행.

1. **분할 선택이 blocker다.** P1_SPEC은 미래→과거 학습을 허용하는 양측 leave-one-block-out이고 현재 계약은 blocked chronological/inner 선택이다. 배포 시리즈의 양방향 관측 특징이 허용된다고 미래 시기의 **학습 label**을 쓰는 검증이 forward와 같아지는 것은 아니다. 추천: 배포 2026으로의 시간 이전을 묻는 주평가는 forward, 양측 반기 평가는 별도 retrospective 진단으로 명명한다. 양측 4블록을 주평가로 택한다면 그 목적 변경을 명시하고 forward 결과와 같은 열에 합치지 않는다.
2. 반기 경계는 KST `[2024-01-01,2024-07-01)`, `[2024-07-01,2025-01-01)`, `[2025-01-01,2025-07-01)`, `[2025-07-01,2026-01-01)`로 제안한다. 실제 지원 행만 교집합으로 사용하고 마지막 관측이 12-11이라는 기록을 fold의 임의 종료시각과 혼동하지 않는다. forward에서는 H1_2024 이전 train이 없어 그 fold OOF는 만들 수 없다. 따라서 **forward + H1 두 블록 pooled 주평가 + 4개 fold 전부 OOF**를 동시에 보장할 수 없다. forward 선택 시 H1_2025만 seasonal 확인으로 남기고 warm-up 제외를 분모에 기록한다.
3. run은 `(station,layer)` 정확 10분 연속 양성 run으로 정의하고, calendar/year 경계 자체로 자를지 명시한다. 추천: 실제 gap만 끊고 run 시작 block에 단일 귀속한다. 다른 block에서는 같은 run 행을 반드시 제거해 중복 OOF를 막는다. purge와 겹친 run은 train에서 통째 제외한다. scoring rows와 context rows를 별도 정의하고, 이동된 run의 끝까지 포함한 실제 평가 footprint로 train 누출을 검사한다.
4. `f1_season`의 시즌이 **run 소유 H1 block**인지 **행 시각의 1–6월**인지 선택해야 한다. 경계 run을 통째 이동하면 두 분모가 달라진다. 추천: 주평가명을 `f1_calendar_h1`로 명확히 하고 모든 유효 OOF 중 행 시각 H1만 pooled TP/FP/FN으로 계산하며, run-owner 기준 결과를 부표로 둔다. 이 선택 역시 옛 `f1_season` 결과로 소급하지 않는다.
5. P1_SPEC 54행의 “음성 구간만 절단”은 label에 따라 stress 표본을 선택하며 test 분포 수치도 읽어온다. 추천: validation stress mask는 label/type와 독립된 시간 seed/사전등록 길이로 고정하고, 같은 retained rows에서 intact/stress를 함께 채점한다. train augmentation은 별도 변경이다. 무제한 ffill/bfill·전체 seg_len·run 길이 등은 21일 purge만으로 자동 보호되지 않으므로 split별 feature/decoder의 실제 최대 의존성 또는 차단 규칙을 고정한다.
6. **지원 확인 미완료:** H1_2024는 S-ORS만 있다는 설계 기록, G-ORS H1_2025 양성 72행이라는 P1_SPEC 99행 기록은 표본 한계를 시사할 뿐 새 감사값이 아니다. 필요한 사전 집계는 fold×station×layer의 train/validation 행·양성 run·정상일 수, purge 제외·경계 이동 행, 최장 feature support, bootstrap cluster 수다. 현 421,032행 Q2/Q3/Q4 OOF가 이 반기 지원을 대신하지 않는다. 수일짜리 run을 일별 독립 표본으로 취급하는 CI는 과신하지 않는다.

## P2 — 정확한 8블록, outage 분모, 0.29의 출처

근거: [P2_SPEC](../../docs/ocean_v2_codex/P2_SPEC.md) 4–6, 34–35, 67–72행; [기존 지원 집계](../p2_score_repair_20260905_v1/report-source.md) 34행. “2개월 8블록”은 정확하지 않다. 다음 **KST 반열림** 경계가 문서와 맞는 제안이다.

| block | 시작 포함 | 종료 제외 | 주의 |
|---|---|---|---|
| B1 | 2024-05-01 | 2024-07-01 | 5–6월 |
| B2 | 2024-07-01 | 2024-09-01 | 7–8월 |
| B3 | 2024-09-01 | 2024-11-01 | 가림 계절 analogue |
| B4 | 2024-11-01 | 2025-01-01 | 11–12월 |
| B5 | 2025-04-01 | 2025-06-01 | 4월은 부분 지원 |
| B6 | 2025-06-01 | 2025-08-01 | 6–7월 |
| B7 | 2025-08-01 | 2025-09-01 | **1개월** |
| B8 | 2025-11-01 | 2026-01-01 | 겨울 혼합 regime |

1. 기존 배포 train 지원 집계는 2024-05-01~2026-01-01, 학습 가능 target 166,268행이지만 **위 8블록×7일 purge 지원 수는 아직 확인되지 않았다.** 2025-09/10 hidden과 2026 padding은 target scoring에서 제외한다. 각 fold의 truth유효·공개 수온층≥2·층/수심 유효 조건을 모델 공통으로 고정한다. frac 목표의 작은 온도차 학습 제외가 scoring 분모 제외로 이어지면 안 된다.
2. `testmatched`는 `[block_end−17일,block_end)` T5 **temp+psal** joint mask로 명시하고 모든 baseline/lag/rolling/staleness를 다시 계산한다. 예컨대 B3는 10-15부터다. P2_SPEC의 배포 “10-14~10-31/17일” 서술과 달라 exact 시각·포함 규칙 확인 없이 그대로 복사하지 않는다. M1은 문서대로라면 T19/S19 `[2024-10-08,2024-11-01)` **24일**, S4 `[2024-09-01,2024-09-23)` 22일; M2 `[2025-08-15,2025-09-01)` 17일이다. 이들은 하나의 동일 outage가 아니다.
3. **Composite 분모가 blocker다.** `pooled_testmatched`에 이미 outage 행이 들어 있는데 M1을 다시 0.29 가중하면 stress를 중복 강조한다. 이 정의는 가능하지만 공식 기대 RMSE가 아니라 `stress_composite`다. 추천: 주평가는 배포 계절 analogue B3의 같은 key pooled RMSE를 유지하고, 8블록 pooled·M1/M2·Composite는 함께 보고한다. Composite를 주평가로 바꾸려면 현재 [P2 계약](../../01_P2_MUST_READ_FIRST.md)의 September–October 우선 원칙 변경을 명시한다.
4. 혼합 population 해석을 원한다면 non-outage와 outage scoring strata를 서로 배타적으로 정하고 `sqrt((1-p)·SSE_non/N_non + p·SSE_out/N_out)`로 계산한다. M1의 전체 B3행을 쓸지 **실제 mask 구간의 층별 scored rows만** 쓸지 먼저 선택한다. 추천은 masked interval rows만 outage 지표로 사용하고 전체 B3 stress RMSE도 부표로 제공하는 것이다. baseline과 후보는 같은 mask/keys/N를 공유한다. 여러 mask 변형은 같은 target을 복제하므로 독립 표본처럼 합산하지 않는다.
5. **p의 메타데이터 출처는 명시할 수 있다.** P2_SPEC 6·70행의 7,621/26,061은 각각 T5 결측에 해당하는 *query 행*과 전체 *query 행*이라는 기존 문서 주장이다. 산술로 `p=0.2924293004873182`이며 정확히 0.29는 아니다. 추천: 승인된 메타데이터를 쓰기로 정하면 exact 분수를 config에 저장하고 반올림 0.29와 혼동하지 않는다. 이는 leaderboard 점수 역산값은 아니지만 이번 작업에서 원자료 대조하지 않았으므로 **문서 유래·미재검증**이다. test_index 키만으로 T5 결측을 알 수 없고 공개 T5 가용성과의 join이 추가로 필요하므로 “test_index에서 단독 계산”도 정정해야 한다. 여기서는 공식 입력을 새로 읽지 않았다.
6. 목표 3층의 temp/psal은 validation 전 구간 joint mask, 공개 T5 mask는 별도 적용한다. 모든 특징을 fold별 격자에서 계산하고, 무제한 last/next가 있으면 7일 purge만으로 충분하다고 하지 않는다. 날짜·층별 N, 자연 T5 결측/새 mask 추가 수, 공개층≥2 미충족, M1/M2 독립 KST일 수와 mask 후 feature support를 0-fit 집계로 먼저 확인해야 한다. 현재 문서의 T5 전문가 재추진은 로드맵 §0/§3의 반복 금지 우선순위와 충돌하므로 새 평가 구현에 자동 포함하지 않는다.

## P3 — S_dense는 weighted RMSE로, greedy는 검증된 운영진 알고리즘이 아니다

근거: [P3_SPEC](../../docs/ocean_v2_codex/P3_SPEC.md) 5, 29, 43–53행; [현재 P3 계약](../../02_P3_MUST_READ_FIRST.md) 6–10행. 배포 README가 보장하는 것은 hs≥1.5, 모든 6리드 유효, 동일 정점 간격≥78h이지 정시 first-eligible greedy 자체가 아니다.

1. **시간 방향부터 선택해야 한다.** SPEC은 2024Q1~2025Q2 **UTC** 6분기 양측 LOBO지만 현재 P3는 earlier-only이다. 추천은 current chronological 계약을 유지하고 첫 분기는 warm-up, 이후 지원 있는 분기만 forward OOF로 평가하는 것이다. 이 경우 “모든 24,360 anchor/6분기 OOF”를 보장하지 않는다. 양측 6분기 실험은 별도 retrospective 이름과 명시적 계약 변경 없이는 실행하지 않는다. UTC/KST 경계를 섞지 말고 실제 anchor의 `[t−48h,t+24h]` 발자국과 같은 정점 episode 중복을 확인한다.
2. `p·S_onset+(1−p)·S_storm`은 각 S가 RMSE인지 error population인지 불명확하다. 추천 정의는 **`S_dense=sqrt(p·MSE_onset+(1−p)·MSE_storm)`**이다. 각 MSE는 선택된 anchor의 6리드 SSE/전체 리드행 수이며, 가중 RMSE와 `p·RMSE_onset+(1−p)·RMSE_storm`은 다른 지표다. latter는 쓰지 않는다. 동일 anchor의 6리드는 하나의 사례이고 bootstrap에서도 함께 움직인다. 원래 비가중 pooled RMSE도 반드시 남긴다.
3. `p`는 fold **training**의 정시 greedy anchor에서 `jc3` 비율을 산출해 고정하고 validation/test 비율로 맞추지 않는 것을 추천한다. `jc3`의 범위는 예컨대 `[t−3h,t]`의 유효 hs 중 min<1.5로 고정하고 최소 유효 슬롯 수·결측-only 처리·t 포함을 정의한다. unknown을 자동 storm으로 분류하지 말고 별도 수/처리 방식을 명시한다. 각 fold의 p와 두 stratum N이 달라지므로 pooling 시 `w_on=p_f/N_on,f`, `w_storm=(1-p_f)/N_storm,f` 및 fold 질량을 지원 anchor 수 비례로 고정하는 안을 추천한다. stratum이 비면 임의 재정규화하지 말고 지원 부족/대체 지표를 기록한다.
4. greedy는 **label-independent 선정 proxy**로 다음처럼 명세해야 한다: station별 시간 오름차순; hs≥1.5·6 target 유효·48h context 계약 충족·정시 후보 중 처음을 선택; 이후 마지막 선택 시각과 차이≥78h인 최초 후보를 순차 선택; station 간 상태 독립; 동률은 원 key 순서. training 내 p 계산은 training 구간만 사용한다. 평가 `S_greedy`는 전체 train-derived anchor에서 station-global 목록을 먼저 고정한 뒤 fold와 교차해 분기마다 spacing을 초기화하지 않는다. `S_storm`은 jc3=false 부분에서 같은 규칙의 **6h 간격**을 별도로 적용한다. 이 선택 순서·초기 phase에 따라 표본 수가 달라지므로 **272/1,250/5,600을 목표로 맞추지 않는다.**
5. episode도 “hs<1.5 또는 결측 6h”가 시작/끝 중 어디에 귀속되는지, 결측을 고요함으로 볼지, 정확 20분 18슬롯인지 elapsed 6h인지 고정해야 한다. 추천: station+timestamp interval의 deterministic ID, 6h 경계 포함 규칙과 missing break를 명시하고 train/validation 공유 episode는 제외한다. `(station,episode)` paired bootstrap에서 후보·기준·6리드·가중치를 함께 재표집한다. 보정/라우터/선택은 earlier-inner에서만 적합하며 all-OOF 미래 label을 쓰지 않는다.
6. **지원 확인 미완료:** README의 배포 wave 기간은 2024-01-01~2025-06-30이다. SPEC 24,360 dense/272 greedy는 기존 정찰 주장이고 새 계약의 검사값은 아니다. 현재 v4의 실측 181사례/181 station-episode·1,086행은 3 expanding folds의 49/79/53사례다([기존 결과](../p3_wind_only_dropout_20260905_v4/report-source.md)). 이를 6분기 지원으로 바꿔 적지 않는다. 필요한 0-fit 사전 표는 quarter×station별 eligible anchors, onset/storm/unknown, greedy 수, 독립 episode 수, 78h+episode 제외 후 train 수, target-ready 시각과 기상 지원이다.

## 코드 봉인 전 최소 완료 조건

다음 한 번의 **평가 계약 확정**에서 (1) P1/P3 forward 또는 two-sided 및 warm-up, (2) P1 run/season 소유와 stress mask, (3) P2 정확 block/mask 날짜와 primary/Composite strata·p, (4) P3 weighted RMSE/greedy/p/episode, (5) 공통 paired bootstrap·선택·fallback 규칙을 명시한다. 그 뒤에만 새 ID의 evaluator/config/합성 계약검사와 **train-only 0-fit 지원 집계**를 만들 수 있다. 지원 부족을 확인하면 그 사실을 보고하고 날짜나 gate를 성능에 맞춰 자동 변경하지 않는다. 현재 문서는 그 실행을 수행하거나 승인한 것이 아니다.
