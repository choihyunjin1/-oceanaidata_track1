# 추가 검증과 정보 확보용 제출 — 실행 전 판단

2026-09-05 KST. 사용자의 요청: “검증하세요 한번 제출해보는 것도 방법이긴합니다.”
기존 당일 답안 제출 승인 범위에서 내부 결과를 검증하고, 새로운 질문에 답하는 완성 후보만 대회 답안으로 비교한다.
최종 모델 잠금·Git commit/push는 하지 않는다. 이 기록은 새 공식 점수를 확인하기 전에 작성했다.

## P1 — INFO_ONLY 완성 정책 1회 비교

- 선택: 이미 학습한 `p1_depth_contract_repair_20260905_v2`의 year-safe depth, final-earlier-inner **balanced/threshold0.2/decoder OFF** 그대로.
- 질문: train2024–25→test2026에서 수심 범주가 전부 결측으로 바뀌던 기존 계약을 제거한 완성 정책이 실제 배포 평가에서 어떤 결과를 내는가?
- 한계: 기존 공식 control은 Q4-inner union O0.2/B0.3이다. 수심뿐 아니라 최종 선택 정책도 다르므로 순수 수심 효과 인과실험으로 해석할 수 없다.
- 내부 pooled F1은 0.851174240→0.848961444(−0.002212796), Q4도 악화했다. 개선 예상이나 최고점 보장은 없다. 현재 공식 기준 F1 0.790733 / 27.771400점, 새 예상 공식 점수 미산정.
- 신규 학습0. 별도 v3 inference adapter에서 기존 학습/모델/recipe 해시를 확인한 뒤 test 공개 관측과 sample 키만 읽는다. sample label/hidden truth 접근0.
- 169,011개 정확한 키·순서·이진 label·중복·hash, 별도 프로세스 byte-exact CSV replay와 독립 코드 리뷰를 통과하고 기존 SHA와 다를 때만 P1에1회 업로드한다.
- G-ORS2026 raw depth 자체가 결측이라는 원본 README 예외는 남는다. 모든 수심 정보를 복원했다고 주장하지 않는다.

## P2 — 결측 조건부 보완의 별도 검증

- 기존 A/B 결과를 바꾸지 않는다. 공개 `temp_5` 또는 `psal_5`가 비유한이면 기존 첫-seed R, 나머지는 C라는 단일 규칙을 봉인한다. 목표 layer2/3/4의 가림은 trigger가 아니다.
- 기존 C/R3fold 모델6개를 재사용해 신규학습0으로 시작한다. intact에서 자연 결측 때문에 rule이 켜질 수 있으므로 무조건 무변경이라고 가정하지 않는다.
- 기존 Oct18–Nov1 스트레스는 이미 노출된 development로 분리한다. 달력으로 사전 고정한 autumn/summer/winter × 3/7/14일9episode를 추가 평가한다.
- 목표 수온·염분을 특징 생성 전에 제거한다. 원래 평가행을 유지하고 추가 mask로 공개 수온 지원2개 미만이 생기면 몰래 행을 삭제하지 않고 scenario 지원 한계를 기록한다.
- 같은 historical 관측에 추가 mask를 적용하는 것으로, 독립 fresh 관측/확증적 일반화를 확보한 것은 아니다. 내부 결과를 본 뒤 rule/threshold를 변경하지 않는다.
- 제출할 만한 근거가 생기면 동일 frozen 정책의 추가 seeds/fullfit/공식 키 QA 비용을 root가 확인한다. 이번 첫 단계의 공식 입력/CSV/upload는0이다.

## P3 — 추가 제출 안 함

A global-bias/simplex는 내부 평균 악화이며 비-no-op 배포계수를 적합하지 않았다. B two-seed/episode 정책은 세 fold와 여섯 lead 모두 기존보다 악화했고 full4+router1도 실행하지 않았다.
기존 no-op 재제출의 정보는0이다. 이번에는 P3의 소비된 실험을 재개하거나 새 보정계수를 공식 점수로 정하지 않는다.

## 라이브 운영 정보

- 공식 문제 화면에서 이번 턴 확인: **P1 오늘2/3, P2 오늘2/3**. P3 남은 횟수는 이번 턴 재확인하지 않아 현재값을 단정하지 않는다.
- 로그인된 [제출 안내/일정 공지](https://oceanaidata.org/app/notices) 두 건 모두 예선 마감일을 **2026-09-07**로 명시한다. 공지에서 정확한 시각은 확인되지 않았다. 오늘9월5일을 전체 대회 마지막 날이라고 부르지 않는다.
- 답안은 문제당 하루3회. 최종 모델 제출은 이후 답안 업로드를 잠그므로 누르지 않는다.
- 공식 점수는 사전 고정 완성 정책 비교에만 사용한다. 정답·Public membership·계수·threshold 역산0.

최종 결과와 실제 제출 receipt는 작업 완료 후 별도로 기록한다. 이 문서는 제출 완료 증명이 아니다.

## 별도 P2 3-seed 단계 승인 기록 — 22:03 KST

첫 0-fit 단계의 `RESEARCH_ONLY_NOT_READY` 결과·runner·gate는 보존한다. 추가 결측으로 공개 수온 지원이2개 미만이 된4행은 합성 scenario의 지원 한계이며 공식 입력의 실패로 단정할 수 없다. 가을 intact의 작은 이득과 결측 스트레스의 이득을 동일한 당일3-seed control에 대해 검증하기 위해 별도 `p2_missingness_conditional_3seed_20260905_v3`를 승인했다.

- 같은 R rule·recipe·60epoch·seed20260901/02/03을 고정한다. 기존 C historical9/R historical3/C full3는 exact hash 재사용한다.
- 새 R historical6+full3, 총9fit/15분 예산. 먼저 C/R 각각3-seed 평균 후 같은 공개 결측 trigger를 적용한다.
- 원래 intact/episode/지원 불가 결과를 빠짐없이 기록한다. 겨울의 원래부터 결측인 episode들은 독립적인 추가 결측 실험이 아니다.
- 이 단계의 공식 입력/CSV/upload는0이다. 학습·QA 완료 뒤 root가 별도로 공식 배포 후보 준비도를 판단한다. 최종점수 보장 또는 이전 gate 통과로의 변경이 아니다.
