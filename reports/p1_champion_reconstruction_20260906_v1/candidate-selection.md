# 후보 선택 — 트리 O/B 합집합 ∪ 원형 MS-TCN e150

2026-09-06 09:23 KST. **새 답안 준비 대상으로 전역 O/B 합집합과 MS-TCN의 OR를 선택한다. 현재 공식 점수의 승계나 최종 모델 지정은 아니다.** 새 MS-TCN 전체 학습과 두 구성 요소의 독립 재실행이 완료되기 전에는 공식 입력을 읽거나 답안을 생성하지 않는다.

## 근거와 비교의 한계

- 트리 내부 24fits는 657.652초에 완료했다. 새 PID 재실행 36/36 exact 및 독립 QA 171/171 PASS. 기본 단위는 원 분기 소유권을 보존한 같은 검증 행이다.
- 원형 MS-TCN의 **기존 학습 구간 밖 Q3/Q4 제안**을 새 트리의 같은 287,862행에 결합했다. 과거 공식 답안은 입력이 아니다. 고정 e150/3seed, threshold .8/.4, snap12, min19를 변경하지 않았다.
- 사전 지정한 1차 비교인 Q3+Q4 pooled F1에서 트리 합집합 .889264149 → 결합 .906966(표시 반올림), Δ 약 +.017702. 7일 블록 paired bootstrap CI90 약 [.000335,.036506], 양의 차이 비율 .954. 이 비율은 설명적 재표집 통계이며 미래 개선 확률이 아니다.
- 네 개 고정 트리 arm(O, B, 합집합, 새 inner 셀 라우터)을 모두 보고했으며, 결합 후 1차 F1이 가장 높은 전역 합집합을 선택했다. 새로운 셀/행 상수를 만들지 않는다. 선택한 전역 arm 자체는 반복 노출된 Q3/Q4 결과를 이용한 모델 선택이며, 동일 수치는 **선택 후의 독립 검증이나 fresh confirmation이 아니다**.
- 모든 MS 결합 arm에서 Q4는 악화했다. 최악 7일 블록과 정점·층의 악화도 숨기지 않는다. 위험은 자동 탈락시키지 않되, 기존 공식 27.644124점/SHA9031 후보는 그대로 보존한다.
- 과거 약28.9점은 다른 공식 CSV의 점수다. 내부 F1 .906966을 공식 F1 또는 공식 점수로 치환하지 않는다. 현재 특징·임계 선택 절차 등도 과거와 달라 정확한 옛 답안 복원을 보장하지 않는다.

## 원본 결과와 QA

- 트리: `artifacts/p1_champion_reconstruction_20260906_v1/tree_historical/terminal_result.json` (SHA256 `00e00d2b11ec1547619a943aa31a496a8d9c2f3afe9c649bb4708d395267e8b2`), 같은 폴더 `fresh-replay-qa.json`, `independent-qa.json`.
- 결합: `artifacts/p1_champion_reconstruction_20260906_v2_union_evaluation/result.json` (SHA256 `24ca88594c4cdb5d5dc2c3ae5defa1415823f8cfcebe463820b140dd9376146f`), 같은 폴더 `independent-qa.json`(384checks), `independent-bootstrap-qa.json`(793checks).
- [고정 결합 계약 v2](union-contract-v2.json). v1의 달력 분기 유도 오류는 원래 Q3에 속하는 10월119행을 잘못 옮긴 검증 구현 오류였다. v1 실패를 보존하고 v2에서 원래 fold provenance/hash를 사용했다. 행 제거·예측 변경·임계 재탐색 없이 전체 키를 대조했다.

## 다음 실행과 권한

1. 이미 시작한 새 전체 트리4fits와 MS-TCN3fits/e150를 그대로 완료한다. 추가 fit/재시작 없음.
2. 전체 모델의 새 PID replay와 QA를 exact result SHA로 연결한다. MS 저장 모델 재실행 PASS와 scratch 반복 학습의 결정론 보장은 구분한다.
3. root의 별도 post-QA decision JSON에 `tree_arm: union`과 모든 영수증 해시를 넣는다. `materialize.py`의 tree/mstcn/combine은 각각 새 Python 프로세스로 실행한다.
4. schema·전체 키·순서·유한값·중복·해시·train→answer 계보·6시간을 확인한다. source-to-answer 원 시작 시각은 09:00:49.700005 KST, 마감은 15:00:49.700005 KST다.
5. 기존 일반 답안 업로드 승인 범위에서 현재 UI의 기회/기한을 새로 확인한 뒤 이 후보를 채점한다. 최종 모델 잠금·commit·push는 하지 않는다.

이 문서는 선택 기록이다. 완료/CSV 생성/공식 채점/portable cleanroom 통과를 주장하지 않는다.
