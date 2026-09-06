# P1 28.9점 원형 복원 — 사용자 승인 후 실행

목표는 낮아진 기준 대비 작은 개선이 아니라 **과거 고득점 모델의 재학습 가능한 부품을 복원**하는 것이다. 과거 F1 .833548/28.909341은 역사적 목표이며 새 답안에 승계하지 않는다. 현재9031/F1 .785944/27.644124 패키지를 변경하지 않는다.

## 실행 순서

1. O/B feature·seed·가중치·decoder·라우터 선택 출처와 MS-TCN e150 학습/추론 경로를 독립 정찰한다. old prediction CSV/수동GI패치/공개점수 역산을 입력으로 사용하지 않는다.
2. 재사용할 레시피/코드와 이번에 다시 적합할 정책을 구분해 원장을 작성한다. 원형과 다른 부분은 명시하며 조용히 바꾸지 않는다. 실행 계약·합성 테스트·자원/시간 상한을 학습 전에 봉인한다.
3. 적격 트리 경로를 학습하고, O/B/라우팅을 같은 내부 행에서 대조한다. 셀별 정책은 학습용 OOF에서 코드로 산출하고 성능평가와 분리한다. 과거 정책을 동일 OOF에 맞춰 재현한 수치는 독립 검증으로 부르지 않는다.
4. MS-TCN은 저장 경로/추론 경로를 연결한 별도 재생성 모듈로 분리한다. 원형 e150/3seed와 trial18은 다른 레시피이므로 혼동하지 않는다. GPU 시간·내부 평가 가용성·출처 확인 전 장기 학습을 시작하지 않는다. 학습 실행은 원본 one-shot이 아닌 새 ID이다.
5. 새 PID replay·독립 QA 후 새 답안을 준비한다. 공식 제출은 exact SHA·현재 잔여기회/기한 확인 후 승인범위에서만, 최종 모델 잠금/commit/push는 하지 않는다.

## 발견한 차이 / 아직 미확정

- clean v5는 B를 1 seed로 줄였고 train-only year-depth dictionary, plateau168h cap, final60day 정책선택을 사용한다. 원형과 동일 모델이 아니다.
- 원형 static router의 재구성 코드는 있지만 원 선택 알고리즘은 확인하지 못했다. 새 inner validation에서 B/O/AND/OR를 선택하며 동점은 B, 미지원 셀도 B로 고정한다. 과거 셀 상수를 복사하지 않는다.
- 옛 final `train_model.py`의 기본 출력은 `03_model/retrained_from_scratch`, predictor는 `03_model/weights`를 읽어 자동 연결되지 않는다. predictor의 과거 router CSV/GI patch/옛 answer SHA 요구를 제거해야 한다.
- MS-TCN이 실제 새 학습 후 불일치했다는 영수증은 아직 확인되지 않았다. deterministic 설정 부재는 위험이며 실패 실측과 구분한다.

## 08:48 KST 감사 후 사전 실행 계약

- 트리: XGBoost O 1 seed와 LightGBM B 3 seeds(20260813, 20260829, 20260847). B 원형은 700 trees/leaves63/min-child60/event-day weighting이다. 현재 80열 train-only depth dictionary·168h plateau cap·train-scale rules를 유지하므로 과거 답안의 exact 복원이 아닌 계열 복원이다.
- 트리 내부 적합: Q2/Q3/Q4, 각 earlier inner 60일/21일 purge/양성 run 소유 블록 보존. 각 inner/outer에 O1+B3를 적합하여 총24fits. 전역 임계값과 셀 정책은 inner에서만 적합한다. 전체 학습4fits는 결과/QA 후 별도 실행하며, 추가 선택 학습 없이 Q4 inner selector를 재사용한다. 이 정책의 시간분포 한계도 보고한다.
- 이 별도 복원 진단의 1차 비교는 Q3+Q4 pooled F1이다. Q2(2025 H1 일부), 각 fold, 전체3fold 수치도 보고한다. 기존 H1 전체 평가 계약을 바꿨다고 주장하지 않는다. Q2에서 e150이 선택되었고 Q3/Q4도 이미 노출된 자료이므로 어느 수치도 fresh confirmation은 아니다. 동일 키 대조와 중복 소유권 검사가 선행되어야 한다.
- MS-TCN e150 선택 근거는 `p1_mstcn_checkpoint_diagnostic_20260827_v2`에 실제 존재한다. 기존 '선택 근거 없음' 해석은 정정한다. 과거 Q3/Q4 pooled F1 .902917→.906804(Δ+.003887), Q4 Δ−.015441, CI90 [-.013148,.021144]는 **과거 router와 결합한 결과**이며 새 router에 승계하지 않는다.
- MS-TCN 신규 전체 재학습은 width512/3seeds(20260827,20260839,20260863)/e150/bf16/원 cosine horizon300을 유지한다. 당시3fits 약109.74분. 수치 연산 계약을 몰래 바꾸거나 새 학습의 과거 SHA 일치를 보장하지 않는다. 새 모델 저장·추론 연결과 새 PID replay를 별도로 검증한다.
- 자원: 트리 CPU4, MS-TCN CPU2/GPU0 한 작업. 트리 내부90분, 전체 실행6시간 상한. 실제 fit 전에 각 코드·계약·합성 테스트·Ruff·source pin을 검토하고 새 lock으로 실행한다. 성능 기반 반복 재시작은 하지 않는다.

## 경계

배포 train만 학습/평가/정책 적합에 사용. hidden/외부 관측/과거답안학습0. metadata와 원천코드만 감사; 원본·봉인실험·소비 lock·현재 fallback 보존. 내부 구간은 반복 노출된 retrospective 평가다. 모든 월 양수/anchor제거0 같은 옛 과도한 gate를 자동 복원하지 않는다. 성능과 재현성·규정 적격성은 별도 판정한다.
