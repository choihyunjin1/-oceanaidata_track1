# Portable 코어 및 P1 학습량 검증 — 실행 계획

2026-09-06 사용자 통합 계획 승인으로 실행. 기존 봉인 실행과 공식 패키지를 변경하지 않는다. commit/push/upload/final lock 없음.

**2026-09-06 04:17 KST 완료 상태:** P1/P2/P3 기준 답안 재생성, P1 0-fit 깊이 진단, P3 중단 복구 및 최신 ZIP 실제 재생 검증까지 완료했다. 아래는 보존된 실행 계획이지 재시작 지시가 아니다. 확정 결과는 [통합 보고서](report-source.md), 선택 파일은 [사용 안내](../../docs/ocean_v2_codex/PORTABLE_PACKAGE_HANDOFF_20260906.md)를 따른다. 새 P1 bracket-only 4-fit은 미실행 후속 제안으로 남긴다.

## 소유와 자원

- Root: P1 portable 기존 clean v5 CPU4, 빈 독립 경로 2회 × inner2/full2 = 최대8 fits, 각 실행3600초 상한. 제출 SHA `5971e145…`는 검증 metadata만이며 mismatch 후 재튜닝/강제복원 금지.
- P1 agent: 새 양측 inner/outer 계약 및 700 대비 학습곡선, CPU2·최대60분. 다음 HPO8×2설정은 결과·자원계획 보고 후 별도 단계. 공식 I/O0.
- P2 agent: clean v6 portable C3. 기존GPU recipe를 유지한 재학습, 짧은GPU학습이 끝나면 P3에 양도. 새copula기술수정은 새ID·계보일치 기존40fits 재사용 가능성 우선.
- P3 agent: clean v4 portable, run_a/run_b 각8backbone+3router, GPU독점순차. 약55분은 과거 실측의 계획 추정이며 실제시간 별도. 과거CPU90분 resource-stop을 재개하지 않는다.

## 동일성 / 경계

모델을 지우지 않고 새로운 빈 폴더를 만든다. 코드/환경 고정 → 합성 parity 및 파일경계 검사 → 배포train만 학습 → 같은 실행 산출모델에서 official inference → 별도PID replay → 독립 전체재학습 비교. 원본 train/test/README의 허용파일만 사용하고 sample은 키만 읽는다. 숨은 정답 접근0.

네트워크 Python audit hook 및 원repo 접근 deny와 실제 import 경로를 검사하되, 이를 OS방화벽/타머신 환경에서의 운영진 재현PASS로 확대하지 않는다. 사용한 로컬 의존환경은 명시하며 원프로젝트 코드 의존과 구분한다. 패키지내 코드의 상수는 선고정recipe/문제정의/학습산출물 출처로 분류한다. 파일형식JSON만으로 적격성을 주장하지 않는다.

## 완료 기준

각문제 canonical report와 code/model/answer/환경manifest, 빈폴더학습 및 재학습/SHA/최대수치차이/시간 영수증을 통합한다. 불일치 허용오차를 결과에 맞춰 넓히지 않으며 판정을 그대로 기록한다. 이전 답안이 점수상 열세여도 재생성 fallback으로 보존한다. 새 bracket/범위/정점층정책 후보와 기준 패키지를 혼동하지 않는다.

## 사용자 일시중지 후 재개 — 2026-09-06 03:40 KST

사용자의 재개 지시에 따라 기존 process/terminal/partial artifact를 먼저 확인했다. P1 깊이 스트레스 v1과 P3 v2 run_a는 프로세스가 없고 정상 terminal/failure receipt도 없다. 중지 요청 시각 부근의 부분 산출물은 있으나 정확한 OS 종료 사유가 없어 코드 오류나 wall-cap 도달로 단정하지 않는다. 기존 시도와 lock은 그대로 보존한다.

- P1: 별도 v2 ID에서 학습 0회로 부분 확률 18개를 검증·재사용하고 마지막 fold 확률 9개만 계산한다. 중단 전 개별 digest 원장이 없었으므로 현재 acceptance manifest만으로 과거 무결성을 주장하지 않는다. 완료 QA에서 고정 모델의 별도 PID 예측으로 재사용 확률도 exact 대조한다. 이어가기와 QA 합계 900초 상한, 공식 입력 0. 새 HPO/후보 학습은 시작하지 않는다.
- P3: run_a는 backbone 7개 성공 및 8번째 full multi 시도 중단으로 기록한다. pristine run_b의 전체 학습이 PID 21764로 시작되어 그대로 유지한다. GPU 완료 후 별도 recovery ID에서 검증된 산출물을 재사용해 full multi 1개와 full router 1개만 완성한다. 과거 historical router 2개의 실제 실행·산출 계보를 확인하고 미확인 재학습을 숨기지 않는다. recovery는 빈 폴더부터 중단 없이 수행한 재현 증거로 세지 않는다.
- P1/P2의 이미 완료된 패키지 검증은 반복하지 않는다. P3 GPU 독점/CPU2, P1 CPU2를 유지하며 기존 파일 수정·업로드·Git 커밋/푸시는 하지 않는다.

## P1 추가 검증 분기 — 02:54 KST, 현재 봉인 실행과 별도

정적 코드에서 nominal_depth_m/depth_regime의 학습 통계 키가 station/year/layer이며, 미지원 연도에서 숫자 NaN·unknown 범주가 됨을 확인했다. unknown 문자열이 encoder 학습에서 없었을 때 −1이 된다. raw depth는 남아 있으므로 수심 정보 전체 손실로 과장하지 않는다. 추가 HPO를 시작하기 전에 합성 differential로 이 표현 차이를 확인한다.

검토할 저비용 다음 단계는 **0-fit, 배포연도 반사실 스트레스**다. 이미 적법하게 학습한 동일 outer 모델과 source 검증행/labels/keys/달력특징을 유지하고 깊이 메타 특징 두 개만 unknown-year로 재현한다. 비교안은 train stats에 있는 해당 station-layer의 연도별 median의 median을 고정 fallback으로 사용한다(새 official/statistics/labels 입력 없음). 정책/threshold를 이 스트레스의 outer 점수로 다시 고르지 않는다. 표준 양측 OOF와 스트레스 수치를 별도 보고하고, 후자를 미래 성능 또는 공식 점수 개선으로 주장하지 않는다. 먼저 자원/구현 feasibility를 판단한다. 기존 패키지와 sealed 실험은 변경하지 않는다.
