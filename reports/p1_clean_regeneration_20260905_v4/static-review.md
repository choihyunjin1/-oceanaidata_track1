# 재생성 경로 정적 QA

2026-09-05. 실행 전 `validate-data` 검증 기준으로 원본/계보/분모/재현 주장을 분리했다.

- 기존 canonical runner는 `--output`을 받지만 report 위치는 고정이다. 이를 직접 호출하지 않고 새 runner/새 report를 사용해 기존 영수증 덮어쓰기를 방지했다.
- 실제 학습 호출은 신규 `_fit_model` 및 `LGBMClassifier.fit`뿐이다. 학습 함수에 `joblib.load`, 기존 모델·답안 경로는 없다.
- Q4 inner 범위는 배포 train에서 2025-07-12 이상/09-10 미만이며 training은 06-21 미만이다. 기존 21일 purge와 split 정의를 유지한다. 전체 모델의 학습에는 배포 train의 모든 776,706행을 쓴다.
- 특징은 raw 7열만 통과하며, 전처리 및 spike scale은 해당 training에서 적합한다. inner label은 정책·threshold 선택에만, full label은 전체 학습에만 사용한다.
- O/XGB 파라미터는 `configs/p1.toml`; B/LGBM은 `_lgb_parameters`가 기존 recipe의 `lightgbm_parameters`만 읽는다. 같은 장문 JSON 안의 과거 CSV·cache·score 관련 메타데이터는 실행 입력으로 이어지지 않는다.
- 이전 점수·답안은 모델 선택, 임계값 적합, 보정에 사용하지 않는다. 이전 공식 답안 SHA 문자열은 **출력 후 동일성 확인용**일 뿐이다.
- 4fit 완료 후 recipe/source/model SHA가 맞아야 별도 프로세스 추론을 허용한다. sample은 key 4열만 읽으며 label·baseline·score.py는 사용하지 않는다.
- 빈 03_model에서 실제 학습한 증거와 새 모델을 재로딩한 증거를 별도 JSON으로 기록한다. 같은 환경·같은 저장소에서의 새 학습이며, 독립 머신/차단망/복사 코드 실행이나 최종 제출 ZIP PASS는 아직 아니다.

합성 계약검사: 빈 폴더 fail-closed/기존 파일 보존, key order 정렬 및 중복 거절, inner threshold 선택과 tie-break 3건 PASS. Ruff PASS. 실제 데이터 수치·답안 재현 검사는 종료 후 별도 판정한다.
