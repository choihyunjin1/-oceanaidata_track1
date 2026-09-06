# 추가 개선 및 제출 재생성 진행 기록

현재 작업은 진행 중이다. 준비된 파일과 전체 cold 검증 완료를 혼동하지 않는다.

## 현재 완료된 부분

- P1 오류감사: 신규 fit0, 과거 exact 287862키의 고정 tree∪MS F1 0.9069664687. FN1448 중 drift/offset1293, 실제 양성구간 내부 FN1247. 세부 결과·83-check 독립 QA는 `reports/p1_composed_error_audit_20260906_v1/report-source.md`가 canonical이다. 공식 답안의 오답을 안다는 뜻이 아니다.
- 최신 기존 후보 P1 b2f17/P2 fee6118/P3 ff42a6 및 receipt 해시 재확인. `initial-candidate-inventory.json` 참조. 이 검사는 새로운 수치 검증 또는 공식 점수가 아니다.
- 공통 ZIP 분할·복원 및 무손상 inventory 합성검사15 PASS, notebook generator/executor 합성5 PASS. 별도 kernel 실행과 봉인 원본 notebook 미변경을 확인했다. 실제 수치 학습 검사는 아래 별도 작업이다.
- 기존 환경 `pip check` PASS. 신규 OS/신규 venv/오프라인 설치나 필요한 전체 wheelhouse는 검증하지 않았다.

## 독립 P3 코드 전용 패키지

원 연구 폴더 밖:
`C:/Users/cedis/Documents/OceanFinalCandidates_20260906/P3_numeric_source_v1.zip`

SHA256 `9f5e0b262ad2bd92ad781c0173bb2167173cfc35cacfa17d6dd8b4318abda1d6`.
코드·환경 버전·학습 출처·TRAIN/PREDICT 노트북을 포함한다. 관측 데이터, 모델, 답안, OOF와 attempt lock은 포함하지 않는다. `p3-source-archive-inventory.json`은 파일 목록/경계/CRC 검사이며 실행 PASS가 아니다.

실제 새 추출 실행 경로:
`C:/Users/cedis/Documents/OceanFinalCandidates_20260906/P3_numeric_execution_v1/P3_numeric_cold_v1`

prepare PID7924, 시작 Unix1788678399.9087498, 전체6h deadline은 이 시작 기준이다. CPU2, 현 단계 fit0/GPU0. `P3_DATA_DIR`는 원 배포 데이터, `P3_DENY_REPO`는 연구 저장소를 지정했다. Python runtime 폴더만 예외이며 OS 방화벽 인증이 아니다. prepare 완료 후 GPU 배정 전 train을 자동 시작하지 않는다.

prepare는 345.433초에 `PREPARED_SOURCE_ONLY`로 완료했다. 24360 anchors/591features/45 raw-context 비교 PASS, train SHA 전후 동일, old cache/official 읽기0, 03_model 빈 상태다. 실제 train17fit/QA/답안은 아직 미완료다. 원 6h 타이머는 계속 유지된다.

## P2 28fit 완료 결과

신규28fit 767.375초, 48개 모델 전체 natural/outage 새PID replay43.547초 exact, 독립425 checks PASS. Root도 별도 scalar fsum으로32 checks를 통과했다(`p2-root-independent-qa.json`). B3 가을3seed 주평가의 0.488284326→0.483505057℃는 원 결과와 exact 불변이다. 새 전체8fold3seed 보조평가는1.229592481→1.251771599℃로0.022179118℃ 악화했다. 이 결과를 전반적 개선이나 주평가 교체 근거로 포장하지 않는다. 실제 결측구간 등 별도 위험표와 모델 계보는 `reports/p2_c3_multiseed_completion_20260906_v1/report-source.md`에 연결한다. 새로운 full 후보 선택/공식 점수는 없다.

## 진행/대기

1. P2 나머지28fit → 전체48모델 replay 및 QA. B3 가을 주평가와 기존 배열 유지, full8fold3seed는 보조.
2. P2 외부 새 ZIP 추출에서 L120 cold3fit → TRAIN/PREDICT 실제 notebook 실행 → fee6118 대조.
3. P3 고정 earlier-inner 학습률/학습량 비교. 최대20fit/60분, root GPU 배정까지 fit0.
4. P1 새 cold11fit와 saved exact replay. 약2시간 예상, 모델·원 답안 유지.
5. P3 독립 cold17fit와 새PID 전체OOF/답안 replay, 저장모델 패키지. 별도 GPU 배정 필요.

16:20 갱신: P2 source-only ZIP의 실제 외부 추출·TRAIN/PREDICT 노트북 실행이 끝났다. 신규3fit122.235초, 전체cold173.010초, 전체노트북174.375초. 26061행 SHA fee6118은 기존 후보와 동일하고 새PID replay PASS다. 영구 보존은 `C:/Users/cedis/Documents/OceanFinalCandidates_20260906/P2_L120_verified_v2`이다. 원시로그/lock이 보존된 이 로컬 검증 폴더와 업로드용 ZIP을 구분한다.

P3 60분 tuning v1은 첫2fit 이후 자원예측65.69분으로 RESOURCE_STOP했다. 성능 열람/선택/outer 평가 전에 중단됐으므로 후보 실패로 판정하지 않는다. 측정 시간만 근거로 새90분 계약을 준비하며 기존2fit은 재사용, 최대18신규fit, 원래 총20fit/recipe/선택 표면 유지. 원 시도는 보존하고 실제추론/fit은 새GPU배정까지0. P1 saved→cold GPU 실행이 우선이고, 이후 P3 original numeric cold→budget90 순서다. 상세 사전 계약은 plan.md와 문제별 봉인을 따른다.

각 실행은 새 경로와 새 계약을 사용한다. 기존 실패/소비 lock 삭제·재시작, 성능 기반 중도 변경, 외부 관측/hidden truth, 리더보드 역산, Git commit/push/최종잠금은 없다.

## 제출 및 남은 공백

브라우저는 기존 `Debugger unattached` 및 차단된 재시작 상태로 알려져 있다. 연결 복구 없이 제출 성공·현재 잔여횟수를 주장하지 않는다. 이미 받은 일반 업로드 승인을 재요청하지 않되, 실제 연결과 quota/중복 SHA 확인을 거친다. 신규 후보는 공식 미채점이다.

최종 제출 경로 안내에서는 source ZIP(학습), saved ZIP(저장모델 추론), CSV(문제별 리더보드 업로드)를 명확히 분리해야 한다. 파일 크기 분할 helper의45MB 기준은 보수적 로컬 기준으로 현재 홈페이지 업로드 제한의 실측값은 아니다. 모델 포함 최종 패키지·≤6h cold 실측·오프라인 의존성·현재 최종 제출 양식은 검증 상태별로 별도 기록한다.
