# P3 장기 저장 모델 추론 패키지 — 2026-09-06

## 결론

실제 ZIP을 원 저장소 밖의 새 폴더에 풀어, 세 저장 모델만으로 1,200행 답안을 재생했다. cold 답안 SHA `d45605922ae8ce699c07405d8361765290c39bd38ddd0123f4e2e9ccd01808c5`와 정확히 일치하며 추가 학습은 0이다. 저장 추론은 6.088초, 부모 프로세스를 포함하면 6.308초였다. 합성 테스트 35/35, Ruff 및 추출 재생 독립 검사 28/28이 통과했다. 이는 저장 모델 추론 재생의 성공이며, 새로운 cold 학습이나 다른 OS/차단망 재현을 검증했다는 뜻은 아니다.

## 역할과 경계

`scripts/portable_20260906/P3_forward_saved_v1/`은 장기 저장 모델용 별도 추론 경로다. 원래 cold 패키지의 6시간 학습 시계를 바꾸거나 회피하지 않는다. 새 학습·GPU·업로드는 없다. 최종 제출 모델을 잠그지 않는다.

허용한 실행 입력은 full_single.cbm, full_multi.cbm, full_router.joblib 세 모델, 선택 특징명·고정 recipe·복사된 소스, 주최측 공개 test_context.parquet/test_index.csv 두 파일이다. 과거 OOF, historical 모델, feature cache, replay probe, 기존 답안 값은 요구하지 않는다. builder는 cold의 작은 QA/manifest/columns metadata와 세 full 모델 SHA만 연결한다. ZIP에는 원자료·답안 CSV·OOF·probe·로그·attempt lock을 넣지 않는다.

## 검증

- source-pins 12개와 cold manifest를 정확 대조한다.
- cold source-only prepare, 12 backbone/5 router fit 완료, 새 PID training QA, 1,200행 답안 및 다른 PID의 exact answer replay를 모두 확인한 후에만 build한다.
- 세 모델의 native feature 순서, 범주형 위치, 700/1,200 tree, seed, 고정 router와 0.2 persistence shrink 계보를 확인한다. 새 튜닝이나 Public 역산은 없다.
- case-major 예측을 공개 index의 원래 키·순서에 결합한다. LF 직렬화, finite/0–30m, 공개 입력 전후 SHA, cold 답안 SHA가 모두 일치해야 CSV를 쓴다.
- 새 worker에 600초 제한을 둔다. 부모 timeout 시 Windows process tree를 종료하고, 자식도 자체 종료 타이머를 가진다. 기존 출력 폴더는 거부한다.
- 합성 검증은 200개의 가짜 case/1,200개 가짜 key, native-model stub, 가짜 모델 bytes만 사용했다. guard 단위 테스트는 audit callback 검사이며 실제 프로세스 격리 검증을 대신하지 않는다. Python-level network deny는 OS 수준 완전 차단망이 아니다.

정확 명령과 파일 hash는 [code-qa.json](code-qa.json)에 기록했다. 합성 테스트 개발 중 보조 파서 오류 한 건을 고쳤고 최종 35개 모두 통과했다. 원 실험 재시작은 없었다.

## 실제 검증 결과와 제공 파일

P1 담당의 cold 12 backbone+5 router, 새 PID training QA, 공식 답안 및 또 다른 PID exact answer replay 완료 후 root의 승인 범위에서 ZIP을 만들었다. `P3_DENY_REPO`를 원 연구 저장소 경로로 설정하고 저장소 밖의 새 추출 폴더에서 preflight→새 output/새 PID 추론을 정확히 1회 실행했다. 실제 Python PID는 27384, Windows venv launcher는 36916, supervisor는 35112였다. 기존 `.venv-p1` 라이브러리 환경은 사용했으므로 새로운 OS/설치 환경은 아니다.

저장 모델 ZIP은 `artifacts/portable_cleanroom_20260906_v1/P3_forward_saved_v1.zip`이다. 4,061,290 bytes, SHA `56d4f5adbaea70a1ac6d0a2d962727ab0ba3ba4217d7feeaef17e139a695b9a3`. 파일 22개 중 모델은 정확히 3개이며, 원자료·CSV·OOF·probe·로그·lock payload는 없다. ZIP의 README와 `02_code/infer.py`가 장기 추론 입구이며, 전체 학습은 별도 cold 패키지의 입구를 사용한다.

구체적인 hash, 최초 receipt SHA, PID, 공개 입력 전후 SHA, 모든 28개 검사는 [extracted-replay-qa.json](extracted-replay-qa.json)에 보존했다. `code-qa.json`과 최초 build receipt의 PENDING은 해당 시점의 이력이며, 현재 실제 재생 판정은 이 최종 receipt를 따른다. 새 학습·재시도·튜닝·sample/hidden 접근·업로드·최종 모델 잠금은 모두 0이다.

Root가 수행한 별도 공식 채점은 [공식 제출 receipt](../p3_hmax_official_submission_20260906_v1/receipt.json)에 기록했다. 이 ZIP은 재생 검증을 마친 연구 후보이며, 기존 보존 기준 모델을 대체하는 최종 선택본으로 표시하지 않는다. 공식 결과를 이용한 모델·소스 변경이나 재튜닝은 수행하지 않았다. 위 업로드 0은 이 저장 추론 작업의 범위이며 root의 별도 제출 건과 구분한다.

기존 cold/연구 소스·모델·locks·답안·공통 문서·Git 상태는 이 작업에서 변경하지 않았다.
