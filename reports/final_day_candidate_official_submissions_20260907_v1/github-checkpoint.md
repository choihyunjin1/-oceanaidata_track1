# GitHub 중간 저장 검증 — 2026-09-07

사용자 요청: 현재 진행 내용을 정리해 GitHub에 올리고 남은 기회 활용을 연구할 Fable 프롬프트 작성. 모델 최종 지정이나 추가 업로드 요청으로 해석하지 않았다.

포함: 09-07 사전등록 config, 실행·독립 QA·포장 코드와 synthetic tests, 작은 집계 결과/QA/패키지 해시, 공식 리더보드 영수증, Fable 정정·독립 검토, 최신 README/파일 선택 문서와 다음 검토 프롬프트. 과거 4시간 종료 실패도 성공처럼 덮어쓰지 않고 보존한다. 진행 중 P3 runner/config는 수정하지 않고 현재 원본을 기록한다.

제외: raw/공식 데이터, CSV 답안, 모델·ZIP·예측 배열·parquet/NPZ, 실행 로그/캐시, attempt/finish locks, 이전 날짜에 남아 있던 XML 및 노트북. 로컬 패키지는 기존 Documents 폴더에 보존한다. 기존 사용자 수정은 선택 범위 외에서 손대지 않는다.

검증: 관련 synthetic pytest 7파일 **23 PASS**(약 5.34초). 새 생산 학습 0. Ruff 기본 검사에는 이미 기록된 P2 callback B023 1건이 남으며 무조건 PASS로 표현하지 않는다. `p2_l120_s10_proj_20260907_v1.py:124` callback은 같은 loop iteration의 동기 `fit_model`에서 호출되고 그 뒤 반환하므로 지연 closure가 아니다(core.py:439). 이 파일의 B023만 명시적으로 제외한 Ruff 검사 PASS. sealed 학습 소스/해시를 바꾸기 위해 무의미하게 코드를 수정하지 않았다. 기타 Ruff 진단 0.

선별 파일은 md/json/py만 허용, 파일당 200KB 이하, JSON 파싱·일반 credential/key 패턴 검사·예측/정답 배열 키 검사에서 이상 없음. 이는 비밀이 절대 없다는 증명이 아니라 해당 점검의 통과 기록이다. `.gitignore`의 data/artifacts/submissions/ZIP/log/env 제외를 확인했고 lock은 별도 경로 필터로 제외했다. `git diff --check` PASS. 원격/로컬 시작 HEAD는 c27151fd014980d58a2c90c8af7ca6cfd8bd2c47로 일치했다. 일반 push만 허용하며 원격 충돌이면 중단한다.

추가 staged 검사: sealed JSON의 CRLF 보존 때문에 기본 diff-check는 CR을 trailing whitespace로 보고하고 기존 감사 문서 마지막 빈 줄도 보고했다. 해시 보존을 위해 파일을 재포맷하지 않았다. `core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof`로 실행한 staged diff-check PASS. 09-07 config JSON에도 `.gitattributes -text`를 적용하여 clone 시 bytes를 보존한다. 작업 트리와 staged blob SHA256을 별도 대조한다.

성과 정본은 receipt.json, 다음 연구 요청 정본은 `docs/ocean_v2_codex/FABLE_REMAINING_SLOTS_PROMPT_20260907.md`. P3 no-shrink는 준비/미채점, 전체 fresh cold는 진행 중이다. 이번 체크포인트를 전체 프로젝트 완료로 해석하지 않는다.
