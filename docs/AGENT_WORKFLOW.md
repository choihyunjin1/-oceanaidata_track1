# 짧은 연구·개발·검증 루프

## 완료 단위

문제 계약 → 가설/기존 근거 → 작은 계약 검사 → 승인된 학습 → 내부 테스트 → 독립 QA → 재현 → 후보 판단.
새 방법 자체를 성과로 세지 않습니다. 실패 시 사전등록된 다음 분기로 이동하며 기술 실패·성능 악화·증거 부족을 구분합니다.

## 반복 낭비 방지

- 규정과 담당 문제만 작업 시작에 읽고, 변하지 않았다면 반복하지 않습니다. archive/는 사건 조사 때만 읽습니다.
- 모델명보다 데이터·target·split·feature·후처리 지문과 기존 결과를 먼저 대조합니다.
- parameter/schema/path synthetic 검사를 학습보다 먼저 합니다. GPU는 한 담당자만 사용합니다.
- 환경 설치와 CUDA smoke는 환경 변경/실제 장애 때만 합니다. 감시는 metadata/progress/error만 봅니다.
- 변경과 관련된 pytest/Ruff 한 번. 새 코드·오류·미해결 위험이 있을 때만 확대/반복합니다.
- 실험별 result/보고서가 단일 근거입니다. 통합 문서는 링크와 판단, 공식 점수는 정확한 CSV SHA에 연결합니다.
- progress JSON에는 중간 성능까지 들어갈 수 있으므로 stage/완료 fit 수/경과시간만 선택해 출력합니다. 스레드 조회도 전체 tool trace를 다시 출력하지 않고 관련 완료 요약/오류만 추립니다.
- 긴 출력 잘림이나 patch 문맥 오류가 나면 해당 구간만 다시 읽습니다. 이미 포맷된 파일에 이전 문맥을 추측해 큰 patch를 반복하지 않습니다.

## 집중 코드 검사

먼저 해당 테스트가 합성/로컬 검사이며 학습·공식 입력·네트워크를 건드리지 않는지 읽습니다.
도구는 pytest의 보안 격리가 아니므로 임의 테스트의 부작용을 막는다고 주장하지 않습니다.

```powershell
.venv-p1\Scripts\python.exe scripts\agent_verify.py `
  --test tests/test_agent_verify.py `
  --lint scripts/agent_verify.py --lint tests/test_agent_verify.py `
  --execute --reuse-pass
```

--execute 없이는 선택 경로를 검사하고 계획만 표시합니다. GPU를 숨기고 OMP/MKL을 1 thread로 제한합니다.
PASS만 ignored artifacts/agent_verification/에 저장하며 --reuse-pass에서만 재사용합니다.
선택 파일, 전체 src/scripts/configs/tests 코드·JSON/YAML/TOML, requirements, pytest.ini/setup.cfg/tox.ini/.ruff.toml, 규정/AGENTS,
Python·패키지 버전·주요 pytest 환경·명령을 hash로 대조합니다. 검사 중 변경되면 PASS_SOURCE_CHANGED로
캐시하지 않고 CLI가 비정상 코드를 반환합니다. 실패·손상 캐시는 재사용하지 않습니다.
그 밖의 파일/환경/외부 상태에 의존하는 테스트는 캐시 없이 직접 실행합니다.
PYTEST_ADDOPTS와 pytest addopts를 비워 수집 전용 옵션을 제거하고, JUnit 기록에서 실제 실행된 테스트가 1개 이상일 때만 PASS로 인정합니다.
원본/OOF/모델/답안 QA는 이 캐시로 생략할 수 없습니다.

## 정리 경계와 근거

원문은 docs/archive/instructions_20260905/에 보존했습니다. 오래된 clean/READY는 현재 권한이 아닙니다.
frozen runner/config/receipt는 재현 의존성이 있어 일괄 삭제하지 않습니다. 실제 미사용을 입증한 코드만 제거합니다.
전역 설정·설치된 타사 스킬은 변경하지 않았습니다. 프로젝트 스킬은 .agents/skills/에만 둡니다.
이 프로젝트를 Codex에서 열면 프로젝트 지침 탐색이 적용됩니다. 다른 cwd에서는 해당 지침을 명시적으로 읽습니다.

[Astra 안내](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)의
지침 점검·승인된 작업 완료·비례적인 검증 권고를 적용했습니다.
[AGENTS](https://learn.chatgpt.com/docs/agent-configuration/agents-md)와
[스킬 안내](https://learn.chatgpt.com/docs/build-skills)에 따라 짧은 지침과 필요할 때 읽는 기록을 분리했습니다.
