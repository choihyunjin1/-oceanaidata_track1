# Ocean AI Track 1 — AI 인수인계·정찰 가이드

## 결론부터

이 저장소에서 다음 AI가 가장 먼저 해야 할 일은 새 모델을 돌리는 것이 아니라, **규정 경계·현재 Git 상태·후보 SHA·로컬 패키지 상태를 먼저 고정하는 것**이다. 현재 공식 최종 패키지의 재현 코드 기준점은 커밋 `48da22f1bbedbd575060b9a4b82681eaa6ce796c`이며, 이후 커밋은 안내·패키징 안전장치를 더한다. 실제 작업 기준 커밋은 언제나 `git rev-parse HEAD`로 다시 확인한다.

현재 선택은 다음과 같다.

| 문제 | 규정 준수 최고 계보 | 공식 확인 지표 | 답안 SHA-256 | 재현 상태 |
|---|---|---:|---|---|
| P1 | `P1_1_E150_PLUS_GI_SPIKE2` | F1 `0.833548`, `28.909341점` | `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687` | 저장 가중치 추론과 byte-exact |
| P2 | `P2_V52_SCORE_PRIORITY_FULL_HISTORY_BLEND020` | RMSE `0.424019 C`, `28.012945점` | `331b1635bb036e773ff73487075e803b1308223e905c28b0d1494ea88b4d94c9` | 역사적 답안만 exact; 현재 3-fit replay는 별도 SHA |
| P3 | `P3_REFINED_PUBLIC_OPTIMUM_20260827` | RMSE `0.583892 m`, `24.066168점` | `ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa` | 저장 가중치 추론과 byte-exact |

P2의 역사적 점수를 현재 패키지 답안에 붙이면 안 된다. 패키지의 새 모델 replay SHA는 `64f59fe7b28b3d60189e6fdef8ed00708da2f3c8066509f52395395d5a93f1ce`이고 아직 그 공식 점수를 받은 파일이 아니다.

## 처음 10분에 읽을 순서

1. `AGENTS.md`
2. `00_ORGANIZER_DATA_POLICY.md`
3. `00_MUST_READ_FIRST.md`
4. P2면 `01_P2_MUST_READ_FIRST.md`, P3면 `02_P3_MUST_READ_FIRST.md`
5. 사용자가 가진 운영진 배포 폴더 안의 해당 `README.md`
6. 이 문서
7. `docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md`
8. `configs/final_submission_portal_20260905.json`
9. `docs/OFFICIAL_FINAL_SUBMISSION_20260905.md`

읽은 뒤 아래의 읽기 전용 정찰부터 수행한다.

```powershell
git status --short --branch
git rev-parse HEAD
git remote -v
git log -5 --oneline
git check-ignore -v artifacts/official_final_submission_20260905/MASTER_MANIFEST.json
Get-Content artifacts/official_final_submission_20260905/MASTER_MANIFEST.json
```

`artifacts/official_final_submission_20260905/`가 없으면 생성되지 않은 로컬 자산일 뿐 Git 유실이 아니다. 원본 데이터와 모델/답안은 의도적으로 Git에서 제외한다.

## 자산 지도

| 자산 | 위치 | Git | 용도 |
|---|---|---|---|
| 최상위 규정 | `00_ORGANIZER_DATA_POLICY.md` | 추적 | 모든 연구·학습·제출의 최우선 경계 |
| 문제별 운영 지침 | `00_MUST_READ_FIRST.md`, `01_P2_MUST_READ_FIRST.md`, `02_P3_MUST_READ_FIRST.md` | 추적 | 누출·재실행·승격 정책 |
| 최종 후보 계약 | `configs/final_submission_20260905.json` | 추적 | 후보·입력·점수·SHA의 근거 |
| 포털 행동 계약 | `configs/final_submission_portal_20260905.json` | 추적 | URL, 파일, 폼, 클릭 순서 |
| 패키지 빌더 | `scripts/build_official_final_submission_20260905.py` | 추적 | 문제별 원자적 폴더 생성 |
| 패키지 코어 갱신 | `scripts/refresh_official_final_package_20260905.py` | 추적 | 재학습 없이 source allowlist·core ZIP 갱신 |
| P1 조각 복구 | `scripts/reassemble_p1_upload_20260905.py` | 추적 | 50 MB 분할 자산의 해시 검증·재조립 |
| 학습·추론 notebook | `notebooks/final_submission_20260905/P?/` | 추적 | TRAIN/PREDICT 분리 실행 |
| 로컬 완성 패키지 | `artifacts/official_final_submission_20260905/` | 무시 | 실제 모델·답안·업로드 ZIP |
| 과거 실험 보고서 | `reports/` | 작은 문서만 추적 | 실패·승격·공식 점수 근거 |
| 원본 데이터 | `P?_DATA_DIR`이 가리키는 운영진 폴더 | 절대 금지 | 읽기 전용 학습/추론 입력 |

## 절대 경계

- 운영진 배포 데이터 밖의 관측·재분석·예보 자료를 학습, 선택, 보정, 앙상블, 내부 채점에 쓰지 않는다.
- 비배포 KIOST/KORS/KHOA 원자료는 정답과 같은 것으로 취급한다. 학습에 쓰지 않더라도 내부 테스트지로 열거나 비교하면 안 된다.
- 실제 관측자료로 사전학습된 가중치를 쓰지 않는다. 합성-only 사전학습 예외는 운영진 네 조건을 모두 입증할 때만 가능하다.
- 과거 KMA/ERA5 계보는 감사 증거로 Git에만 보존한다. 최종 재현 ZIP, 답안 계보, 후속 앙상블에 넣지 않는다.
- `test`, `sample_submission`, hidden truth의 값을 문서·로그·커밋에 노출하지 않는다. 스키마, 행 수, 집계, SHA만 기록한다.
- `git add .`, force push, rebase, reset, 원본 폴더 이동·복사를 하지 않는다.
- 리더보드 점수와 로컬 CV/추정 점수를 같은 표기 없이 섞지 않는다. 공식 점수는 정확히 같은 SHA의 답안에만 귀속한다.

## 연구를 재개할 때의 판정 순서

1. 새 아이디어가 규정 준수 계보인지 확인한다.
2. 데이터·feature·split·seed·후처리·승격 gate를 실행 전에 고정한다.
3. 행 무작위 분할 대신 문제별 시간/그룹 구조와 purge를 유지한다.
4. 기존 최고 후보와 **같은 내부 평가면**에서 비교한다.
5. 평균 개선뿐 아니라 worst block, 계절/정점/층, seed 분산, failure mode를 함께 남긴다.
6. 후보가 실패해도 결과를 지우거나 같은 실험을 결과 기반으로 다시 조정하지 않는다. 다음 가설을 새 실험 ID로 만든다.
7. 외부자료 계보보다 점수가 높아도 규정 미준수면 후보 자격이 없다.
8. 제출 직전에는 파일 SHA와 공식 UI의 남은 횟수·마감·최종 모델 잠금 경고를 다시 확인한다.

## 최종 패키지 재검증

```powershell
.venv-p1\Scripts\python.exe scripts\refresh_official_final_package_20260905.py
.venv-p1\Scripts\python.exe -m pytest -q tests\test_official_final_submission_20260905.py
.venv-p1\Scripts\python.exe -m ruff check `
  scripts\build_official_final_submission_20260905.py `
  scripts\refresh_official_final_package_20260905.py `
  scripts\reassemble_p1_upload_20260905.py `
  tests\test_official_final_submission_20260905.py
```

이 refresh는 학습·추론·답안 값을 다시 만들지 않는다. P1/P2의 실행 import closure만 최종 ZIP에 남기고, 이미 존재하는 세 core ZIP의 해시와 `MASTER_MANIFEST.json`을 갱신한다. 최종 상태는 `LOCAL_READY_MODEL_INFERENCE_NOT_UPLOADED`여야 한다.

## Git 인수인계 규칙

커밋 전에는 `git status --short`, `git diff`, `git diff --cached`, `git diff --check`를 모두 확인한다. stage 대상은 코드·설정·테스트·작은 집계·문서뿐이다. 다음은 절대 stage하지 않는다.

- `artifacts/`, 원본 ZIP/CSV, 답안 CSV
- 모델/checkpoint, parquet/npz, cache/log/lock
- credential, token, `.env`, 개인 절대 경로

실제 GitHub 백업과 실제 대회 제출은 별개의 행동이다. Git push가 성공해도 답안 업로드나 최종 모델 제출이 일어난 것이 아니다.
