# Historical snapshot — NOT ACTIVE INSTRUCTIONS

Original paths below were relative to docs/ (or explicitly repository-root paths). Current runbook: [active guide](../../OFFICIAL_SUBMISSION_RUNBOOK_20260905.md). Old eligibility and upload commands are not current authorization.

# Ocean AI 공식 제출 실행서 — 2026-09-05 확인본

> **현재 사용 제한(2026-09-05 저녁):** 이 문서의 UI 조작과 역사적 파일 정보는 보존하지만, 아래 P2/P3 후보의 최신 규정 적격성을 승인하는 문서로 사용하지 않는다. 9월 2일 운영진의 공식 점수 역산 금지 공지를 적용하면 P2 bin17 상위 계수와 P3 refined-public alpha는 재적합 대상이다. P1도 전체 학습 의존 연결을 검증해야 한다. 새 후보는 `SCORE_IMPROVEMENT_PLAN_20260905.md`와 최신 QA/별도 사용자 파일 승인에 따른다. 과거 패키지의 `READY` 문자열만으로 업로드하지 않는다.

## 결론부터

제출은 두 종류이며 서로 바꾸어 생각하면 안 된다.

1. **답안 CSV 업로드**: 각 문제 페이지의 `파일 선택` → `제출하고 채점`으로 리더보드 점수를 얻는다.
2. **최종 모델 제출**: 같은 페이지의 `모델 최종 제출하기`에서 제목·설명·재현 파일·저장소 URL을 등록한다. 이 행동은 해당 문제를 최종 확정하고 이후 답안 업로드를 막을 수 있다.

2026-09-05 로그인 상태에서 OCN-01/02/03은 각각 `오늘 남은 제출 3 / 3`을 표시했다. 이 숫자는 매일·계정 상태에 따라 바뀐다. 공개 랜딩페이지에는 예선 종료/결과 제출일이 `2026-09-30`으로 표시되지만, 로그인 문제 화면에서는 정확한 마감 시각을 확인하지 못했다. 따라서 **실제 클릭 직전에 공지·남은 횟수·정확한 마감 시각을 다시 읽은 값이 이 문서보다 우선**한다.

## 0. 공통 사전 점검

```powershell
$repo = (Resolve-Path .).Path
$package = Join-Path $repo 'artifacts\official_final_submission_20260905'

git status --short --branch
git rev-parse HEAD
Get-Content (Join-Path $package 'MASTER_MANIFEST.json')
.venv-p1\Scripts\python.exe -m pytest -q tests\test_official_final_submission_20260905.py
```

필수 조건:

- `MASTER_MANIFEST.json` 상태가 `LOCAL_READY_MODEL_INFERENCE_NOT_UPLOADED`이다.
- P1/P2/P3 receipt가 `READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED`이다.
- 답안 파일의 SHA가 아래 표와 정확히 같다.
- `upload/`의 모든 파일이 50,000,000 bytes 이하다.
- 원본 운영진 데이터, credential, 외부자료 코드/데이터가 최종 ZIP에 없다.
- 실제 업로드할 정확한 파일에 대해 사용자의 현재 승인이 있다.

## 1. 답안 CSV 업로드

### 문제별 페이지와 파일

| 문제 | 페이지 | 업로드할 파일 | SHA-256 | 열/행 | 점수 귀속 |
|---|---|---|---|---|---|
| OCN-01 / P1 | `https://oceanaidata.org/app/problems/5` | `artifacts/official_final_submission_20260905/P1/05_answer/P1_submission.csv` | `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687` | `station,year,layer,time,label` / 169,011 | 같은 SHA의 공식 F1 `0.833548`, `28.909341점` |
| OCN-02 / P2 | `https://oceanaidata.org/app/problems/6` | 아래 P2 선택 규칙 참조 | 아래 참조 | `station,layer,time,temp` / 26,061 | 파일에 따라 다름 |
| OCN-03 / P3 | `https://oceanaidata.org/app/problems/7` | `artifacts/official_final_submission_20260905/P3/05_answer/P3_submission.csv` | `ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa` | `case_id,station,lead_h,hs_pred` / 1,200 | 같은 SHA의 공식 RMSE `0.583892 m`, `24.066168점` |

### P2 선택 규칙 — 반드시 구분

리더보드의 이미 확인된 최고 규정 준수 성능을 다시 올리는 목적이면, 개인 보관소 `20260901_P2_V52_SCORE_PRIORITY_READY_V1`에서 역사적 `P2_submission.csv`를 찾아 환경변수 `P2_HISTORICAL_CHAMPION_CSV`로 지정한다. 업로드 전 SHA가 반드시 아래 값이어야 한다.

```powershell
$downloadRoot = Join-Path $env:USERPROFILE 'Downloads'
$match = Get-ChildItem -LiteralPath $downloadRoot -Filter 'P2_submission.csv' -File -Recurse -ErrorAction SilentlyContinue |
  Where-Object { (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant() -eq '331b1635bb036e773ff73487075e803b1308223e905c28b0d1494ea88b4d94c9' }
if (@($match).Count -ne 1) { throw "P2 historical champion must resolve to exactly one file" }
$env:P2_HISTORICAL_CHAMPION_CSV = $match.FullName
$historical = (Resolve-Path $env:P2_HISTORICAL_CHAMPION_CSV).Path
(Get-FileHash -Algorithm SHA256 -LiteralPath $historical).Hash.ToLowerInvariant()
# 반드시 331b1635bb036e773ff73487075e803b1308223e905c28b0d1494ea88b4d94c9
```

이 파일에만 공식 RMSE `0.424019 C`, `28.012945점`을 귀속할 수 있다.

최종 재현 패키지가 실제로 만든 모델 답안을 검증하려는 목적이면 아래 파일을 쓴다.

```text
artifacts/official_final_submission_20260905/P2/05_answer/P2_submission.csv
SHA-256 64f59fe7b28b3d60189e6fdef8ed00708da2f3c8066509f52395395d5a93f1ce
```

이 파일은 같은 v52 레시피를 새로 3-fit해 얻은 **아직 미채점 replay**다. 역사적 `0.424019`를 이 파일의 점수라고 쓰면 안 된다.

### 브라우저 클릭 순서

1. 위 표의 정확한 문제 URL을 연다.
2. 답안 카드의 `파일 선택`에서 해당 CSV 하나만 선택한다. 허용 형식은 현재 UI상 `.csv`, `.gz`, `text/csv`다.
3. 화면에 선택된 파일명이 맞는지 확인한다.
4. `제출하고 채점`을 한 번 클릭한다. 동일 직전 파일의 중복 제출은 거절될 수 있으므로 연타하지 않는다.
5. 응답에서 접수 ID/시간/상태를 기록하고 `/app/submissions`에서 문제·파일·점수를 확인한다.
6. SHA, 접수 ID, 공식 지표, 점수, KST 시각을 작은 receipt/report에 남긴다. hidden 정답은 열지 않는다.

## 2. 최종 모델·재현 패키지 제출

### 업로드 파일 집합

모든 파일은 `artifacts/official_final_submission_20260905/upload/` 아래에 있다.

| 문제 | 반드시 함께 고를 파일 | 파일 수 |
|---|---|---:|
| P1 | `P1_official_final_core.zip`, `P1_REASSEMBLY_MANIFEST.json`, 모든 `P1_*.part??.zip` | 24 |
| P2 | `P2_official_final_core.zip` | 1 |
| P3 | `P3_official_final_core.zip` | 1 |

P1은 세 가중치와 두 파생 feature 표면이 50 MB 제한을 넘으므로 22개 조각 ZIP으로 분리되어 있다. core ZIP을 풀어 `P1/` 폴더를 만든 뒤 모든 업로드 파일이 있는 폴더에서 다음처럼 복구할 수 있다.

```powershell
python P1\REASSEMBLE_UPLOAD.py `
  --upload-dir '<24개 P1 업로드 파일이 있는 폴더>' `
  --package-dir '<core ZIP에서 풀린 P1 폴더>'
```

스크립트는 ZIP마다 단일 member·part 크기·part SHA를 검사하고, manifest 순서로 결합한 뒤 원본 크기·SHA까지 확인한다. 같은 SHA의 파일이 이미 있으면 안전하게 건너뛴다.

### 문제별 폼 입력값

값의 단일 근거는 각 로컬 패키지의 `P?/06_submission/FORM.json`이다.

| 문제 | 제출물 제목 | 한 줄 요약 |
|---|---|---|
| P1 | `P1 MS-TCN e150 + GI spike2 clean incumbent` | `배포 데이터만으로 scratch 학습한 3-seed MS-TCN e150 앙상블에 GI 고신뢰 spike 2행만 더한 공식 최고 clean 계보입니다.` |
| P2 | `P2 v52 third-moment input-gradient clean incumbent` | `공식 최고 clean 레시피를 배포 데이터만으로 새로 3-seed scratch 학습해 bin17 anchor와 80/20 혼합한 재현 패키지이며, 현재 replay는 별도 SHA입니다.` |
| P3 | `P3 clean refined long-lead optimum` | `배포 데이터만으로 scratch 학습한 두 고정 CatBoost 계보의 장기리드 축을 봉인된 alpha로 결합한 최고 clean 계보입니다.` |

공통 입력:

- `코드 저장소 URL (선택)`: `https://github.com/choihyunjin1/-oceanaidata_track1`
- `결과물 링크 (선택)`: 비워 둔다.
- P1/P3 `비고`: `운영진 배포 데이터만 사용한 scratch 학습 계보. 모델 가중치와 재현 코드를 동봉했습니다.`
- P2 `비고`: `운영진 배포 데이터만 사용한 scratch 3-fit replay입니다. 역사적 공식 최고점은 별도 SHA의 답안에 귀속되며 현재 replay의 공식 점수라고 주장하지 않습니다.`
- `결과물 파일 업로드`: 위 문제별 파일 집합을 선택한다. 현재 UI는 여러 파일 선택을 지원하고 파일당 최대 50 MB다.

### 최종 클릭 순서와 잠금 경고

1. 해당 문제 페이지에서 필요한 답안 CSV 채점을 모두 끝냈는지 확인한다.
2. `모델 최종 제출하기`를 연다.
3. 제목·한 줄 요약·저장소 URL·비고를 `FORM.json`과 대조한다.
4. 위 표의 파일 수를 정확히 선택하고, 각 파일 크기와 `MASTER_MANIFEST.json`의 SHA를 대조한다.
5. 최종 제출 뒤 답안 업로드가 잠긴다는 경고를 다시 읽는다.
6. 이 **정확한 문제·파일 집합·잠금 효과**에 대한 사용자 최종 승인이 있을 때만 `모델 최종 제출`을 한 번 클릭한다.
7. 현재 클라이언트 흐름은 모델 제출 레코드를 먼저 만들고 파일을 순차 업로드한다. 중간 파일 업로드가 실패하면 새 레코드를 즉시 만들지 말고 `/app/submissions`에서 기존 레코드와 첨부 상태를 확인해 복구한다.
8. 완료 후 제출 ID, 문제, 제목, 파일명/bytes/SHA, KST 시각, UI 상태를 receipt에 기록한다.

최종 모델 폼은 현재 `제출물 제목`이 필수이고, 파일·저장소 URL·결과 링크 중 하나 이상이 필요하다. 우리는 재현 파일과 저장소 URL을 모두 넣는다.

## 3. P1 업로드 파일 수·크기 감사

```powershell
$upload = 'artifacts\official_final_submission_20260905\upload'
$p1 = Get-ChildItem -LiteralPath $upload -File |
  Where-Object { $_.Name -eq 'P1_official_final_core.zip' -or
                 $_.Name -eq 'P1_REASSEMBLY_MANIFEST.json' -or
                 $_.Name -like 'P1_*.part??.zip' }
$p1.Count                 # 24
($p1 | Measure-Object Length -Maximum).Maximum  # 50,000,000 미만
$p1 | Sort-Object Name | Get-FileHash -Algorithm SHA256
```

P2/P3 core ZIP도 같은 방식으로 `MASTER_MANIFEST.json`의 `upload_files`와 대조한다. Git에는 이 ZIP이나 답안 CSV를 올리지 않는다.

## 4. 종료 조건

다음 네 가지가 모두 확인돼야 “공식 제출 완료”라고 표현한다.

- 브라우저 응답과 `/app/submissions`에 접수 레코드가 있다.
- 문제 ID와 선택 파일이 맞다.
- 점수/처리 상태가 확인됐거나 처리 중임을 명확히 구분했다.
- 제출 receipt가 로컬 보고서에 기록됐다.

로컬 파일 준비, GitHub push, 브라우저 파일 선택만으로는 공식 제출 완료가 아니다.
