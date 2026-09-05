# Codex 마스터 브리프 — Ocean AI Data Track 1 `ocean_v2` 재구축 (2026-09-05)

이 문서는 코드 작성 담당(Codex)이 처음 읽는 문서다. 이 브리프 → `COMMON_SPEC.md` → 담당 문제의 `P?_SPEC.md` → `PACKAGING_SPEC.md` 순으로 읽고,
`CODEX_TASK_PROMPTS.md`의 태스크를 순서대로 수행한다. 배경 근거는 `reports/claude_recon_20260905/`(정찰 보고 4건 + 설계안 6건 + 요약)에 있다.

## 0. 왜 이 작업을 하는가 (한 문단)

대회(oceanaidata.org, 대학부) 세 문제의 현재 최고 제출물은 (P1) 학습 코드가 없는 동결 예측 CSV 94.8% + 수동 2행 패치, (P2) 리더보드 점수로 적합한 동결 anchor CSV 80%,
(P3) 리더보드 점수로 적합한 외삽 상수 α=−10.217 로 만들어져 있다. 운영진 09-02 공지는 "리더보드 점수를 역산해 파라미터를 정하는 것 금지, 재현 검증 1항(과도한 상수 리터럴)·4항(학습 산출물 제거 후 예측 재생성)으로 확인,
검증 통과 팀만 본선 진출, 최종 순위는 Private"이라고 명시했다. 따라서 **09-07(모델 제출 마감)까지** 세 문제 모두 "배포 데이터 → 학습 → 가중치 → 예측"이 코드만으로 완전히 재생성되는 새 파이프라인을 만들고,
정직한 로컬 CV로 후보를 고르고, 최종 재현 패키지를 만든다. 답안 업로드와 최종 모델 제출은 사용자가 직접 한다(코드가 네트워크 업로드를 해서는 안 된다).

## 1. 절대 규칙 (위반 시 실격 위험)

1. **입력은 배포 데이터만**: P1 `train.csv/test.csv/sample_submission.csv/baseline_rule.csv`, P2 `observations.csv/test_index.csv/sample_submission.csv/baseline_interp.csv`, P3 `train_wave.csv/train_atmos.csv/test_context.parquet/test_index.csv/sample_submission.csv/baseline_persistence.csv`. 외부 관측·재분석·예보·사전학습 가중치·인터넷 접근 금지.
2. **리더보드 점수는 어떤 파라미터에도 쓰지 않는다.** 보정계수·혼합비·임계값·분위 상수·라우팅 규칙은 전부 `train` 실행이 CV OOF에서 계산해 `fitted_params.json`(P2는 `derived_constants.json`)에 기록하고 `predict`는 그 파일을 읽기만 한다. 코드 안 숫자 리터럴은 설정 JSON의 구조·물리 상수(README에 적힌 값: 1.5 m, 78 h, 48 h, 리드 시간, 층 번호, 클립 범위 등)만 허용한다. 기존 후보의 상수(α, bin17, 0.2 blend, 0.834 등)를 옮겨 적는 것도 금지.
3. **기존 동결 산출물 재사용 금지**: `router_anchor.csv`, `gi_spike2_patch.json`, `bin17_anchor.csv`, 과거 제출 CSV, `artifacts/**` 안의 예측 파일을 예측 입력으로 쓰지 않는다. 기존 학습 가중치(.pt/.cbm/.joblib)도 새 패키지에 넣지 않는다.
4. **결정론**: CPU 전용 학습. LightGBM `deterministic=True, force_row_wise=True, num_threads=<config>`; XGBoost `tree_method="hist", device="cpu", nthread=<config>`; CatBoost `task_type="CPU", thread_count=<config>, random_seed=<config>`; numpy `default_rng(seed)`; 정렬은 `kind="mergesort"`. 같은 머신·같은 라이브러리 버전에서 `train`+`predict`를 두 번 실행하면 CSV SHA-256이 동일해야 한다. GPU 사용 금지(테스트 포함).
5. **런타임 예산**: 문제당 전체 재생성(특징 → CV → 최종 학습 → 예측) ≤ 2시간, 세 문제 합계 ≤ 6시간. 실제 소요시간을 receipt에 기록한다.
6. **누출 금지**: P2 hidden 구간(2025-09-01~10-31 KST, layer 2/3/4)의 temp/psal은 NaN이어야 하며 특징에 목표층 값이 들어가면 안 된다(assert). P3는 사례별 289행 문맥 밖의 행·절대 시각·미래 행을 쓰지 않는다. P1은 라벨을 특징 계산에 쓰지 않는다. fold 검증 행은 학습·통계·임계값 선택에 쓰지 않는다.
7. **데이터 폴더는 읽기 전용**. 원자료를 복사·수정·이동·커밋하지 않는다. 문서·로그·리포트에 원시 관측 행을 인용하지 않는다(집계값·해시만).
8. **Git**: 커밋·푸시하지 않는다(사용자가 한다). `artifacts/`, `submissions/`, 캐시, 가중치는 이미 `.gitignore` 대상이다.
9. **파일 경계**: 새로 만드는 경로만 쓴다 — `src/ocean_v2/**`, `configs/ocean_v2/**`, `scripts/ocean_v2/**`, `tests/ocean_v2/**`, `docs/ocean_v2_codex/**`, `artifacts/ocean_v2/**`, `submissions/claude_v2/**`, `artifacts/official_final_submission_v2_20260907/**`. 기존 `src/p1_qc`, `src/p2_restore`, `src/p3_wave`, `scripts/final_submission_20260905`, `artifacts/official_final_submission_20260905`는 **수정하지 않는다**(필요 함수는 새 패키지로 복사해 단순화). `pyproject.toml`에 `ocean_v2` 패키지 추가만 허용.

## 2. 환경

- 저장소: `C:\Users\cedis\PycharmProjects\PythonProject` (Windows 11, PowerShell). Python: `.venv-p1\Scripts\python.exe` (3.12.10; numpy 2.3, pandas 3.0, scipy 1.18, scikit-learn 1.9, lightgbm 4.7, xgboost 3.4, catboost 1.2.10, pyarrow 25, polars 1.43, torch 2.13 cu130 — torch는 CPU 모드로만).
- 하드웨어: Ryzen 7 7800X3D 8코어, 64 GB RAM, 디스크 700 GB 여유. 스레드 기본값 8(설정으로 조정).
- 데이터 경로는 환경변수 또는 CLI 인자로만 주입: `P1_DATA_DIR=C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly`, `P2_DATA_DIR=C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore`, `P3_DATA_DIR=C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast`. 코드에 개인 절대경로·한글 경로를 하드코딩하지 않는다.
- 실행 예: `$env:P3_DATA_DIR="..."; .venv-p1\Scripts\python.exe -m ocean_v2.p3 all --config configs\ocean_v2\p3_base.json --out artifacts\ocean_v2\p3\base`.
- 긴 작업(CV/학습)은 반드시 로그 파일로 실행(`... > artifacts\ocean_v2\p3\base\train.log 2>&1`)하고 진행률을 주기적으로 출력한다. 중간 산출물(특징 캐시 parquet)은 입력 SHA·특징 버전 키로 저장해 재실행 시 재사용한다.

## 3. 공통 산출물 계약

각 문제 실행 결과 `artifacts/ocean_v2/<p>/<candidate>/`:
- `features_cache/*.parquet`(+ `cache_manifest.json`: 입력 SHA, 특징 버전, 행/열 수)
- `cv/oof.parquet`(키 + 예측 + fold), `cv/cv_report.json`, `cv/cv_report.md`(사람이 읽는 표: 블록/정점/층/리드/유형별, persistence·baseline 비교, bootstrap CI)
- `fitted_params.json`(또는 `derived_constants.json`): 값 + 유도 방법 문자열 + 근거 OOF 표 요약
- `models/`(LightGBM `.txt`, XGBoost `.json`/`.ubj`, CatBoost `.cbm`, torch `.pt`) + `MODEL_MANIFEST.json`(파일별 SHA·bytes, seed, 하이퍼, 학습 행 수, 소요시간)
- `TRAINING_RECEIPT.json`(시작/종료 시각 KST, 총 소요시간, 스레드, 라이브러리 버전, 입력 SHA 6개)
- 예측: `submissions/claude_v2/<p>/<candidate>/{P?_submission.csv, sha256.txt, receipt.json, validator.json, cv_summary.md}`

CSV 규격(배포 README·score.py와 동일): P1 `station,year,layer,time,label` 169,011행 label∈{0,1}; P2 `station,layer,time,temp` 26,061행 −5~45; P3 `case_id,station,lead_h,hs_pred` 1,200행 0~30. 키·행 순서는 `sample_submission.csv`와 완전히 같아야 하고 index 열 없음, UTF-8, `lineterminator="\n"`.

## 4. 코딩 표준

- Python 3.12, 타입 힌트, `from __future__ import annotations`, dataclass 설정, `logging` 사용. 파일당 500줄 이내 권장, 함수는 단위 테스트 가능하게.
- 설정은 `configs/ocean_v2/<p>_<candidate>.json` 하나로 실행 전체를 결정(특징군 on/off, seed, 스레드, 블록 정의, 하이퍼, 사다리 옵션). 코드는 설정을 읽고 그대로 실행한다.
- 모든 CLI는 `python -m ocean_v2.<p> {audit|features|cv|train|predict|all}`. `train`은 CV를 포함해 fitted_params를 생성하고 전체 학습까지; `predict`는 models+fitted_params만으로 CSV 생성.
- 테스트는 `pytest tests/ocean_v2 -q`. 데이터가 필요한 테스트는 `P?_DATA_DIR`가 없으면 skip.
- 상수 감사: `python -m ocean_v2.common.audit_constants --package ocean_v2.p3` 가 predict 경로 모듈의 숫자 리터럴을 나열하고 allowlist(설정 JSON의 물리 상수) 외 리터럴이 있으면 실패해야 한다.

## 5. 작업 순서와 시간 예산 (09-05 저녁 → 09-07 오전)

| 순서 | 태스크 | 예산 |
|---|---|---|
| T0 | `src/ocean_v2/common` + 테스트 | 1h |
| T1 | P3 골격·특징·CV·B1(안전 기준선) → 후보 CSV | 3h (실행 ≤1h) |
| T2 | P2 골격·특징·CV·R0(안전 기준선) → 후보 CSV | 3h (실행 ≤1h) |
| T3 | P1 골격·특징 캐시·CV·C0(안전 기준선) → 후보 CSV | 4h (실행 ≤1.5h) |
| T4~T6 | 문제별 사다리(사전등록 게이트) | 09-06 하루 |
| T7 | 최종 패키지 빌더 v2 + 클린룸 재현 | 09-06 저녁~09-07 오전 |
| T8 | 사용자 전달물(업로드 파일 집합·폼 값·SHA·삭제 목록) | 마지막 |

세 문제를 동시에 실행하면 8코어를 나눠 쓰므로 벽시계가 2~3배 늘어난다. 학습은 P3 → P2 → P1 순으로 순차 실행하고, 코드 작성은 병렬로 진행한다.

## 6. 각 태스크 완료 보고 형식 (사용자에게 붙여넣을 내용)

1. 만든/수정한 파일 목록. 2. 실행한 명령과 소요시간. 3. CV 리포트 핵심 표(블록별·전체, baseline/persistence 대비). 4. 생성된 후보 CSV 경로·행 수·SHA-256·validator 결과. 5. fitted_params 요약(값과 유도 방법). 6. 결정론 확인(2회 실행 SHA 동일 여부). 7. 미해결 문제·리스크. 리더보드 점수 예측은 "sanity 범위"로만 적는다.

## 7. 실패·막힘 시

- 런타임이 예산을 넘으면 seed 수·트리 수·멤버 수를 줄인 설정을 별도 후보로 만든다(원 설정도 보존). 
- CV가 baseline(persistence/선형보간/기존 계보 재평가)보다 나쁘면 파이프라인 버그(정렬·키·클립·특징 정합)를 먼저 의심하고 등가성 테스트를 돌린다.
- 규정 해석이 모호하면 보수적으로(사용하지 않음) 결정하고 README에 기록한다.
