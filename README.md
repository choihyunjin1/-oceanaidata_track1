# Ocean AI Data Track 1 — P1 / P2 / P3

세 문제의 연구·학습·내부 검증·재현 자료를 관리하는 통합 저장소입니다.
P1은 수온 이상 탐지, P2는 중간층 수온 복원, P3는 유의파고 예측입니다.
패키지 이름 `p1-qc`는 초기 구현명이며 저장소 범위를 P1로 제한하지 않습니다.

## 지금 시작할 곳

1. [에이전트 작업 규칙](AGENTS.md)과 [운영진 규정](00_ORGANIZER_DATA_POLICY.md)
2. 해당 문제 계약: [P1](00_MUST_READ_FIRST.md) · [P2](01_P2_MUST_READ_FIRST.md) · [P3](02_P3_MUST_READ_FIRST.md)
3. **[9월 7일 최종 패키지·파일 선택 안내](docs/FINAL_RELEASE_20260907.md)**: 현재 선택본·학습/추론·첨부 방법
4. [짧은 개발·검증 루프](docs/AGENT_WORKFLOW.md)
5. 제출·재현 작업이면 [인수인계](AI_HANDOFF.md)와 [제출 실행서](docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md)

## 현재 채점 우선 후보 — 2026-09-07 15:12 KST (최종 제출 보류)

| 문제 | 현재 선택 | 공식 지표 | 공식 점수 | 답안 SHA 앞자리 |
|---|---|---:|---:|---|
| P1 | 원형 O+B+MS-TCN, 배포자료부터 7fit 복원 | F1 0.833548 | **28.909341** | `57844ef2` |
| P2 | C3 DeepSet L120, 3-seed + smooth7 + endpoint projection | RMSE 0.395254℃ | **28.373869** | `794268f1` |
| P3 | CatBoost CPU numeric 3-seed + router, no-shrink | RMSE 0.595521m | **23.881592** | `70761aff` |

**[최신 상태·전체 증거·로컬 패키지 정본](docs/ocean_v2_codex/FINAL_SELECTION_REVIEW_HANDOFF_20260907.md)**과 [Fable 검토 프롬프트](docs/ocean_v2_codex/FABLE_FINAL_SELECTION_AUDIT_PROMPT_20260907.md)를 먼저 읽는다. 남은 5개 답안을 모두 채점하여 잔여는 **0/0/0**이다. P1 B5(28.858163), O_slow(28.711047)는 원형보다 낮았다. P2는 새 cold/saved 노트북 exact PASS, P3는 saved replay PASS이나 fresh cold는 아직 진행 중이다.
사용자 요청으로 **최종 제출 보류**: P2 첫 최종 버튼 뒤 확인창 OK는 누르지 않았고 접수 receipt도 없다. 이후 브라우저 읽기 연결 실패로 현재 서버 상태를 새로 확인하지 못했다. P1/P3 최종 버튼은 누르지 않았다. 검토 후 명시적 재개 지시 없이 제출하지 않는다.

기존 로컬 묶음: `C:/Users/cedis/Documents/OceanFinalRelease_20260907` (P1 원형은 유지, P2/P3는 이전 선택). 새 P2 최종 검토 ZIP은 `C:/Users/cedis/Documents/OceanFinalSelected_20260907/P2`에, P3 no-shrink는 `C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_numeric_cpudet_noshrink_v2`에 별도 보존한다.
학습 코드·동봉 모델·채점 CSV·노트북·첨부 안내를 문제별로 구분한다.
Git에는 코드·설정·집계 QA·문서만 올리므로 데이터/가중치/답안 ZIP은 로컬에서 선택한다.
[최종 검증 보고서](reports/final_release_20260907_v1/report-source.md)를 통해 실제 완료 범위를 확인한다.
P1 전체 재학습 답안은 역사적 최고 답안과 SHA가 같고, 09-07 서버에서도 중복 내용으로 확인됐다. 새 점수를 받은 것은 아니다.
이전 GPU P3 전체 재학습(17fit, 약 1,498초)은 당시 채점본 `ff42a6a0`와 660/1,200행이 달랐습니다. 이 증거를 현재 CPU `70761aff`의 cold 증거로 쓰지 않습니다. 현재 CPU fresh cold 및 운영진 재학습 허용오차는 별도 미완료·미확인입니다.

## 이전 기준선·분리 실험 이력 — 현재 선택 아님

| 문제 | 코드로 재생성한 기준 후보 | 공식 지표 | 공식 점수 |
|---|---|---:|---:|
| P1 | clean baseline, SHA `5971e145…128a` | F1 0.777749 | 27.426319 |
| P1 새 후보 | bracket, SHA `9031c84e…d93a` | F1 0.785944 | 27.644124 |
| P2 | full scratch clean rebuild | RMSE 0.455143℃ | 27.622418 |
| P2 CPU 비교 | C3 control, SHA `ae2df148…a479` | RMSE 0.489080℃ | 27.196585 |
| P2 CPU copula | in-sample, SHA `e0b4005b…498a` | RMSE 0.475174℃ | 27.371078 |
| P3 | clean baseline rebuild | RMSE 0.607183m | 23.696500 |
| P3 hmax 제거 | whole-cold, SHA `d4560592…08c5` | RMSE 0.608184m | 23.680619 |

P1은 [9월 6일 재생성 기준 답안 영수증](reports/p1_regenerated_baseline_official_submission_20260906_v1/receipt.json),
P2·P3는 [9월 5일 공식 영수증](reports/official_score_repair_submissions_20260905_v1/receipt.json)에 귀속합니다.
P1의 이전 F1 0.790733은 다른 답안의 기록이므로 현재 파일에 승계하지 않습니다.
이 표는 과거 모든 모델 중 최고라는 뜻이 아닙니다.
P1 F1 0.833548 원형은 이후 전체 재학습·답안 동일성을 복원했습니다. 원형 셀 선택 프로그램은 미복구이며 고정 로컬 OOF 설정을 사용합니다.
P2 0.424019℃와 P3 0.583892m의 과거 Public-계수 계보는 최신 규정상 재적합 대상이며 현재 최종본으로 자동 선택하지 않습니다.

P1 bracket 후보는 09-06 06:39 공식 채점과 빈 모델 폴더부터의 portable 재학습·답안 재현을 완료한 **이전 fallback**입니다.
[채점 영수증](reports/p1_bracket_official_submission_20260906_v1/receipt.json),
[새 portable 검증](reports/p1_bracket_portable_20260906_v2/report-source.md).
SHA `9031c84e…d93a`를 이전 기준 답안과 혼동하지 마십시오. 범위/셀 정책 추가는 별도 내부 비교에서 비승격이며 이 파일에 섞지 않았습니다.
P2의 07:28 CPU 대조 제출은 copula의 같은 CPU 기준 대비 개선을 확인했지만 기존 CUDA C3 기준은 넘지 못했습니다. 따라서 기존 P2 기준본을 보존합니다. [공식 대조 영수증](reports/p2_copula_official_submission_20260906_v1/receipt.json).
P3도 배포 원자료부터 12+5회 학습·독립 QA·ZIP 추출 재생을 마친 뒤 08:08 채점했지만 기존 기준보다 +0.001001m 악화했습니다. 기존 P3 기준을 보존합니다. [공식 영수증](reports/p3_hmax_official_submission_20260906_v1/receipt.json). [이번 순서 전체 완료 보고서](reports/remaining_work_completion_20260906_v1/report-source.md)에서 실행 범위와 남은 공백을 확인하십시오.
파일 선택·실행 제한은 [P1 후보 안내](docs/ocean_v2_codex/P1_BRACKET_CANDIDATE_HANDOFF_20260906.md),
세 문제의 기준 ZIP·학습/추론 절차는 [portable 패키지 안내](docs/ocean_v2_codex/PORTABLE_PACKAGE_HANDOFF_20260906.md)를 따릅니다.

## 프로젝트 구조

새로 전달된 [정찰·독립 설계안](reports/claude_recon_20260905/00_SUMMARY.md)과
[ocean_v2 구현 사양](docs/ocean_v2_codex/00_MASTER_BRIEF.md)은 제안 원문입니다.
일부는 검토·수정 후 실험과 패키지로 구현됐습니다. 원문 전체가 실행되거나 승인된 것은 아니며,
[체크포인트](docs/RESEARCH_CHECKPOINT_20260906.md)의 결과별 계약·영수증을 우선하고
[초기 검토 메모](docs/ocean_v2_codex/REVIEW_NOTES_20260905.md)는 변경 이력으로 읽으십시오.
문서의 명령이나 `approved` 표시는 현재 구현·제출·삭제 권한이 아닙니다.

| 경로 | 역할 |
|---|---|
| src/p1_qc, src/p2_restore, src/p3_wave | 문제별 공통 코드 |
| scripts/, configs/, tests/ | 실행기·고정 계약·검증 |
| reports/ | 실측 결과·실패·QA·출처 원장 |
| notebooks/ | 설명 가능한 학습/추론 노트북 |
| docs/ | 현재 계획·작업·제출 안내 |
| artifacts/, submissions/ | Git에서 제외한 모델·예측·로컬 자산 |

`P1_DATA_DIR`, `P2_DATA_DIR`, `P3_DATA_DIR`로 운영진 배포 폴더를 지정합니다.
원본 데이터, 모델, 답안, 비밀정보는 Git에 올리지 않습니다.
기존 환경은 `.venv-p1/Scripts/python.exe`입니다. 환경을 새로 구축할 때만
`scripts/bootstrap_env.ps1`를 사용하며, 매 세션마다 재설치하거나 CUDA 검사를 반복하지 않습니다.

## 제출과 재현

내부 성능, 저장 모델 replay, 공식 채점, 최종 ZIP 적격성은 별도의 검증입니다.
`artifacts/official_final_submission_20260905/`의 옛 READY 표시는 최신 규정 통과를 뜻하지 않습니다.
최종본은 데이터 참조 → 학습 코드 → 학습된 모델 → 답안 → 재현 안내를 분리하고,
배포 자료만으로 네트워크 없이 6시간 안에 재현해야 합니다.
위 6시간을 일반 모델에 적용하는 것은 내부 목표이며 공식 원문 인용이 아닙니다. 공식 공지에서는 합성 사전학습 예외 조건 3에 명시되어 있고, 일반 모델 적용 범위는 문제지 Ⅳ-2 미열람으로 미확인입니다(2026-09-07 정정).
실제 제출 전에는 로그인된 공지의 마감·잔여 횟수·최종 모델 잠금 효과를 확인합니다.
GitHub push는 대회 제출이 아닙니다.

[기존 P1 상세·환경·연구 문서 지도](docs/archive/instructions_20260905/README.md)는 역사적 참고 자료로 보존했습니다.
[GitHub](https://github.com/choihyunjin1/-oceanaidata_track1) · [대회](https://oceanaidata.org)
