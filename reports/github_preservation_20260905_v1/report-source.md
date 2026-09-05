# 2026-09-05 연구 보존 및 제공 사양 검토

## 결론

현재 연구 코드·설정·검증·집계 결과와 사용자 제공 연구 원문을 함께 보존한다. 제공된 정찰/설계/사양 **19파일은 존재하며 문제별 독립 설계6건을 포함한다.** 다만 현행 규정과 최신 결과에 맞추어 수정해야 하므로 T0~T8을 그대로 실행하지 않는다. [검토 메모](../../docs/ocean_v2_codex/REVIEW_NOTES_20260905.md)가 실행 전 입구다.

문서는 Claude로 표기되어 있으며 파일만으로 Fable 모델의 작성 여부까지 입증하지 못했다. 원문은 수정하지 않고 역사적 제안과 실제 수행 결과를 분리했다. 특히 공식 점수 역산을 제안 근거로 재사용하지 않는다.

## 보존 범위와 검증

- 브랜치: `codex/p1-qc`, 출발 commit: `535f94a1791f2398f82ad27659b58701513ab327`.
- 사전 점검 snapshot: 237파일 / 2,503,134 bytes. 이 보존 보고서·목록·QA3파일을 추가하므로 명시적 포함 대상은240파일이다.
- [포함 목록 및 로컬 SHA-256](inclusion-manifest.json), [독립 QA](independent-qa.json).
- 새 Python50파일 Ruff PASS. 새 focused tests 통합 실행 **174 PASS / 12.98초**. 학습·공식 제출을 재실행한 검증은 아니다.
- notebook2건의 출력은 집계/해시/QA이며 원시 관측·정답행·예측행은 포함하지 않는다.
- 새 파일 중1 MiB 초과·금지 데이터 경로·고신뢰 비밀 패턴·JSON 파싱 오류0건.
- 새 sealed JSON 및 notebook의 byte hash를 유지하기 위해 범위를 좁힌 `.gitattributes`를 추가했다. 일반 Markdown 등은 기존 Git 줄바꿈 정규화를 따른다. manifest의 SHA는 로컬 파일 byte 기준이고 저장소 내용의 최종 식별자는 본 보고서를 포함하는 Git commit이다.
- 원시 데이터·공식 입력/답안 CSV·외부자료·모델/체크포인트·예측 배열·credential·cache·log·attempt lock은 새로 포함하지 않는다. 기존 이력의 추적 lock3건은 이번 변경과 무관하며 삭제하거나 재작성하지 않는다.
- 선택된240파일과 staged 목록 일치, JSON/notebook90건은 로컬 byte와 Git blob 일치, 새 보고서의 로컬 링크12건 존재 확인. `cr-at-eol` 기준 공백 검사에는 원문/이전 Markdown의 사소한 경고10건이 남는다. 원문 보존을 위해 수정하지 않았으며 코드 오류는 아니다.

## 실제 성과 / 반대 증거

| 문제 | 현재 확인한 결과 | 다음 판단 |
|---|---|---|
| P1 | 당일 clean 공식 F1 0.790733 / 27.771400점. 후속 고정 정보가치 제출은0.767370 / 27.150461점으로 하락 | 후속 후보 승격 안 함. 수심 하나만 바꾼 인과실험은 아니므로 모든 수심 접근을 반증했다고 말하지 않음 |
| P2 | 당일 clean 공식 RMSE 0.455143℃ / 27.622418점. 조건부 결측 전문가3-seed 후속은9 fits 후 가을 intact +0.000783517℃, outage +0.030801703℃ 악화 | 첫seed 개선을 일반화하지 않음. 조건부 후보의 공식 CSV/업로드 없음 |
| P3 | 당일 clean 공식 RMSE 0.607183m / 23.696500점. hybrid 0.608143m / 23.681268점은 더 나쁨 | clean 비교군 유지. 패널의 미제출 예상 점수나 과거 부적격 계보를 새 근거로 사용하지 않음 |

v2 후속 사이클의 새 실행량은46 base fits와8 meta fits이며 새 primary 승격은 없다. P1 v2는16 fits, P2 objective/tree는9+3 fits, P3 episode는18 fits다. 별도 P2 3-seed 후속은6 historical+3 full fits /365.671초다. 이 보존 작업에서는 새 fit0, 새 CSV0, upload0이다.

근거는 [당일 공식 영수증](../official_score_repair_submissions_20260905_v1/receipt.json), [v2 결과](../parallel_score_improvement_20260905_v2/report-source.md), [v3 추가 검증](../conditional_validation_and_information_submission_20260905_v3/report-source.md)에 연결한다. 과거 최고 기록 전체를 적격 현재 최고로 재확정한 보고서는 아니다.

## 재사용 안내

새 AI는 README → AGENTS → 운영진 데이터 규정 → AI_HANDOFF의 현재 결과를 먼저 확인한다. 이후 [사용자 제공 정찰 요약](../claude_recon_20260905/00_SUMMARY.md)과 [마스터 브리프](../../docs/ocean_v2_codex/00_MASTER_BRIEF.md)를 검토 메모와 함께 읽는다. 외부 모델의 계획 문서나 `approved`라는 파일명 자체는 새 실행·제출 삭제·최종 모델 잠금 권한이 아니다.

문서 압축 과정의 이전 지시는 `docs/archive/instructions_20260905/`에 보존되어 있다. 원문 아카이브를 현재 지시로 재활성화하지 않는다. 이 파일은 commit 전 작성된 보존 기록이며 push 성공 여부는 실제 Git 원격 SHA로 확인한다.
