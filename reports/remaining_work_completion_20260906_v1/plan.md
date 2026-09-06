# 남은 순서 실제 완료 계획 — 2026-09-06

사용자의 "전부 진행하세요"에 따라 미완료 작업을 수행한다. 이전 게시 checkpoint와 봉인된 실험은 보존하며, 이 계획은 점수 역산·외부자료·최종 모델 잠금·Git push 권한을 추가하지 않는다.

| 담당 | 범위 | 사전 자원 상한 | 완료 조건 |
|---|---|---|---|
| root | P1 frozen9031 답안 공식 채점 | 1회 업로드, 학습0 | QA26 재확인·화면 파일확인·접수와점수 확인; 06:39 완료 |
| P1 worker | bracket 장기 saved replay + cold-start portable | CPU2/GPU0, 새 cold4fit/3600s | 기존봉인불변, ZIP새추출·새PID·9031답안비교, 상수/출처/학습검증 |
| P2 worker | copula 수치정정 완료 | CPU2/GPU0, 0-fit검증→추가48fit, 남은3590.515s 학습cap | 기존40fit검증재사용, 전체88fit과동일평가표면·독립QA·후보판단 |
| P3 worker | numeric lead, 이후 hmax 제거 독립변형 | CPU2+독점GPU0, numeric15backbone/90분; hmax10/60분, router별도 | 지원parameter/feature합성검사, 성능보기전자원파일럿, 동일기준비교·QA |
| root | P1 train-fit 범위규칙·셀정책 독립비교 | 새 backbone fit0, CPU2이하 | 사전등록, 기존OOF지문검증, inner정책/outer평가분리·위험보고 |

P2/P3는 내부 검증된 개선 완성정책이 생기면 별도 full학습·동결·답안/replay·portable 준비까지 진행한다. 동일 사전등록의 technical stop을 덮어쓰거나 성능을 보고 범위·강도·평가기간을 조정하지 않는다. 새 후보 업로드는 root가 현재 횟수와 SHA를 확인하고 수행한다. 세 문제 기준 패키지는 삭제·교체하지 않는다.

P1 추가 제안은 공식 결과를 보기 전 사용자에게서 전달된 가설이다. 0/35 고정범위의 관찰을 공식 FP0 보장으로 사용하지 않고, train 라벨0 min/max로 학습한 범위와 inner OOF 셀별 O/B 선택을 분리 평가한다. 라벨0 자연급변 삭제·가중축소는 하지 않는다. 상세 비교 계약은 별도 experiment config에 봉인한다.

## 완료 갱신

- P1 공식 업로드: 06:39 KST F1 0.785944 / 27.644124점. 기존 5971 대비 F1 +0.008195 / 점수 +0.217805. [영수증](../p1_bracket_official_submission_20260906_v1/receipt.json).
- P1 package: 새 ZIP 빈 모델 cold 4fit·독립47QA·새 PID replay 모두 완료,753.030초에 기존9031 답안 exact. 저장 ZIP 추론도29.234초 exact. [보고서](../p1_bracket_portable_20260906_v2/report-source.md), [root 지문 대조](p1-package-root-check.json).
- P1 범위/셀: backbone0fit/201.28초, 독립190QA+새PID6replay PASS. 주평가 Δ0/−0.005102573으로 모두 비승격. [기록](../p1_trainfit_postpolicy_20260906_v1/report-source.md). 새 CSV/제출 없음.
- P2 수치정정: 추가48fit/전체논리88fit 완료, 독립1117QA·fresh replay32·root73산술검사 PASS. in-sample만 주평가 평균 개선, crossfit은 비승격. [연구](../p2_crossfit_copula_forward_numeric_20260906_v2/report-source.md).
- P2 별도 full scratch: 4fit/167.125초·학습QA17·CSV새PID/ZIP추출replay·독립73QA·root23대조 PASS. 07:28 공식 CPU기준0.489080→copula0.475174℃로 개선했으나 보존CUDA0.455143℃ 미달. 기존CUDA기준유지, 오늘남은1회. [공식기록](../p2_copula_official_submission_20260906_v1/receipt.json).
- P3 numeric: 15backbone+8router 완료, 독립149QA·전체103602행fresh replay PASS, 내부 Δ−0.000500418m, CI0포함. hmax 독립10+4는 학습완료 후 fresh replay/QA 진행 중. 아직 새 후보 공식 성적이나 전체 cold 완료를 주장하지 않는다.
- P3 package: 별도 source-only prepare 후 고정 후보 한 개를 12backbone+5router whole-cold로 재생성한다. 이 경로가 full2+1까지 포함하므로 checkout의 별도 warm full2+1은 중복 실행하지 않는다. 모델 잠금·Git commit/push는 이번 실행에 포함하지 않는다.
