# P3 hmax 제거 후보 — 공식 확인

**공식 RMSE 0.608184m / 23.680619점으로 기존 재생성 기준 0.607183m / 23.696500점을 넘지 못했다. 기존 기준본을 보존한다.** 차이는 +0.001001m / −0.015881점이다. 내부 개선의 보편성 또는 Private 성적에 대한 결론은 아니다.

09-06 08:08 KST(UI 분 단위)에 [P3 문제 페이지](https://oceanaidata.org/app/problems/7)에서 단 1회 제출했고 [제출 관리](https://oceanaidata.org/app/submissions)의 신규 채점완료 기록으로 대조했다. 남은 기회는 3→2. 서버가 답안 SHA/ID를 표시하지 않으므로 로컬 SHA·선택 경로·파일명 스크린샷·단일 클릭·신규 기록을 연결하며 ID를 만들어 쓰지 않는다.

- 파일: `artifacts/p3_forward_candidate_cold_20260906_v1/completed/P3/05_answer/submission.csv`
- 1,200행, SHA `d45605922ae8ce699c07405d8361765290c39bd38ddd0123f4e2e9ccd01808c5`.
- 배포 원자료부터 12 backbone+5 router, 학습 QA→추론→새 PID exact replay 1,611.919초. [cold 증거](../p3_forward_candidate_cold_20260906_v1/report-source.md).
- 별도 저장 모델 ZIP 실제 외부 폴더 추출/새 PID 6.088초 exact replay, 추가 fit 0. [saved 증거](../p3_forward_saved_20260906_v1/report-source.md).
- [root 22-check 사전 검증](../remaining_work_completion_20260906_v1/p3-pre-upload-root-qa.json), [공식 영수증](receipt.json).

후보는 공개 결과를 보기 전에 hmax 제거 단독으로 고정했다. numeric-lead를 혼합하지 않았고 공개 점수로 shrink/router/threshold를 바꾸지 않았다. 옛 OOF에서 선택된 고정 shrink 0.2의 이력은 cold 패키지의 `shrink-provenance.json`에 분리해 남겼으며 물리상수나 새 학습값으로 위장하지 않는다. hidden truth 접근, 재시도 제출, 최종 모델 잠금, Git commit/push는 0이다.
