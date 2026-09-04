# P1 v32g deployment verdict

## 결론

이 파일은 기술적으로 제출 가능한 비중복·비퇴화 후보지만, 내부 중심 추정은 현 챔피언보다 낮다. 따라서 `READY AS EXPLICIT INTERNAL-NO-GO INFORMATION PROBE`이며, 사용자가 남은 제출권을 공식 정보 획득에 쓰겠다는 위험을 명시적으로 감수할 때만 업로드한다.

- CSV: `C:\Users\cedis\Downloads\해양 해커톤 제출용\20260901_P1_V32G_CATBOOST_UNION_SCORE_PRIORITY_READY_V1\P1_submission.csv`
- SHA-256: `2f6b115dd706d42b8851685af4c2e4a065f111e2b51b8dee9bc6170396e5e279`
- 169,011행, 양성 6,419행, 음성 162,592행
- 현 챔피언 대비 add-only 23행, 제거 0행
- 현 챔피언 SHA와 비중복
- 내부 예상 중심 `28.902994`, 현 챔피언 `28.909341`, 차이 `-0.006347`점
- 내부 CI90 `[28.891794, 28.912180]`
- 배포 데이터만 사용한 scratch CatBoost 1회 fit; 외부/KIOST/hidden/pretrained/upload 0

독립 배포 QA는 30/30 PASS다. 공식 점수는 숨은 정답을 열지 않고는 알 수 없으며, 내부 결과 자체는 `NO_GO_INTERNAL_GATE`였다는 점을 제출 제목과 기록에서 숨기지 않는다.
