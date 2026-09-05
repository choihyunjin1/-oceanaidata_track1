# 첫 실행 전 경로 검사 수리

2026-09-05 18:57 KST launcher PID 31220은 source runner 검사에서 `run_` 접두사가 빠진 파일명을 참조하여 종료됐다. ATTEMPT_LOCK 생성 전이며 학습·전이 추정·내부/외부 평가·공식 접근 모두 0이다. 원래 execute.stderr.log/execute.stdout.log는 보존했다.

root 승인에 따라 실제 import된 screen 모듈의 __file__을 기대 경로와 대조하는 resolver로 수정하고, 실재 파일과 완료된 screen receipt SHA를 검증하는 테스트를 추가했다. 모델, 데이터, lambda=1, Laplace=1, threshold, selection, split에는 변경이 없다. 첫 실제 decoder 실행은 새 로그 execute.v2.stdout.log/execute.v2.stderr.log를 사용한다. 이는 성능을 확인한 실험 재시도나 재튜닝이 아니다.
