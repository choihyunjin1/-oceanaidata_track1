# P2 portable cleanroom 사전 계약

기존 적격 clean regeneration v6 C3를 새 빈 폴더에서 3 seeds × 60 epochs scratch 학습한다. 원본과 기존코드/모델/답안/lock은 불변이며 이전 답안은 읽지 않고 SHA만 최종 대조한다.

- 신규 학습은 GPU 3 fits, 추론도 원래 CUDA 경로로 2회(서로 다른 PID) 실행한다. GPU 사용 전체 약90초 예상: 기존 v6 train69.094초 근거. 다른GPU작업과 중복하지 않는다.
- 패키지 기본 CPU2이나 이번 동시실행은 root 자원배정에 따라 `--cpu-threads 1`; 이는 원래 v6 fit 내부 설정과 같다. 추가 모델/threshold/결과맞춤 변경 없음.
- 실행은 원 저장소 밖 OS 임시 폴더로 portable template만 복사하고 `python -I -B`로 시작한다. P2_DATA_DIR만 배포 source 위치를 제공하며 code는 개인 절대경로를 포함하지 않는다.
- 학습은 observations.csv만, 추론은 공식 index/sample 키만. sample 값/hidden/비배포자료/기존 모델·OOF·답안 0. 원본 데이터 패키징 0, 네트워크/업로드/Git 0.
- Python audit hook/network socket 차단은 OS 또는 물리적 네트워크 차단 검증과 다르다. 현재 설치 환경을 사용하며 새 venv/운영진 장비 검증은 수행하지 않는다.
- 최종 목적 SHA: 46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071. 불일치하면 그대로 보고하고 원본 recipe를 결과에 맞추어 조정하지 않는다.
- 진행전 synthetic 6 tests(P2목표mask/augmentation질량/architecture/gradient/schema/원repoimport부재/nativewriter 실제guard/빌더빈폴더) 및 Ruff PASS가 필요하다. 실제생성모델 hash·전체키·finite·순서·별도PID replay 독립QA를 별도로 수행한다.

완료 패키지는 artifacts/portable_cleanroom_20260906_v1/P2에 복사 보존하고 하나의 canonical report-source.md로 완료/미완료를 구분한다. 이전 copula technical failure 재시작은 하지 않으며 기술수정 설계만 후속 문서화한다.
