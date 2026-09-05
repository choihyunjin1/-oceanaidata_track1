# 결론: 새 P1 clean 전체학습 답안 준비, 공식 미채점

과거 답안이나 저장 가중치에 의존하지 않고 운영진 train 776,706행으로 XGBoost와 event-day LightGBM을 각각 1회, 총 2회 새로 학습했다. 새 CSV 169,011행의 스키마·키·순서·정수 0/1·유한값 검사가 PASS이며, 복제된 코드 폴더의 별도 프로세스에서 동일 답안을 byte-exact 재생성했다. 공식 업로드는 0이다.

- 답안: `artifacts/p1_clean_control_fulltrain_20260905_v1/05_answer/P1_submission.csv`
- SHA-256: `064ef022faf2a3e8bc7c70633210847aa494060858374aa43f28f4eced84ec43`
- 6,929,481 bytes, 169,011행, 양성 6,505행. 형식 `station,year,layer,time,label`.
- 학습 160.234초(O fit+save+probe 95.062초, B 40.187초 포함), 공식 추론 14.250초, 별도 저장모델 replay 14.484초.
- 두 저장 모델의 train probe 재생성은 exact, 새 CSV 전체의 fresh-process 재생성도 exact.
- CSV의 공식 점수는 미확인이다. 역사적 28.909341점이나 과거의 최고점은 이 파일에 귀속하지 않는다.

## 무엇을 학습하고 무엇을 고정했는가

XGB 1-fit, LGBM 1-fit, encoder/train 통계는 배포 train만으로 학습했다. 기존 2025 Q4의 inner 선택 B_union, O high=0.2 / B high=0.3을 그대로 가져왔다. low ratio=0.5, close_gap=0, minimum_run=12. 추가 calibration search 0, decoder OFF, 전이 추정 0이다.

먼저 수행한 long-flank screen은 강한 동일행 control에 실패했다. 후속 binary decoder도 사전등록 inner on/off 전략이 실패했다. 따라서 둘을 섞거나 다른 threshold를 다시 찾아 결과를 개선한 척하지 않고 새 clean control만 보존했다. 같은 개발 평가면 control F1 0.851174와 XGB-alone 0.843227의 차이는 참고 근거이지 이 전체학습 파일의 공식 점수 예측이 아니다.

## 데이터 접근·재현 경계

모델과 recipe를 봉인하기 전 공식 입력 접근 0. 이후 승인된 inference 1회와 verify 1회에 각각 test 169,011행과 sample의 key 169,011행만 접근했다. 따라서 test 고유행 169,011이며 두 추론 프로세스의 합산 로드 행수는 338,022이다. sample prediction 값·hidden·외부자료·기존 답안 CSV·기존 checkpoint는 0이다. verify가 읽은 답안은 바로 이번 실행에서 만든 새 CSV 하나뿐이다.

새 `02_code` 작업 디렉터리 및 그 `src`를 PYTHONPATH로 지정했고, 20개 프로젝트 모듈이 모두 복제된 트리 내부에서 import되는 것을 별도 확인했다. 원 연구 repo의 src를 상속하지 않았다. 단 동일 Python .venv 및 이미 설치된 dependency를 사용했으므로 인터넷 차단 새 환경 설치와 별도 fresh 2-fit 재학습 검증까지 완료했다고 말할 수는 없다.

## 남은 한계

train의 station×year×layer depth 사전에는 2026 키가 없으므로 공식 169,011행 전부 nominal-depth가 missing이다. 원래 테스트한 계약을 보존했으며 현재 fallback을 넣지 않았다. Q4-inner와 공식 입력의 covariate shift 때문에 성능 하락 가능성이 남는다.

현재 파일은 업로드 전 검증용 후보다. 최종 ZIP 구성·clean machine 설치·6시간 전체 재현 검증·공식 채점은 별도 미완료다. 기존 final 패키지는 변경하지 않았다. 모든 P1 학습/평가/추론 프로세스가 정상 종료했고 추가 실험을 자동 시작하지 않는다.
