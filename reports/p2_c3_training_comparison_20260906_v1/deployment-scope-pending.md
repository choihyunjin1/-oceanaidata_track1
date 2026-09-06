# 후속 배포 범위 — 결과 확인 전 고정, 아직 미승인·미실행

현재 28-fit historical 실험은 변경하지 않는다. 아래는 root가 terminal·전체 saved-model replay·독립 QA를 확인한 후 유지 후보가 생길 경우 판단할 별도 단계다. 지금 full fit, 공식 입력, CSV, 업로드는 모두 0이다.

1. B3 3-seed 평균이 C60보다 개선된 경우에만 이번에 고정 선택된 challenger를 대상으로 한다. L120 또는 D60 중 결과가 보기 좋은 별도 arm을 다시 선택하지 않는다. 첫 seed 선택 편향, 추가 2-seed 민감도, 3-seed 평균, 모든 결측/계절 악화를 함께 판단 자료로 보존한다. root 승인 없이는 다음 단계가 시작되지 않는다.
2. 새 ID p2_c3_training_comparison_full_20260906_v1에 결과/QA/replay SHA, 정확한 선택 recipe, core/config/source SHA를 봉인한다. 기존 28-fit 경로·lock·모델·현재 CUDA fallback 답안 SHA46d194...c071은 수정하지 않는다.
3. 빈 03_model에서 배포 observations.csv의 공개 학습 정답만으로 선택 recipe를 seed20260901/20260902/20260903으로 각각 처음부터 학습한다. 신규 full fits는 정확히 3회다. historical checkpoint/OOF/과거 답안으로 full 모델을 만들거나 역산하지 않는다. 최종 학습 데이터는 현재 C3와 동일한 지원 가능한 166268행·동일 augmentation/domain weights/gradient penalty다.
4. GPU0 단독/CPU2/DataLoader0, 11-context/8-token-feature DeepSet, 학습률·batch·mask·후처리는 그대로다. 선택 결과가 L120이면 세 모델 모두120epoch/wd0.0001, D60이면 모두60epoch/wd0.001이다. 최종 epoch를 성능에 맞춰 추가 조정하지 않는다. baseline+scale×normalized를 각 seed마다 계산한 뒤 세 seed 산술평균하며 envelope/PAVA projection을 추가하지 않는다.
5. 학습 산출물·키/배열 hash·유한값·source 불변·새 PID 저장모델 재현 검사와 연결된 내부 결과 QA가 모두 PASS여야만 root 승인 범위에서 공식 공개 index/관측 입력을 읽는다. hidden truth는 계속 금지한다.
6. 별도 05_answer에 공식26,061개 키의 station,layer,time,temp CSV를 생성하고 schema/전체키/순서/중복/finite/UTF-8·LF/해시를 확인한다. 새로운 PID에서 전체 답안을 재생성하여 해시 일치까지 검증한다. 기존 C3 답안과는 모델/설정/SHA가 다른 파일로 식별한다. 과거 최고점 file46d194를 새 후보의 점수로 붙이지 않는다.
7. 학습·검증·로그·답안·문서 경로를 분리하고 즉시 root에 exact CSV 경로/SHA/QA/제출 제목을 전달한다. 실제 포털 업로드·기회 확인은 root 담당이다. 이 문서는 업로드나 commit/push를 허가하지 않는다.

예상 추가 비용: 기존 CUDA C3 full3 68.391초를 참고하면 D60은 유사, L120은 epoch 기준 약137초의 학습 계획치다. CPU 병행/하드웨어 상태와 준비·추론·replay 비용은 별도이며 실제 시간은 다시 기록한다. 총3fit/새 wallcap은 실행 전 receipt로 봉인한다. GPU 우선권은 다시 root에게 받아야 한다.

현재 28-fit historical 결과 및 향후 full 후보는 기존 C3와 **내부 비교 표면**이 동일하다. 실제 공식 성적은 exact 새 CSV 제출 전에는 알 수 없으며, 공식 RMSE→점수 역산을 하지 않는다.
