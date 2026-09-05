# P3 점수 복구 대조 — 실행 전 고정

배포 자료만 사용하는 기존 591-feature compact residual LightGBM에 기상 묶음 결측 증강 한 가지를 대조한다. 과거 공식 점수 역산 alpha 및 외부자료 계보는 전부 제외한다. GPU는 P2에 예약되어 있으므로 P3는 **CPU 2 threads**, 한 fit씩, RSS 10 GiB와 전체 screen 2,700초 상한을 사용한다.

## 비교와 금지선

- 입력은 `P3_DATA_DIR`의 train_wave/train_atmos와 hash가 고정된 train cache만 허용한다. 공식 입력/답안/숨은 정답/업로드는 0이다.
- 기존 clean corrected OOF의 single/multi/equal/prequential router/fixed persistence-shrink 및 TabPFN 단독/25% blend를 동일 키·truth에서 0-fit 비교한다. 이들은 저장 당시 clip된 값이며 raw preclip 예측을 역복원하지 않는다.
- 기본 표면은 기존 181사례/1,086리드행, 78h 정점별 간격, 181개 정점×episode, 3 expanding folds다. 이미 여러 번 본 표면이며 fresh confirmation이 아니다.
- fit은 3 folds × 2 arms × 2 seeds = 12회. 기존 `src/p3_wave/models.py` LightGBM 기본값을 재사용하며 tree 700, lr 0.025, leaves 15 등 모델 조건은 두 arm에서 동일하다. thread만 총 자원 배분에 맞게 2로 고정했다.
- train 기상 지원량이 한 개 이상 양수인 사례를 원본·전체 기상 가림으로 복제한다. 원본/가림 각각 기존 case weight의 0.5를 받아 합계가 원래 중량과 같다. 완전히 미관측인 사례는 중량 1.0으로 복제하지 않는다. 같은 anchor의 6개 리드는 일관되게 가린다.
- 바람·기압·기온·습도 및 풍파 정렬/풍력 proxy까지 전부 가리고, valid indicator는 0, 기타 파생값은 NaN으로 둔다. 합성 289행 context에서 원시 기상을 먼저 가린 뒤 1,275개 요약 특징을 재생성한 결과와 정확히 같아야 한다. hs/target은 보존한다.
- metrics는 raw per-seed, clipped standalone, 2-seed 평균, 사전고정 25% 모델+75% clean fallback을 구분한다. 통합 RMSE는 전체 SSE/행수로 계산한다. 정점·리드·fold 악화는 위험으로 표시하며 숨기지 않는다.
- 무 backbone-fit 후속 후보는 미리 정한 3개뿐이다: TabPFN25 6h-only, single25 all-lead, multi25 all-lead. 6h 후보는 이전 결과로부터 제안된 사후 탐색이며 독립 확인이라고 쓰지 않는다. 공식 점수로 가중치를 조정하지 않는다.
- 45분 또는 RSS 예산 초과는 미완료로 기록한다. tree 수를 몰래 줄이거나 결과 기반 재실행하지 않는다. 첫 fit 소요시간과 자원 실측을 root에 보고한다.

## 산출물과 다음 판단

`zero-fit-audit.json`은 현재 규정 기준 출처와 동일 표면 검사를 기록한다. 모델·raw per-seed OOF·root QA용 NPZ는 ignored `artifacts/p3_score_repair_20260905_v1/`에만 둔다. 최종 집계·hash는 이 보고서 폴더에 기록한다. 예상 공식 점수를 약속하지 않으며 조건부 환산만 표시한다. deploy 후보는 배포 train→모델 저장→별도 프로세스 추론 경로가 마련된 뒤 root에게 보고하고, 공식 입력 접근 전에 별도 승인을 받는다.
