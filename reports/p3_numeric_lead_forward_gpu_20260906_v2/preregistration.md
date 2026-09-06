# Numeric lead — 새 GPU 자원 수정안

2026-09-06 결과 열람 전 고정. 기존 `p3_numeric_lead_forward_20260906_v1`의 CPU 2-fit resource stop, 모든 모델/lock/코드는 보존하며 재시작하지 않는다.

변경 가설은 single `lead_h` string/cat_features에서 float numeric으로의 전환 하나. single은 기존 CPU2 700trees/depth6/.035/l2=8/random_strength=.2, multi는 원 clean GPU0 Plain 1200trees/depth7/.03/l2=10/.15이다. multi CPU override만 취소한 새 자원 baseline이며 예전 CPU baseline나 sparse 181case GPU 성능을 승계하지 않는다. 양 arm은 이 새 실행에서 fit한 같은 GPU multi를 사용하고, router는 각 arm의 purged earlier-OOF만으로 Ridge10/temp2/strength.5를 재적합한다. 장기 persistence .2는 과거 train-only adaptive 후 고정한 계보 그대로, 공개 점수 역산0.

v5 Q1_2024 warmup 후 정확5fold, 78h purge+raw20min station high-run episode exclusion, 24360anchor/591features, validation17267anchors/103602rows. 주평가 unweighted totalSSE/rows RMSE. CI90은2000회 같은 station-episode paired bootstrap이며 독립 storm/공식 향상 확률이 아니다. worst-quarter/station/lead/wind-observed를 별도 보고한다. onset/greedy NOT_ENABLED. 평균delta<0이면 후보보존,0.8 gate없음. 새로운배합/lead별선택없음.

실제 budget15backbone(controlsingle5+multi5+numericsingle5)+router8,90분. 첫 예정 control single+multi를 완료 후 규모선형1.2margin으로15fit 전체시간 추정, 성능을 열람하기 전에 cap초과면RESOURCE_STOP. 이후 성능 미열람으로 baseline10fits 완성→numeric5fits→두 정책 점수 계산. CPU2, GPU독점은 multi만. GPU callback지원이 없어GPU fit전/후 단계시간을검사하고1200trees고정한다. OS강제 wall-timeout이 아니며 root의 process/progress외부감시를 병행한다. cap초과/기술오류 시 현재attempt보존,자동재시작0. 예상55–70분은 과거CPU single149.738s 및 현재장비GPU clean 이력 근거인예상이며실측아님.

source-derived cache는 fresh-clean prepare의 source/config/hash/키연결 후 허용, 이전 model/OOF/answer입력0. source current+6targets 전체 대조,45rawcontext특징대조, 미래48h밖 perturbation 불변, nativeCPU/GPUtiny합성검사→Ruff→preflight/seal→execute. 완료 후 saved-model fresh PID exactreplay 및독립SSE/chronology/hash/sliceQA. 내부/공식점수는구분, 공식score예상은산식/분포정보부족으로미산정.

다음 별도고정 가설은 [hmax 사전등록](../p3_hmax_removed_forward_20260906_v1/preregistration.md). numeric와결합하지 않고새categoricalbaseline을 exact-hash 재사용한다. 각개선정책에대한full학습/추론/portable은후속새배포receipt로분리하며이historicalrunner공식입력/CSV/upload/fullfit는0.
