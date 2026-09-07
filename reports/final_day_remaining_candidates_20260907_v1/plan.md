# 남은 P1/P2 후보 준비 — 실행 조정 기록

2026-09-07 사용자 "그럼 실행하세요"에 따라 P1 bracket B 교체와 P2 L120 10-seed+투영 후보 준비를 착수했다. 직전 문서 감사의 새fit0 제한은 그 작업에만 해당한다. P2의 첫 공식 채점 결과 대기는 **후속 후보 준비에 한해** 앞당겼고, 공개 점수로 계수/threshold를 바꾸지 않는다. 자동 업로드/삭제/최종 지정/commit/push는 없다.

## 작업 분리

| 담당 | 범위 | 자원·기존 작업 경계 |
|---|---|---|
| P1 전담 | 원형57844ef2의 B만 bracket107 특징 모델로 교체; 내부 비교·재현·후보·패키지 | CPU4, 원형 O/MS/셀정책/GI/threshold 유지. GPU 필요 시 P2와 조율 |
| P2 전담 | 고정L120 seeds20260901..10 동등평균+정확clip/PAVA, 내부 평가·재현·후보·패키지 | GPU0, 기존L120 CPU2지원 설정 유지(같은recipe비교와CPU경합관리). 7새seed×8fold+full=63fit 계획; 상세seal이 정본 |
| root | 실행 조정·독립QA·공용UPLOAD_SET_2·후보 상태 통합 | 기존P3 supervisor/runner/config 불변; 같은 완료 검사를 반복하지 않음 |

무의미한4h/6h/15시강제종료 제한을 새로 두지 않는다. 다만 현재P3 CPU4 고정조건은 prefixcompletion과freshcold재현 비교를 위해 유지한다. GPU 한 소유자 원칙은 충돌 방지를 위한 작업 조정이며 GPU 자체를 유휴 상태로 묶는 제한이 아니다.

## 해석·검증 계약

- P1 기존 bracket 공식F1 0.785944는 final-inner에서 B107단독 threshold0.1을 선택한 파일9031의 값이다. 그 값을 새 원형조합 B교체 효과로 승계하지 않는다. 원형B의3seed와 이전bracket모델의실제seed/recipe가 같지 않으면 기존가중치를 동일후보로 재사용하지 않는다.
- P2 s3_proj9c5fec38은 완성된 미채점후보. 새s10은 이것과 같은split/row/투영규칙에서 비교하고 B3 primary와all8 pooled·outage위험을 구분한다. per-blockRMSE단순평균 대신SSE/N을 쓴다.
- 내부평가에 이미 노출된fold는 retrospective로 표기한다. worstblock악화는 별도위험이며 자동탈락조건으로 쓰지 않는다. 내부RMSE/F1은 새 공식점수가 아니다.
- 기존모델재사용은 정확SHA/recipe/data/split검증을 전제로 한다. 완성모델이어쓰기와빈폴더새cold재학습을 구별한다. 저장추론만으로전체학습재현을 주장하지 않는다.
- 후보준비는 사전등록→합성계약검사→실제학습/내부평가→독립QA→답안→별도PIDreplay→SOURCE_ONLY/SAVED패키지순이다. 실행과완료는실제PID/receipt로만갱신한다.

## 독립 QA 체크리스트

1. seal의source/config/seed/fitcount와실제receipt일치, 소비된폴더미변경.
2. 배포자료전용·목표누출없음·Public역산없음, train-only선택경계와지원행 명시.
3. comparator와candidate 키/분모/단위 동일, pooled직접재계산, 평균개선과위험분리.
4. 공식추론 전에 내부QA완료, hidden truth/sample예측값미사용.
5. CSVschema/key/order/finite/중복/행수/SHA 및별도PID재생.
6. 코드부터재학습경로와저장모델경로를구분,새ZIP추출재생,당일최종포털적격성은별도미확인.

상태(11:32 KST): P1 11:31:01 launcher8380/worker41780 첫 O fit 실제 진행 확인. P2 11:28:10 launcher2408/worker11152 시작, 7/63fit 완료 후 B1 진행 확인. 두 stderr0. P1/P2 신규후보완료·공식성적은아직주장하지않는다. P2 후속 portable fresh cold10fit는 별도이며 시행 시 총새fit73/재사용27로 구분한다.

11:26 사전검토 갱신: P1 기존bracket B는1seed이며원형B3와exactreuse불가. 전담계획은Q3/Q4 B80control3seed와B1073seed paired12fit +fullB1073fit =15fit이다. 이 paired 비교의평가/전처리표면이역사적원형OOF와다른부분은별도명시하며그차이를특징효과로귀속하지않는다.

11:27 pre-fit 정정: 옛 O OOF의7일purge와MS의21일purge차이를확인해O2fit도같은21일계약으로새로맞춘다. 따라서총17fit(O2+controlB6+bracketB6+fullB3), 같은O를두팔에공유하고MS의정확287862키/split검증후재사용한다. 위15fit는사전검토초안이며실행된fit수가아니다.
