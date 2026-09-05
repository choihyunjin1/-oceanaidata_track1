# P1 수심 계약·기존 부품·고정 decoder 실행 결과

## 결론

**승인된 A/B/C 분기를 완료했지만 새 개선 후보는 나오지 않았다.** A의 연도 의존 수심 lookup은 제거됐고16회 실제 학습과 재로딩 QA도 끝났으나, 내부 pooled F1은 기준선보다0.002213 낮았다. 후속 C의 고정 decoder는 그 손실 일부만 회복했다. B는119행 split 및 router purge 불일치를 입증해 부적격 OOF 결합을 하지 않았다. 과거 모델을 성능 실패나 외부자료 위반으로 일괄 분류한 결과는 아니다.

동일421,032행에서 선택 규칙은 기존 control을 유지한다. 이미 제출한 동일 control을 다시 제출하도록 추천하지 않는다. 새 year-safe full model은 기술적으로 보존돼 있으나 현재 근거상 **개선 후보가 아니라 별도 INFORMATION_ONLY 검토 대상**이다. 공식 추론·제출 여부는 root의 추가 QA/지시 범위이며 이번 실행에서 공식/hidden 입력, CSV, 업로드는 모두0이다.

## 실제 내부 채점

| 고정 정책 | pooled F1 | control 대비Δ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| exact 기존 control | 0.851174240 | 0 | 12,794 | 1,213 | 3,261 |
| A year-safe, earlier-inner 선택 | 0.848961444 | −0.002212796 | 12,793 | 1,290 | 3,262 |
| A + C 고정 always-ON | 0.849871585 | −0.001302655 | 12,740 | 1,186 | 3,315 |
| A에 기존 control threshold 고정, 진단 전용 | 0.850061301 | −0.001112939 | 12,827 | 1,297 | 3,228 |

선택된 A 정책은 기존 control보다 TP를115개 추가하면서116개 제거했고, FP264개를 추가하면서187개 제거했다. C는 A보다FP104개를 줄였지만TP53개도 줄였다. 따라서 단순히 오탐 감소만 보고 성능 향상으로 부르지 않는다. 최종 primary는 pooled TP/FP/FN의 F1이며 fold F1의 단순 평균이 아니다.

| 평가 | control F1 | A F1 | A+C F1 |
|---|---:|---:|---:|
| Q2, 133,170행 | 0.772156714 | 0.773007298 | 0.768818234 |
| Q3, 176,619행 | 0.878226811 | 0.878526093 | 0.879609929 |
| Q4, 111,243행 | 0.914581328 | 0.905495572 | 0.911732655 |

A는Q2/Q3에서 소폭 개선, Q4에서−0.009086 악화했다. 이 표를 보고 Q3 우승정책과Q4 fallback을 사후 조합하지 않았다. 평균 악화 때문에 개선 후보로 고르지 않은 것이며, 모든월양수/3점/anchor제거0 같은 새 hard gate를 붙인 것은 아니다.

## 불확실성·리스크

7일 공유 시간블록38개를2,000회 paired resampling(seed20260905)한90% 구간은 A [−0.006314,+0.002370], A+C [−0.005435,+0.002958]다. 모든 정점은 같은 날짜 블록 가중치를 받는다. 이 표는 이미 반복 노출된 historical development에서의 불확실성이다. fresh holdout·독립 유의성·공식 수송성 증명이 아니며 CI가0을 포함한다는 이유만으로 탈락시키지 않았다.

월별 및 station/layer별 전체 위험은 [decoder-result.json](decoder-result.json)에 있다. 대표적으로 A는S-ORS/L4를 개선했지만I-ORS/L5·L7과S-ORS/L2에서 손실이 있었다. 관측 outlier나 평가행을 제거하지 않았다.

## 기술 가설은 무엇까지 확인됐나

A는 train/inner/outer/fulltrain에서 같은 current-depth 함수를 사용했다. 변경은 `nominal_depth_m`과 `depth_regime` 두 열뿐이다. 현재 depth를2m 단위로 반올림하고 실제 결측을 explicit unknown으로 둔다. 기존80특징의 나머지 offline 시간/peer/rule 함수, recipe, seed20260813, purge21일, inner60일, grid는 동일하다.

- synthetic에서 관측·시간을 그대로 두고 year키만 바꿔도 전체 특징이 같다.
- Q2의 관측 depth가 있는데 nominal-depth가 결측인65,230행은 A에서0행이다. unseen `(station,year,layer)`는55,742행이며, 나머지 결측 원인과 구분했다.
- A의 outer nominal-depth missing은Q2/Q3/Q4=607/104/39행으로 실제 raw-depth missing과 같다.
- 그러나 **계약 수정의 성공은 점수 향상의 성공이 아니다.** 개별 depth를 쓰는 것은 train년-배치 median보다 다른 잡음/범주 변화를 만들 수 있다. 현재 결과만으로 원인 기여를 추가로 단정하지 않는다.

## 학습·저장·재현

- 신규 O/B×3fold×inner/outer=12 fits. 기존 control O/B12개는 model/dependency/recipe hash와 inner selection, outer key/확률/binary exact replay를 확인하고 재사용했다. control 재학습0.
- 최종 inner: 배포 train의 마지막 시각만 사용해2025-10-11 03:10~12-10 03:10 KST60일을 정의, 그 이전21일 purge 후2fits. 선택은 balanced, threshold0.2, decoderOFF다. oldQ4 threshold를 복사하지 않았다.
- fulltrain: 배포 train776,706행으로 original/balanced 각각1회, 합2fits. 합계16fits. 전체 A765.219초(CPU4/GPU0), final-inner+full 포함12분45초. C0 backbone/transition4회41.328초.
- A fresh-process QA21.594초:6개 outer 저장모델 확률 재계산의 max-abs-diff 모두0,16개 모델 hash 일치. full models 각각train-only4,096행 probe의 finite 검사PASS.
- 학습 중 원모델 vs 저장·재로딩 확률은exact였지만, **16fits를 두 번째 재학습한 결정성 검사까지 했다는 뜻은 아니다.** 기존 exactly-once 실험을 재실행하지 않았다.
- 연구 artifact에는01_data manifest,02_code snapshot,03_training,04_models,05_answer(empty),06_report 폴더를 분리했다. 배포 데이터 원본을 복제하지 않았고 경로는환경변수로 받는다. 02_code는연구 snapshot이며 모든 의존성을 포함한 독립 공식 제출ZIP으로 검증한 것은 아니다. 기존 최종 패키지를 변경하지 않았다.

최종6시간 조건에는 실제 fulltrain2fits가 충분히 짧다는 시간 증거가 있으나, 향후 공식 제출 전 source부터CSV까지 별도 end-to-end QA가 필요하다.

## B 후속 분기를 멈춘 정확한 이유

[출처 감사](provenance-audit.md)와 [key 영수증](provenance-audit.json)에 상세히 남겼다. 전체421,032키가 같아도119행은옛Q3/현재Q4다. 옛e150은21일 purge이나 그 anchor O/B source는7일이다. 현재21일 calendar 계약의 OOF를 만들려면 추가9 historical+3 full GPU fits가 필요하며 root가 이번 시간 예산에서 자동 실행하지 말라고 지시했다. GPU학습0, oldcheckpoint/OOF/lock 수정0이다.

## QA·추적 파일

- [독립24검사 영수증](cycle-independent-qa.json):24/24 PASS. sklearn confusion/F1와추가/제거수 재계산, exact key/hash/fitcount/접근0 확인.
- focused pytest11 PASS, Ruff6개새Python파일 PASS.
- [실행한 companion notebook](analysis-companion.ipynb):10cells top-to-bottom 실행PASS. aggregate만 표시하며 원시행·공식값이 없다.
- A runner SHA `4f6db75d1ba45cba6e5bd4159712a6d1ab1851e691dbb21c371bbed09b5e2a0c`, config SHA `f6514ffe6fe868d4d44f01e4b6c57dd6b702404878764e200ac1caec0ff9d972`.
- A result SHA `e1c9051222dbd5cc0a391c699ee3f1ab5bb73d628c17aed139bf28b76b9accd7`, C result SHA `ef2c1ed0f2d9251bee208f2317f713ab8016beba53aab48c9c79a26e52bdc5bd`.
- [A 사전등록](preregistration.md), [C 사전등록](decoder-preregistration.md)과 실행코드가 별도로 보존돼 있다.

현재 공식 기준은9월5일 clean control F1 0.790733 /27.771400점이다. 새A/C의 **예상 공식 점수: 미산정**. 역사최고28.909341과의 차이를이번 후보의예상상승폭으로 쓰지 않는다. 이번 P1 lane은 git stage/commit/push를 하지 않았다.
