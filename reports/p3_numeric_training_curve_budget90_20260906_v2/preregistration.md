# P3 numeric curve — budget90 v2 사전등록

**이 변경은 사용자의 60분 초과 진행 승인에 따른 자원 보완이며, 성능을 본 뒤의 재튜닝이 아니다.** 원 [60분 시도](../p3_numeric_training_curve_20260906_v1/report-source.md)는 RESOURCE_STOP으로 보존한다. 원 시도는 예정 baseline inner 2 fits를 완료했지만, 결과를 열람하기 전에 비용 예측 65.693분으로 중단됐다. 새 시도는 GPU 배정 전 추가 실제 fit/예측 0이며, 아래 계약을 먼저 봉인한다.

## 바꾸지 않는 과학적 계약

원 config의 `recipes`, `held_inner`, `outer`, `fixed`, `baseline`을 새 실행 시 구조적으로 동일한지 확인한다. 동일 591 특징 및 hmax 보존, numeric lead, 기존 가중치, router 및 long persistence 0.2는 불변이다.

| recipe | single iterations / lr | multi iterations / lr |
|---|---:|---:|
| baseline | 700 / 0.035 | 1200 / 0.03 |
| compact | 525 / 0.04666666666666667 | 900 / 0.04 |
| gentle | 1050 / 0.023333333333333334 | 1800 / 0.02 |

Iteration×learning-rate를 고정하지만 학습 결과가 등가라는 뜻은 아니다. Earliest outer train 내부의 2024-03-01 이상~03-25 미만 held-inner에서만 같은 여섯 lead의 unweighted pooled SSE로 recipe를 한 번 선택한다. Exact tie는 baseline 우선이다. Inner train 4,664 anchors / 27,984 rows, validation 1,729 anchors / 10,374 rows, 78h 및 station/raw-high-run episode 분리를 그대로 쓴다. 모든 inner target은 earliest outer context보다 앞서 이용 가능해야 한다.

Baseline 선택은 exact-hash 과거 numeric OOF를 재사용하고 추가 outer 학습을 하지 않는다. 다른 recipe 선택 때만 고정 5 forward folds의 10 backbone + 4 earlier-only router를 새로 학습하여 기존 numeric과 같은 103,602행에서 평가한다. 선택 timestamp를 outer fit보다 먼저 저장한다. 평가면은 이미 노출된 retrospective 면이며 virgin 검증이라고 부르지 않는다. 전체 pooled RMSE 평균 개선이 주 기준이고 quarter/lead/missingness/cluster CI 위험은 별도로 보고한다. 위험값 0.8 같은 자동 탈락 gate는 없다.

## 자원 및 원 모델 재사용

- 사용자 명시 승인에 따라 새 execute의 hard elapsed deadline은 5,400초다. CPU 2, root가 배정한 독점 GPU 0만 사용한다. 원 60분 lock/runner/config/progress/receipt/model은 수정하거나 재시작하지 않는다.
- 원 성공 2 fits를 재사용하고 최대 18 신규 fits만 추가한다: 다른 두 recipes의 inner 4 backbone + 선택된 recipe의 outer 10 backbone + router 4. 고유 모델 전체는 원 계획과 같은 최대 20 fits다. Baseline 선택이면 새 inner 4 fits만 수행한다.
- 원 자원 pilot은 137.049017초 실측 × 고정 보수식 = 3,941.581초다. 이 값과 비용식은 성능과 무관하며 바꾸지 않는다. 새 실행 중 남은 한도를 결과를 보고 늘리지 않는다. GPU CatBoost 내부에는 CPU callback이 없으므로 별도 process timer가 기한 종료를 담당한다. OS 수준 독립 샌드박스 증명은 아니다.
- 원 single SHA `e26e30cb5c0daf6c3e49f09276097dcc9d0342c7b0ec305fd5c796eeb9b8b811`는 원 fit receipt와 일치한다. 원 multi SHA `f7197f8aea417642fb51047d2f69dc0c4f109d858f1c812d154d990c04913ff2`는 **중단 후 acceptance-time hash**이며 개별 원 receipt는 메모리에서 소실됐다. 이를 save-time digest처럼 쓰지 않는다.
- 재사용 전 source/config/data/anchor/inner key hash, native 700/1200 trees, CPU/GPU recipe·seed·thread/lr/iterations, feature 순서와 cat index `[0]`을 검증한다. 원 두 모델만 새 소유 폴더에 exact copy한다. 모델 교체·재학습은 없다.
- 새 봉인 후 execute에서 두 saved models의 동일 held-inner 예측을 계산한다. Single은 원 prediction SHA와 exact 비교하고 multi는 새 계약에서 처음 저장하는 prediction SHA임을 표시한다. 이 예측의 SSE는 나머지 고정 inner 4 fits가 끝난 정해진 선택 시점에만 계산한다.

## 검증과 인터페이스

[합성 검사](code-qa.json)는 24 PASS 및 Ruff PASS다. 실제 재사용 native acceptance 및 preflight가 완료되어야 seal을 만든다. 이후 새 PID에서 전체 저장 model predictions 및 prior-only router를 replay하고, 독립 SSE/row denominator/paired cluster CI/선택·시각·fit count/hash/공식 0을 QA한다. Single+multi+router의 own training probe 점수를 outer 성능으로 쓰지 않는다.

```powershell
$env:P3_DATA_DIR = '<organizer P3 dataset directory>'
python scripts/run_p3_numeric_training_curve_budget90_20260906_v2.py --stage preflight
# root GPU 배정 이후에만, exactly once:
python scripts/run_p3_numeric_training_curve_budget90_20260906_v2.py --stage execute --gpu-released-by-root
python scripts/run_p3_numeric_training_curve_budget90_20260906_v2.py --stage replay
python scripts/run_p3_numeric_training_curve_budget90_20260906_v2.py --stage qa
```

현 scope에는 full fit, 공식 입력, CSV, upload, Git 또는 최종 모델 lock이 없다. 기존 numeric `ff42a6a0…37960` 후보 및 root의 별도 numeric whole-cold 12+5 패키지와 독립이다. 튜닝 개선이 있더라도 새 full 학습·제출은 별도 결정 전까지 실행하지 않는다.
