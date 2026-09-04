# Public transport calibration v2

## 결론

최소 승격 폭 `+0.01점`은 유지한다. 다만 서로 다른 모델·개입 계열의 Public
역전 잔차를 교환 가능한 표본처럼 취급하던 v1의 문제별 단일 최악값은 새 실험부터
사용하지 않는다. v2는 실험 결과를 보기 전에 `family_id`와 복잡도 tier를 등록하고,
동일 family의 공식 잔차 → 동일 tier의 최악 잔차 → global 최악 잔차 순서로
penalty를 고정한다. 이것은 통계적 신뢰구간이 아니라 6개 공식 pair에 근거한
계층적 경험 guardrail이다.

기존 NO_GO 후보를 소급 PASS시키지 않는다. Public 점수는 family-level penalty
영수증으로만 사용하며 row, event, threshold 학습 label로 쓰지 않는다.

## 새 실험의 주요 raw PASS 기준

| 사전등록 family/tier | penalty | raw 기대점수 하한 |
|---|---:|---:|
| P1 exact fixed add-only union | 0.005383691 | 0.015383691 |
| P2 exact fixed drop-layer | 0.007990193 | 0.017990193 |
| P3 exact fixed KMA long-lead factor | 0.049586054 | 0.059586054 |
| unseen low-DOF fixed | 0.049586054 | 0.059586054 |
| unseen smooth learned profile | 0.121682092 | 0.131682092 |
| hard conditional router / unknown compound | 0.321905690 | 0.331905690 |

P3의 단순 고정 long-lead factor는 low-DOF 계열을 쓸 수 있다. 반면 learned
support selector, tree gate, 또는 representation과 router를 함께 바꾸는 후보는
global hard-router penalty를 사용한다.

## 검증 계약

- strict time order와 group/episode blocking
- nested selection 또는 결과 전에 완전히 동결된 후보
- `raw_expected_points_delta - penalty >= 0.01`의 inclusive 판정
- 후보 family/tier와 active-share 규칙을 내부 결과 전에 등록
- 같은 family의 새 adverse residual은 penalty를 줄이지 않고 max로만 갱신
- 최소 3개 same-family pair 전에는 완화 검토 금지

모델 선택 편향을 막기 위해 평가와 선택을 분리해야 한다는 원칙은
[Cawley & Talbot 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html)에
따른다. train-to-target shift에서 보정이 representation과 weighting mechanism에
의존한다는 근거는 [Tibshirani et al. 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)와,
분포 이동 하 model-selection 프로토콜의 중요성을 보인
[DomainBed](https://openreview.net/pdf?id=lQdXeXDoWtI)에 맞춘다.

## QA

공식 pair 6개에서 family/tier penalty를 재계산했다. exact-family 우선순위,
unseen low-DOF fallback, compound global fallback, inclusive `+0.01`을 focused
pytest 4개로 검증했고 py_compile 및 Ruff도 통과했다.
