# 평가 데이터 지원 검사 — 학습 0회로 드러난 평가 제약

## 결론

배포 P1 train / P2 observations / P3 train_wave만 읽어 고정 v5 분할의 행·키·시각·지원 수를 검사했다. **147.839초, fit 0, 공식 입력/답안/업로드 0**, 세 원본 SHA 검사 전후 동일. 실행 상태는 `SOURCE_COUNT_AUDIT_COMPLETE_FEATURE_ADAPTER_AUDIT_PENDING`이다. 원본 데이터 오류나 모델 성능 검사로 부르지 않는다.

## 중요한 발견

| 대상 | 근거 | 영향·조치 |
|---|---|---|
| P1 H1_2025 | train 270,708 / validation 208,093. 그중 107,125행(51.48%)은 학습에 없던 station/layer. train은 S-ORS 1~7층뿐이고 validation에는 G/I 및 S8이 추가됨 | **High / 확인됨:** 이 fold는 계절 이전과 새 정점/층 이전이 섞인 검사다. 알려진/새 station-layer 성적을 별도로 내고, 전체 학습 후 모든 정점을 아는 배포와 동일 조건이라고 주장하지 않음 |
| P2 B4·B8 | 각각 validation 18,093/16,884행이 있지만 달력상 마지막17일 outage 대상행은 둘 다 0 | **High / 확인됨:** 두 구간의 outage RMSE/CI는 `NOT_ESTIMABLE_NO_ROWS`, PASS 또는 delta0 금지. 날짜를 결과에 맞춰 옮기지 않음. 빈 구간을 제외한 진단임을 명시하거나 이후 별도 source-only 계약 개정으로 검사 날짜 결정 |
| P2 B1·B3 | 온도 두 층은 있지만 유효 positive actual depth까지 요구하면 각각 33/34행 부족 | **Medium / 확인됨:** 실제 모델의 nominal-depth fallback을 적용할 수 있는지 adapter 검사 필요. 이 행을 몰래 삭제하지 않음 |
| P3 | 배포 wave 118,152행에서 dense eligible 24,360 anchors를 재구성. forward 다섯 fold 모두 train/validation 지원 있음 | **지원 수 확인:** 여섯 리드 target와 48h 시작 경계, 78h purge/정점 episode 제외 검사. 기상/파생 특징 지원 검사는 아직 별도이며 episode 수를 독립 폭풍 수로 부풀리지 않음 |

P2 자연 T5 결측 수는 temp만 기준이며 이전 실험의 temp **또는** psal 결측 trigger와 분모가 다르다. 특히 B3의 이번 180행을 과거 212행과 같은 지표로 비교하지 않는다. 온도 support는 공식 조건의 공개 가용성 proxy이며 숨은 QC를 확인한 것이 아니다.

## 고정 분할의 실측 수

| 문제/fold | train 행 또는 anchor | validation 행 또는 anchor |
|---|---:|---:|
| P1 H2_2024 | 86,987 | 172,638 |
| P1 H1_2025 | 270,708 | 208,093 |
| P1 H2_2025 | 444,081 | 287,862 |
| P2 B1 | 140,311 | 22,940 |
| P2 B2 | 134,207 | 26,018 |
| P2 B3 | 133,948 | 26,273 |
| P2 B4 | 145,162 | 18,093 |
| P2 B5 | 146,830 | 16,417 |
| P2 B6 | 134,168 | 26,308 |
| P2 B7 | 149,909 | 13,335 |
| P2 B8 | 149,384 | 16,884 |
| P3 Q2_2024 | 7,057 | 1,051 |
| P3 Q3_2024 | 7,912 | 2,628 |
| P3 Q4_2024 | 10,665 | 5,388 |
| P3 Q1_2025 | 15,974 | 6,708 |
| P3 Q2_2025 | 22,808 | 1,492 |

## 방법·검증 경계

- P1 grain=station/layer/time, KST. 양성 run 귀속과 purge는 고정 v5 helper를 실제 사용. 경계 이동행은 세 fold 모두0이다.
- P2 grain=station/layer/time, KST. target2/3/4를 public count에서 제외하고, finite target와 공개 finite temp≥2의 166,268행을 지원 proxy로 사용. 숨은 QC·공식 query는 미열람. T5를 제거한 뒤 남은 temp 수는 검사했지만 실제 모든 파생 특징 재계산을 실행한 것은 아니다.
- P3 grain=station/anchor_time, UTC. raw20분 hs≥1.5 연속 run이 episode, low/missing/gap에서 끊는다. 미래 target 결측으로 episode를 쪼개지 않는다. storm6h 병합 정의와 동일하지 않다. 같은 폭풍의 여러 run/정점 상관은 남는다.
- 합성 tests 4 PASS/Ruff PASS. 고정 evaluator의 기존45 tests 기록과 별개다. import 정렬 lint 한 건을 고친 뒤 관련 파일만 재검사했다.
- 검토 notebook은 nbformat 구조 검사 및 새 Python kernel에서 코드4셀 top-to-bottom 실행 PASS. 집계 slice 합계·빈 outage·6lead 분모·runner/config hash를 재대조했다. nbconvert 미설치로 nbclient를 사용했으며 패키지 설치는 하지 않았다. 이것은 원본 재스캔이나 새 학습이 아니다.
- source hash는 각 clean 재생성 영수증의 source hash와 일치했다. 접근은 이 runner의 명시적 세 파일 읽기 범위이며 OS 전체 감사 증명은 아니다.
- inspectable [실행 코드](../../scripts/audit_ocean_forward_support_20260906_v1.py), [합성 tests](../../tests/test_audit_ocean_forward_support_20260906_v1.py), [원본 집계 영수증](result.json), [검토 노트북](review.ipynb).

## 다음 행동

새 모델 평가 전 실제 특징 adapter의 mask/context 의존성을 검사하고 B의 동일 평가키 OOF를 생성한다. P1의 새 정점/층 slice, P2 빈 outage 및 depth fallback을 반드시 함께 출력한다. 기존 config와 실험 영수증을 수정하지 않는다. [다음 디자인](../../docs/ocean_v2_codex/NEXT_SCORE_DESIGNS_20260906.md)은 후보 가설이지 이미 성능이 오른 결과가 아니다.
