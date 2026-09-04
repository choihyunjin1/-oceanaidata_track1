# P1 공격적 초기 정찰 보고서

작성일: 2026-08-13 (KST)

## 결론

원본은 ZIP과 압축 해제본이 바이트 단위로 일치하고, 핵심 키·시간·라벨 구조에 치명적인 품질 문제는 없다. 이 문제는 10분 간격 다중 정점·수심층 시계열에서 약 4.14%의 연속 이상 구간을 찾는 binary F1 문제다. 향후 성능의 핵심은 행 단위 무작위 분할이 아니라 시간/그룹 블록 검증, 이상 유형별 특징, 연속 구간 후처리이다.

## 소스와 무결성

- 원본 폴더: `P1_DATA_DIR`이 가리키는 로컬 전용 폴더
- ZIP SHA-256: `2a64050a22c1ea372d7ba826c87538a36d885367f3190ca1ec7414530c1bd042`
- ZIP CRC 검사: 통과
- ZIP 내 `train.csv`, `test.csv`, `sample_submission.csv`, `baseline_rule.csv`, `score.py`, `README.md`와 압축 해제본 SHA-256: 전부 일치
- 원본 README 제약: 재배포 금지

## 데이터 구조

| 파일 | 행 | 핵심 열 | 기간 |
|---|---:|---|---|
| train | 776,706 | 기본 7열 + `label`, `anomaly_type` | 2024-01-01 ~ 2025-12-10 |
| test | 169,011 | 기본 7열 | 2026-01-01 ~ 2026-06-30 |
| sample | 169,011 | 4개 키 + `label`, `anomaly_type` | test와 키/순서 일치 |
| baseline | 169,011 | 4개 키 + `label`, `anomaly_type` | test와 키/순서 일치 |

- train/test 키 중복: 0
- train/test 정확 중복 행: 0
- 시간 파싱 실패: 0
- 타임존 접미사: 전행 `+09:00`
- 정점: G-ORS, I-ORS, S-ORS
- train 23개 station-year-layer 구간, test 15개 station-layer 구간
- test의 station-layer 조합은 모두 train에 존재하며, I-ORS layer 3만 공식 예외로 test에 없다.

## 라벨 구조

- 정상: 744,580 (95.8638%)
- 이상: 32,126 (4.1362%)
- 양성의 `anomaly_type` 결측: 0
- 음성이면서 `anomaly_type`이 있는 행: 0

| 기본 유형 | 해당 문자열을 포함한 행 | 연속 구간 수 | 관측 행 기준 범위 |
|---|---:|---:|---:|
| spike | 104 | 104 | 1행 |
| noise | 9,656 | 52 | 23~353행 |
| flatline | 6,441 | 55 | 12~283행 |
| offset | 7,507 | 33 | 48~519행 |
| drift | 8,929 | 30 | 101~519행 |

`noise+drift`, `flatline+offset` 같은 겹침 표기가 존재하므로, 유형별 분석은 정확 일치가 아니라 `+`로 분해한 membership로 해야 한다.

## 결측과 관측 커버리지

| 데이터 | temp | psal | depth |
|---|---:|---:|---:|
| train 결측 | 0 | 16,725 (2.1533%) | 1,130 (0.1455%) |
| test 결측 | 0 | 798 (0.4722%) | 16,368 (9.6846%) |

- test depth 결측의 16,331건은 G-ORS 전체이며 공식 README와 일치한다.
- train의 큰 psal 결측 구간: I-ORS layer 7 16.84%, S-ORS layer 1 6.34%.
- 10분 정규 간격 비율: train 99.8443%, test 99.4592%. 나머지는 관측 공백이므로 구간 후처리에서 간격 단절을 존중해야 한다.

## baseline 정찰

- 양성: 4,402/169,011 = 2.6046%
- 정점·층별 예측률 범위: 1.1984% ~ 4.0667%
- 양성 연속 구간: 1,199개
- 1행 singleton: 699개 (58.3%)
- 최장 예측 구간: 232행 = 38.67시간
- README의 hidden test F1: 0.548255

기준선은 사용 가치가 높지만 단독 최종 모델로 보기 어렵다. 특히 실제 학습 이상은 장시간 구간이 많은데 baseline 예측은 singleton이 많아, 구간 확장·병합·최소지속 후처리의 개선 여지가 크다.

## 신호 정찰

- 이전 행과 temp가 정확히 같은 비율: 정상 0.2012%, 이상 19.4827%
- `abs(temp.diff())` 평균: 정상 0.2285, 이상 1.0035
- `abs(temp.diff())` 99백분위: 정상 2.8598, 이상 11.5614
- 순수 절대 temp보다 국소 차분·곡률·rolling 중심/산포·반복 길이·이웃 층 차이가 더 유망하다.
- test는 1~6월에 집중되어 train 전체 연간과 절대값 분포가 다르다. 절대 temp 임계값은 계절 이동에 취약하다.

## 위험과 후속 정찰 순서

1. 대회 전체 문제 설명에서 online/causal QC인지, 사후/bidirectional QC인지 확정
2. 2024→2025 및 정점/층 holdout으로 검증 프로토콜 비교
3. anomaly type별 탐지기: spike, flatline, noise, offset, drift
4. 국소 robust 특징과 이웃 층/정점 공통 신호 비교
5. 규칙 baseline 재현, supervised 모델, 유형별 앙상블의 OOF 비교
6. threshold·gap closing·minimum duration·boundary padding을 validation에서만 튜닝
7. 평균 F1 뿐 아니라 정점·층·시간·이상 유형별 최악 성능과 안정성 점검

## 환경·Git 상태

- 현재 PyCharm 프로젝트는 상속 SDK 설정만 있고 전용 인터프리터가 아직 없다.
- 정찰은 Codex 번들 Python 3.12.13, pandas 3.0.1, NumPy 2.3.5로 수행했다.
- 로컬 폴더는 정찰 시작 시 Git 저장소가 아니었다.
- 지정 GitHub 저장소는 2026-08-13 확인 시 public·empty 상태였다.
