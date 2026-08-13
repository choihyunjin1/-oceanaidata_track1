# 반드시 먼저 읽을 메모 — Ocean AI Data Track 1

> 이 파일은 **모든 작업·실험·제출 전에 매번 전체를 다시 읽는다.**
> 사용자가 최신 문제 배경·과제·규정을 제공하면 이 메모를 즉시 갱신한다.

## 1. 작업 대상과 원본

- 로컬 프로젝트: `C:\Users\cedis\PycharmProjects\PythonProject`
- P1 원본: `C:\Users\cedis\PycharmProjects\PythonProject\데이터셋 원본\데이터셋_P1\P1_qc_anomaly`
- 필수 원문: `...\P1_qc_anomaly\README.md`
- GitHub 백업: `https://github.com/choihyunjin1/-oceanaidata_track1`
- 원본 README에 **재배포 금지**가 명시되어 있다. ZIP/CSV/원본 README/채점 자료를 공개 GitHub에 올리지 않는다.
- 원본은 수정·덮어쓰기·이동하지 않는다. 파생 데이터는 별도 무시 폴더에 저장한다.

## 2. P1 문제 — 종합해양과학기지 관측 수온 자동 품질관리

- 기본 특징: `station, year, layer, time, temp, psal, depth`
- 학습 정답: `label, anomaly_type`
- 키: `station, year, layer, time`
- 시간: 전부 KST `+09:00`; `year`도 KST 달력연도
- 학습: 776,706행, 2024~2025년
- 테스트: 169,011행, 2026-01-01~2026-06-30
- 평가: 행 단위 binary F1
- 공식 규칙 기반 기준점: `0.548255`

### 반드시 지킬 제출 형식

- 허용 열 순서는 정확히 다음 둘 중 하나이다.
  - `station,year,layer,time,label`
  - `station,year,layer,time,label,anomaly_type`
- `label`은 결측/무한값 없는 정수 `0` 또는 `1` 만 허용된다.
- 169,011행이어야 하며, 키 중복이 없고 test와 키 집합이 완전히 같아야 한다.
- `anomaly_type`은 선택 열이다. 정확한 유형을 예측하지 않으면 열 자체를 제외하는 편이 안전하다.

## 3. 공식 예외·운영 조건

- P2 정답 보호: S-ORS 2025-09-01~10-31, layer 2·3·4는 P1 train에서 제외되었다.
- G-ORS는 단층이고 2026년 `depth`가 전부 결측이다.
- I-ORS 2026 layer 3은 유효 관측이 없어 test에 없다.
- 정점·연도·층별 센서 운영기간과 관측 수가 다르다. 일괄 보간이나 전체 구간 정상 가정을 하지 않는다.
- 이상 구간은 10분 단위 연속 구조다: spike 10분, noise 3~58.8시간, flatline 2~47.2시간, offset 8~86.5시간, drift 9~86.5시간.

## 4. 2026-08-13 정찰에서 확인한 사실

- ZIP CRC 검사가 통과했고, ZIP 내 6개 파일과 압축 해제본의 SHA-256이 모두 일치한다.
- train/test의 키 결측·중복과 정확 중복 행은 0건이다. 모든 시간은 파싱되며 `+09:00`이다.
- train 양성은 32,126/776,706 = 4.1362%로 희소하다.
- test 결측: `psal` 798건(0.4722%), `depth` 16,368건(9.6846%). depth 결측의 16,331건은 G-ORS 전체이다.
- baseline은 4,402/169,011 = 2.6046%를 양성으로 예측한다. 양성 연속 구간 1,199개 중 699개가 1행 singleton이다.
- sample/baseline의 키와 행 순서는 test와 완전히 같다.
- 자세한 근거: `reports/P1_RECONNAISSANCE_2026-08-13.md`
- 재현 노트북: `notebooks/00_p1_reconnaissance.ipynb`

## 5. 모델링·검증 금지선

- **무작위 행 분할을 주 검증으로 사용하지 않는다.** 10분 연속 이상 구간과 시계열 자기상관 때문에 인접 시점 누출이 발생한다.
- 기본은 연도/시간 블록, 정점·층 그룹, 이상 구간 경계를 보존한 validation이다.
- 미래 값을 쓰는 centered rolling feature는 online QC 요구와 충돌할 수 있으므로, 문제의 실시간/사후 처리 허용 범위를 확인하기 전에 사용하지 않는다.
- train의 `anomaly_type`은 보조 학습/오류 분석용일 뿐 test 특징이 아니다.
- test 예측 수·임계값을 공식 train 양성률에 인위적으로 맞추지 않는다. validation의 정밀도/재현율/F1과 구간 후처리로 결정한다.

## 6. 하루 1회 제출 게이트

사용자가 **정확한 파일을 명시하여 최종 승인**하기 전에는 어떤 제출도 업로드하지 않는다. 승인 전 반드시 다음을 기록한다.

1. 실험 ID, Git commit, seed, 피처/모델 버전
2. validation 분할 정의와 전체·정점·층·이상유형별 precision/recall/F1
3. baseline `0.548255`와의 비교 및 과적합/누출 위험
4. 제출 스키마·행 수·키·binary label 검증 통과
5. 제출 CSV의 절대경로, 파일명, 바이트, SHA-256
6. 해당 날의 기존 제출 여부와 제출 기회 초기화 기준 시간대

검증 명령: `.venv\Scripts\python.exe scripts\validate_submission.py <candidate.csv>`

## 7. 아직 공식 확인이 필요한 사항

- 대회 주최·주관 기관의 정확한 명칭
- 외부 데이터/사전학습 모델/앙상블 허용 범위
- 미래 시점·양방향 rolling 특징을 포함한 사후 자동 QC 허용 여부
- 제출 1일 기준 시간대와 취소/재제출 규정
- 공식 리더보드의 public/private 분리와 최종 평가 절차
