# Ocean AI Data Track 1

종합해양과학기지 관측 수온 자동 품질관리(P1) 문제를 위한 코드·노트북·실험 기록 저장소입니다.

## 중요

- 대회 원본 데이터는 재배포 금지이며 이 저장소에 포함하지 않습니다.
- 모든 작업 전에 [`00_MUST_READ_FIRST.md`](00_MUST_READ_FIRST.md)를 읽어야 합니다.
- 대회 제출은 하루 1회이므로, 정확한 제출 파일에 대한 사용자의 명시적 승인 없이 업로드하지 않습니다.

## 현재 자료

- `notebooks/00_p1_reconnaissance.ipynb`: 초기 데이터 품질·구조 정찰
- `reports/P1_RECONNAISSANCE_2026-08-13.md`: 정찰 결과와 모델링 위험
- `scripts/validate_submission.py`: hidden 정답 없이 제출 스키마·키·라벨·SHA-256 검증

## 로컬 환경

Python 3.12 기준입니다.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
