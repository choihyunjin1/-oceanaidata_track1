# P1 별도 양측 split 지원 점검 — 0fit

Root의 2026-09-06 추가 지시에 따른 **별도 지원 점검**이다. 완료된
`p1_bracket_forward_20260906_v1` 및 frozen `ocean_forward_v5`를 수정하거나 재채점하지 않는다.

- 검증행은 기존 세 반기 H2_2024/H1_2025/H2_2025의 양성 run 시작 귀속 + 달력 음성행과 정확히 같다.
- 새 학습 후보행은 배포 train의 나머지 모든 시점 중 `[fold_start−21d, fold_end+21d)` 밖이다.
  양성 run이 이 제외 구간에 걸치면 전체를 학습에서 제외한다. 정점/층이 미지원이어도 검증행을 삭제하지 않는다.
- 읽기 열은 배포 train의 station/year/layer/time/label뿐이다. 원 관측 특징·모델·OOF·공식 파일을 열지 않는다.
- fit 0, 성능 지표 계산 0, CSV/upload 0. 보고 내용은 학습/검증/지원행 개수, 기존 forward와 검증 key hash 일치뿐이다.
- 이 점검은 미래 시점의 **배포 train label**을 학습 후보로 허용하는 retrospective 보간형 평가의 지원도이다.
  실제 미래 배포 예측 검증이나 fresh confirmation이 아니며, 기존 점수 상승을 의미하지 않는다.
- 후속 학습 전 Root가 별도 prospective 계약에서 inner selection, partition 격리와 feature/decoder sentinel을 정해야 한다.
  현재 모델/threshold를 보며 inner 날짜를 조정하거나 자동 학습하지 않는다.
