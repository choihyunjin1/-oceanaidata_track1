# 결론: 기존 clean 기준선 재현 성공, 작은 6h 탐색 후보까지 로컬 준비 완료

2026-09-05 실행에서 원본 배포 train 두 파일부터 feature를 재생성하고, 고정 8 CatBoost + 3 router + 1 TabPFN fitted-context 학습을 끝냈다. 기존 OOF나 제출 CSV를 런타임 입력으로 쓰지 않았다. 새 clean baseline OOF RMSE는 **0.7791048399763751m**로 과거 기준선과 정확히 같고, root의 독립 key/truth/prediction 대조도 exact PASS다.

6h에만 TabPFN을 25% 섞는 로컬 후보와 기준선 CSV가 각 1,200행으로 준비되었다. **공식 채점/업로드는 하지 않았다.** 과거 181 episodes의 탐색적 혼합 이득 −0.0007650883m를 새 CSV의 실측 성적으로 표시하지 않는다. 해당 이득의 95% bootstrap 구간은 0을 포함한다. 따라서 현재 결과는 '큰 돌파 입증'이 아니라 **적법한 기준선 재학습 경로 확보 및 작은 탐색 후보 보존**이다.

## 실제 검증

| 검증 | 실제 결과 | 해석 |
|---|---|---|
| 배포 train → 자체 feature | 24,360 anchors / 591 features, 697.831s | 외부자료·과거 cache 입력 0 |
| 동일 historical OOF | 181 distinct station-episodes / 1,086행 / 중복 0 | 새 독립 시험지는 아님 |
| 새 기준선 RMSE | 0.7791048399763751m, 과거 대비 drift 0.0m | 기존 성능 재현, 개선 아님 |
| 고정 baseline gate | persistence 대비 3/3 fold 개선, 전체 PASS | 새로운 6h 후보 gate 아님 |
| 학습 | CatBoost 8 + router 3 + TabPFN fitted context 1 | 추가 HPO·결과 기반 재시도 0 |
| 새 프로세스 재로딩 | 181 cases/1,086행, 두 정책 최대 오차 0.0m | 직렬화 재현성이지 일반화 성적 아님 |
| 로컬 CSV | 각 1,200행, 6h 200행만 차이 | 공식 채점 전 후보 |
| 전체 실측 | 1,678.413s = 27분 58.4초 | 현재 RTX 5090, CPU 2 threads |

실행 중 준비 runner는 변경하지 않았다. prepare 종료 후 root 검토에 따라 UTF-8/CSV index 순서 비교/replay 포함 runtime 계산만 수정했다. 모델·seed·특징·split·혼합비는 불변이다. [qa-amendment.md](qa-amendment.md)에 구/신 runner 해시와 테스트를 기록했다. focused pytest 14개와 Ruff PASS, [independent-qa.json](independent-qa.json)에 재계산 및 접근 범위를 기록했다.

## 적법성/재현 범위

배포 train_wave/train_atmos만 관측자료로 사용한다. 허용된 합성 데이터 전용 TabPFN-3 가중치는 정확한 로컬 checkpoint를 자체 model 디렉터리에 동봉하고 hash/license receipt를 검증했다. 예측 프로세스는 네트워크 연결을 차단하고 저장 state 및 동봉 가중치를 로컬에서 다시 로드했다. 실제 관측자료 사전학습 가중치, KMA/ERA5, Public-optimum alpha, hidden truth는 사용하지 않았다.

후보 생성 전에 별도 root 승인을 받았으며 익명 test_context 57,800행과 test_index 1,200행만 읽었다. sample/hidden/과거 제출 CSV 입력/업로드/Git stage/commit/push는 0이다. 후보 경로 및 전체 source→model→prediction 계약은 [README.md](README.md)를 따른다.

## 다음 판단

현재 단계에서 추가 fit이나 혼합비 탐색을 하지 않는다. 기준선은 보존하며 6h-only 후보는 작고 불확실한 recipe-level 근거를 가진 탐색 후보로 표시한다. 공식 점수 이득은 미확정이다. 실제 최종 제출 ZIP·독립 폴더 이동·OS 수준 인터넷 차단·운영진 하드웨어의 6시간 충족·최종 모델 잠금은 별도 검증 작업이며 이 실행의 완료 주장에 포함하지 않는다.
