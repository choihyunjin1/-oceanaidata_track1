# P3 원본 재학습 → 저장 모델 → 로컬 후보 준비

현재 상태는 **원본 재생성·9 backbone fit·저장 모델 fresh-process replay·로컬 후보 2개 생성/QA 완료**이다. 이 디렉터리는 새로운 로컬 후보 준비 기록이며, 운영진 최종 제출 ZIP 또는 격리 환경 재현 번들의 검증 완료를 뜻하지 않는다. 업로드·최종 모델 선택·Git commit/push는 하지 않았다.

완료 파일(모두 저장소 root 상대경로):

- 기준선: `artifacts/p3_score_repair_deploy_20260905_v1/candidates/clean_baseline.csv`, SHA-256 `6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7`.
- 6h 탐색 후보: `artifacts/p3_score_repair_deploy_20260905_v1/candidates/tabpfn25_6h_only.csv`, SHA-256 `a7c7b247e5d74e7a0b6c8be42a7d4298a220b73881acb99490fcf7fe85f82f29`.

각 200 cases/1,200행, schema/key/order/중복/finite/range QA PASS다. 두 파일은 6h의 200행만 다르며 나머지 1,000행은 정확히 같다. 공식 채점 결과는 아직 없다. 공식 입력은 별도 root 승인 후에만 context 57,800행 및 index 1,200행을 읽었으며, hidden/sample/과거 제출 CSV 입력·업로드는 0이다.

## 후보와 근거의 한계

- 기준선: 배포된 P3 학습 자료만으로 학습한 single-target CatBoost와 multi-target CatBoost의 동등 혼합, 과거 완료 fold만 사용하는 loss router, 12/18/24시간 persistence shrink 0.2.
- 탐색 후보: 위 기준선의 **6시간 예측만** 75% 기준선 + 25% TabPFN-3로 혼합한다. 다른 5개 lead는 기준선과 같다. 이 정책은 이번 full training 전에 고정했다.
- 과거 동일 181개 station-episode/1,086행에서 기준선 RMSE 0.7791048399763751m, 후보 0.7783397516664796m이었다. 차이 −0.0007650883098955m는 작고 10,000회 episode bootstrap 95% 구간이 0을 포함한다. 이미 관찰한 사례에서 고른 탐색적 후보이며 새 독립 확인 결과가 아니다.
- 새로 만든 OOF는 기존 OOF/CSV를 읽어서 복원하지 않는다. 따라서 재학습·GPU 비결정성 등에 따른 RMSE drift를 위 작은 차이와 별도로 보고한다. 새 후보의 공식 점수 향상을 단정하지 않는다.
- **과거 recipe-level 증거**, **새 clean baseline OOF의 재학습 drift**, **저장 모델 fresh-process replay 일치**는 서로 다른 질문이다. 새 full-trained CSV에 과거 `0.77834m`를 그 파일의 직접 검증 성적으로 붙이지 않는다. full-training 후 로컬 case replay는 저장/재로딩의 재현성 검사일 뿐 일반화 검증이 아니다. 새 full-trained 6h TabPFN 후보 자체의 fresh OOF 확인은 이번 승인 범위에서 추가 fit하지 않는다.
- KMA/ERA5/외부 관측 및 실제 관측자료로 사전학습한 모델, Public 점수로 맞춘 alpha는 사용하지 않는다.

## 입력/학습/모델/예측 분리

| 단계 | 입력 | 출력 | 재학습 |
|---|---|---|---|
| `--prepare` | 배포 `train_wave.csv`, `train_atmos.csv` | 자체 train feature/anchor, 181개 로컬 검증 case | 0 fit |
| `--train --gpu-approved` | 자체 prepare 산출물, 학습 원본, 로컬 합성 사전학습 가중치 | CatBoost·router·TabPFN fitted state 및 모델 해시 | CatBoost 8 + router 3 + TabPFN 1 |
| `--predict-replay --gpu-approved` | 저장된 모델, 자체 로컬 replay case | fresh-process 예측 일치 QA | 0 fit |
| `--predict-official --gpu-approved --official-approved` | 저장 모델과 **별도 승인 후에만** 익명 test context/index | 기준선·6h 후보 로컬 CSV 각 1개 | 0 fit |

원본 학습 데이터는 `P3_DATA_DIR`의 두 배포 CSV만 사용한다. 공식 예측 단계 전에는 test context/index/sample/hidden/제출 CSV를 읽거나 만들지 않는다. 공식 예측에도 sample, baseline CSV, hidden truth, 기존 제출 파일은 필요하지 않다. 모든 실행에는 네트워크 연결을 차단하는 Python audit hook 및 Hugging Face offline 모드를 적용한다. 이는 별도 OS 격리 환경 검증을 대신하지 않는다.

### 실행 위치와 필수 코드

현재 검증 작업의 저장소 root는 `C:\Users\cedis\PycharmProjects\PythonProject`이다. runner는 실행 직후 CWD를 이 root로 변경한다. 별도 위치에 옮길 때는 다음과 같은 **저장소 상대경로 구조**를 유지해야 한다.

- `scripts/run_p3_score_repair_deploy_20260905_v1.py`
- `scripts/run_p3_corrected_repeated_forward_catboost_v1.py` — 일부 고정 fit/router 함수만 재사용하며 이전 main/공식 추론 엔트리포인트는 호출하지 않는다.
- `configs/experiments/p3_score_repair_deploy_20260905_v1.json`
- `configs/experiments/p3_corrected_repeated_forward_catboost_v2.json`
- `src/p3_wave/`, `src/ocean_tabpfn3/` 및 해당 모듈의 설치 의존성.
- `artifacts/p3_score_repair_deploy_20260905_v1/feature_columns.json` 및 prepare/training/replay manifest.
- `artifacts/p3_score_repair_deploy_20260905_v1/models/full/single.cbm`, `multi.cbm`, `router.joblib`, `tabpfn6.tabpfn_fit`, `license-receipt.json`, `weights/tabpfn-v3-regressor-v3_default.ckpt` (학습 성공 후 생성).

TabPFN fitted state는 foundation checkpoint를 내장하지 않는다. 저장 state의 `model_path`가 저장소 root 상대경로이므로 **별도 `weights/` 파일을 반드시 함께 두어야 한다**. 현재 fresh-process replay는 이 상대경로 계약을 실제 검증한다. 추론에는 과거 OOF/기존 제출 CSV가 필요하지 않다. 처음부터 재학습할 때는 원본 배포 학습 데이터와 합성 가중치·사용자 license receipt도 필요하다.

## 합성 사전학습 가중치 예외

- 모델: TabPFN-3 regressor, 로컬 Python package `tabpfn==8.5.0`.
- 사전학습 데이터 성격: 프로젝트에서 검토한 공식 모델 카드 및 provenance contract 기준 **합성 데이터만으로 사전학습된 테이블 모델**이다. 실제 기상·해양·관측자료 사전학습 모델은 허용하지 않는다.
- checkpoint: `tabpfn-v3-regressor-v3_default.ckpt`, SHA-256 `311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301`.
- provenance: `configs/compliance/tabpfn3_offline_transition_20260901.json`; 사용자 라이선스 수락·대회 이용 및 synthetic-only 검토 receipt를 확인하고 credential 없는 receipt만 자체 model 폴더로 복사한다.
- 가중치 로컬 동봉/로컬 load, README 명시, 라이선스 확인, 6시간 실측을 각각 구분해 검증한다. 학습·replay 완료 전에는 4조건 전체 충족으로 표시하지 않는다.

## 시간과 준비도

실제 CatBoost는 배포 데이터로 8개 모델의 트리를 새로 학습했다. full single 146,160행 학습은 230.669초, full multi 24,360 anchors 학습은 24.607초였다. TabPFN은 gradient fine-tuning 방식이 아니라 **in-context 학습**이다. 배포 데이터 24,360개 6h 학습 사례의 전처리·fitted context를 새로 만들었고(`fit_preprocessors`, 1.744초), 실제 예측 및 fitted state 저장까지는 59.871초가 걸렸다. 합성 foundation 가중치를 새로 훈련했다는 뜻도, 과거 관측자료로 완성된 fitted state를 그대로 로드했다는 뜻도 아니다.

학습 단계 전체는 927.571초, 원본 prepare까지 합쳐 1,625.401초였다. 별도 fresh replay 24.814초와 공식 예측 28.197초를 포함한 총 실측은 **1,678.413초(27분 58.4초)**다. fresh-process reload의 최대 예측 오차는 기준선/후보 모두 0.0m였다. 새로 재생성한 clean baseline OOF RMSE는 0.7791048399763751m로 과거 기준선과 집계 drift가 0.0m였다. Root의 독립 대조에서도 과거와 새로운 OOF key/truth/prediction이 모두 정확히 일치했다. 이는 새로운 성능 개선이 아니라 기준선 재현 확인이다.

재현 시간은 source prepare + 9 backbone fits/3 router fits 및 학습 중 replay/save + **별도 fresh-process replay** + 공식 예측을 모두 합산한다. 현재 머신은 RTX 5090이며 P3 CPU 2 threads로 실행한다. 대회 재현 하드웨어는 확정되지 않았으므로 현재 머신의 6시간 이내 실측이 대회 환경의 6시간 보장을 의미하지 않는다.

각 단계는 exclusive lock으로 중복 실행을 막는다. 기술 실패 시 기존 lock/산출물을 지우거나 자동 재시작하지 않는다. 이 디렉터리의 최신 JSON receipt가 실제 완료 단계와 시간을 기록한다. 최종 제출 ZIP, requirements/wheelhouse, 독립 폴더 이동 테스트, 네트워크 차단 OS 환경, 운영진 제출 UI의 최종 잠금은 별도 작업이며 여기서 완료했다고 주장하지 않는다.
