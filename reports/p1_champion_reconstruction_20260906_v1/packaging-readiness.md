# P1 champion reconstruction — packaging readiness

**현재 판정: 새 MS-TCN 모델을 사용하는 추론 경로는 정적으로 연결돼 있지만, 단일 portable 최종 제출 패키지는 아직 완성·검증되지 않았다.** 이 문서는 2026-09-06 09:21 KST 기준 소스/설정/집계·메타데이터만 검토한 결과다. 진행 중인 MS-TCN 학습 PID 41824, 원본/복사 소스, 모델, 설정, lock은 수정하지 않았다. 추가 학습·공식 데이터 읽기·CSV 생성·업로드·Git 쓰기는 0이다.

이 검토는 [제출 실행서](../../docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md)의 “빈 모델 폴더에서 배포 데이터로 학습 → 모델 → 답안을 네트워크 없이 6시간 이내 재생성”과 저장 모델 추론만의 재생 검증을 구분한다. 배포 README는 P1 train 776,706행, test 169,011행, 제출 열 `station,year,layer,time,label`, 정수 0/1, KST 시각을 규정한다. 배포 자료는 재배포 금지이므로 ZIP에는 데이터 참조/연결 안내만 넣는다.

## 1. materialize.py 읽기 전용 검토

검토한 파일과 SHA-256:

- [materialize.py](../../scripts/p1_champion_reconstruction_20260906_v1/materialize.py): `890636e706de2fce69f7ab54ec1e13636e43dd404b98a2964e7a423e8bd8f91d`
- [composition.py](../../scripts/p1_champion_reconstruction_20260906_v1/composition.py): `42fcfe13da0f4fc10c05c25b4706e5f3d418cd202b781f60e30b9f3fd70010e0`
- 봉인 [mstcn.py](../../scripts/p1_champion_reconstruction_20260906_v1/mstcn.py): `d46b8a63d12459ef565edb03eb9bc8fe38d745624f192ca502c775a29743bbf8`

### 확인한 연결

1. `decision()`은 root의 사후 materialization 승인, tree arm, 내부 QA, tree historical/full/replay, MS training/replay의 상태·SHA·교차 실행 계보를 먼저 확인한다. 다른 실행의 replay 또는 내부 QA를 연결하면 실패한다.
2. `mstcn_inference()`는 `mstcn_model_output/training-result.json`을 decision SHA와 대조하고 `verify_owned()`를 통과한 뒤 공식 관측을 읽는다. `predict_proposal(..., model_dir=output/03_model)`이 새 모델 폴더를 명시적으로 받는다. 옛 `weights`나 `retrained_from_scratch` 경로로 fallback하지 않는다.
3. `verify_owned()`는 자체 `02_code`, encoder, 세 state, validation inventory와 **실행 중 mstcn.py 자체 SHA**까지 검사한다. `predict_proposal()`은 저장된 encoder를 재사용하고, 모든 source/model/replay hash를 receipt에 연결한다. 기존 anchor CSV·patch·답안을 읽지 않는다.
4. MS-TCN은 새 copied source에서 import한다. tree stage를 먼저 같은 Python 프로세스에 import하면 `p1_qc`가 외부 경로에서 이미 로드되어 fail-closed될 수 있다. **tree / mstcn / combine을 각각 별도 CLI·새 PID로 실행**해야 한다. `materialize.py` 자체가 세 프로세스를 자동 spawn해 주는 것은 아니다.
5. tree stage는 새 full O1/B3 네 모델과 새 `preprocess.joblib`, `selector.json`을 읽는다. station/layer/time 정렬 후 학습 encoder로 추론하고 `_order`로 입력 순서를 복구한다. tree 선택 arm의 bits와 MS proposal은 composition에서 동일 키 집합 여부를 확인해 재정렬한 뒤 OR한다. 행 교집합 자동 축소나 정점별 수동 override는 없다.
6. 최신 materialize는 test/sample의 읽기 **전후** SHA를 확인한다. sample은 키 열만 파싱한다. combine은 169,011행/schema/key/order/binary를 저장 전후 검증하고 LF 줄바꿈으로 답안을 쓴다. hidden truth를 여는 경로는 없다.

이상은 코드 경로 검토이며 실제 새 모델 로드, 공식 추론, 최종 CSV 재생 성공을 뜻하지 않는다. root의 최종 decision에는 어떤 완성 정책을 왜 선택했는지 및 별도 new-union 내부 QA 근거도 설명해야 한다. 이 구현의 `internal_qa` 필수 필드는 tree historical 결과에 연결되는 기술/산술 QA이며, 그 자체가 새 union의 승격 판단 전부를 자동 표현하지 않는다.

## 2. 현재 코드 그대로 실행할 때 필요한 파일

### MS-TCN 자체 실행 트리

`<MS_RUN>`에는 다음이 필요하다. 단순히 세 `.pt`만 복사하면 현재의 무결성 검사를 통과하지 못한다.

- `03_model/seed_20260827.pt`, `seed_20260839.pt`, `seed_20260863.pt`, `encoder.json`.
- `02_code/mstcn.py` 및 그 안의 `PINS`에 선언된 정확한 9개 원형 소스/설정 파일. 이 목록에는 original MS runner, model/data API, feature/config/data/package init, original experiment JSON, `configs/p1.toml`이 들어간다.
- `04_validation/own_train_probe.npz`, 각 seed의 `probe_expected_<seed>.npz`, `fit_<seed>.json` 등 `training-result.json.files_sha256`에 열거된 **모든** 파일.
- `training-result.json`, `fresh-process-replay.json`. 실제 운영 이력으로 `ATTEMPT_LOCK.json`, `REPLAY_ATTEMPT_LOCK.json`, `terminal.json`, `prepare.json`도 보존하되, 이 lock을 새 cold 실행 입력으로 복사해서 재사용하지 않는다.

probe는 이번 학습에서 생성된 입력 특징/예측이며 hidden 값은 아니다. 그러나 연구 artifact로 취급하고 Git에 넣지 않는다. 최종 제출을 모델-only로 경량화하려면 원 runner를 수정하는 것이 아니라 **별도 새 hash의 saved-inference adapter**가 필요하다. 그 adapter는 full model/encoder/source/recipe만을 검증하고 같은 계산을 해야 하며, 실제 ZIP 새 추출·별도 PID 재생을 검증해야 한다. 아직 구현된 상태가 아니다.

### Tree 및 합성 실행 트리

- `tree.py`, `tree-contract.json`, `qa_tree.py`, `composition.py`, `materialize.py`와 고정된 선택 decision.
- `tree_full/full/models/O_20260813.joblib`, `B_20260813.joblib`, `B_20260829.joblib`, `B_20260847.joblib`, `preprocess.joblib`.
- `tree_full/selector.json`, `terminal_result.json`, `fresh-replay-qa.json`, full probe 등 full terminal의 `files` inventory 전부. 최종 추론은 검증용 파일의 hash도 검사한다.
- `tree-seal.json`과 decision이 직접 가리키는 tree historical/full/QA/replay/MS receipt. 현재 decision 경로는 로컬 실행별 경로이므로 외부 폴더에서는 **새 실행이 생성한 자체 경로/해시**를 연결하는 orchestration이 필요하다. 과거 절대 경로를 조용히 원 repo로 fallback하면 안 된다.
- 현재 `tree.source_hashes()`가 요구하는 `src/p1_qc/*.py` **106개 전체**, 아래 5개 공통 파일 및 tree 자체 계약/코드. 파일 목록 자체를 equality 비교하므로 일부만 복사하면 원 seal 검증에 실패한다.
  - `configs/p1.toml`
  - `configs/p1_meaningful_learning_curve_generation_v1.json`
  - `scripts/ocean_evaluation_contract_v5.py`
  - `scripts/run_p1_meaningful_learning_curve_generation_v1.py`
  - `scripts/run_p1_score_repair_20260905_v1.py`
- 위 외에 아래 transitive import 두 파일도 필요하다. 현재 tree source seal에 없으므로 추가 provenance를 별도로 기록한다. **진행 중인 seal을 수정하거나 과거 검증 결과를 새 seal로 바꾸지 않는다.**

### Supplementary transitive-source provenance

| 현재 파일 | 현재 SHA-256 | 이번 경로의 역할 |
|---|---|---|
| `src/ocean_goal/meaningful_score.py` | `69b9dc1168a47e0d1b1a50e5590c3d2f0966f2885f0f92730d4c38c4ea92800c` | legacy helper 모듈의 top-level import dependency |
| `src/ocean_goal/__init__.py` | `a7a4f8f969d1425d5573d7ea7ff0f3253f9c17f19a7b377ee70f31334b013bf7` | package import/re-export |

호출 위치와 helper 본문을 확인했다. `tree.py:175`는 `core._lgb_parameters`, `tree.py:226`은 `core._event_day_weight`를 호출한다. `run_p1_score_repair_20260905_v1.py`가 이 둘을 `run_p1_meaningful_learning_curve_generation_v1.py`에서 import하며, 그 모듈이 다시 meaningful_score를 import한다.

- `_event_day_weight` (`...meaningful_learning_curve...py:407`)는 전달받은 **학습 metadata/target**의 연속 양성 run과 정상 일별 개수로 가중치를 계산하고 원 행 순서를 복구한다. 파일/네트워크/Public score 읽기나 meaningful_score 함수 호출이 없다.
- `_lgb_parameters` (`:469`)는 전달된 LGBM parameter dict에 objective/seed/deterministic/row-wise 설정을 붙이는 순수 변환이다. tree wrapper가 CPU threads를 4로 명시한다. Public score나 외부 관측을 조회하지 않는다.
- 그 옛 runner의 `load_contract(...)`는 별도의 historical CLI 실행 함수 안(`:1149`)에 있으며 이번 pure-helper 호출 경로에서 호출하지 않는다. imported 모듈의 존재와 외부 데이터 사용을 혼동하면 안 된다. 반대로 import-only라도 파일이 없으면 원 repo 밖 import가 실패하므로 패키지에는 필요하다.

`materialize.combine`은 `p1_qc.submission.validate_submission`을 import하고, 이는 `p1_qc.experiment.sha256_file`을 import한다. 현재 넓은 106개 snapshot에는 들어 있지만 최소 adapter를 만들 때는 이 전이 의존성까지 확인해야 한다. 넓은 snapshot에는 이번에 실행하지 않는 외부 관련 **소스 코드**도 존재한다. 이는 외부 데이터 학습 증거가 아니지만, 최종 최소 제출 패키지는 실행 의존성만 별도 봉인하는 편이 명료하다.

## 3. 명령 계약 — 아직 자동 단일 패키지가 아님

아래는 현재의 실제 엔트리와 순서다. 예시 `<...>`는 새 실행 경로이며 그대로 실행할 명령이 아니다. 학습 중인 기존 run에는 다시 실행하지 않는다.

```text
# 공통: P1_DATA_DIR=<배포 P1 디렉터리>, Python/라이브러리와 소스 경로 고정
python tree.py preflight --seal <new-tree-seal.json>
python tree.py historical --seal <new-tree-seal.json> --output <new-tree-historical>
python qa_tree.py <new-tree-historical> --seal <new-tree-seal.json>
python tree.py replay --seal <new-tree-seal.json> --output <new-tree-historical>
# full은 해당 역사 실행/독립 QA에 연결된 root 승인 후
python tree.py full --seal <new-tree-seal.json> --output <new-tree-full> --historical <new-tree-historical>
python tree.py replay --seal <new-tree-seal.json> --output <new-tree-full>

# MS는 별도 새 프로세스/독점 GPU, 최초 시작부터 6h clock 공유
python mstcn.py train --source-root <code-root> --train-csv <P1_DATA_DIR>/train.csv --output <new-MS_RUN> --training-approved --gpu-approved
python <new-MS_RUN>/02_code/mstcn.py replay --output <new-MS_RUN>

# QA 후 새 decision을 생성한 다음 각각 별도 PID로 실행
python materialize.py tree --decision <new-decision.json> --data-dir <P1_DATA_DIR> --output <stages>/tree
python materialize.py mstcn --decision <new-decision.json> --data-dir <P1_DATA_DIR> --output <stages>/mstcn
python materialize.py combine --decision <new-decision.json> --data-dir <P1_DATA_DIR> --stage-dir <stages> --output <new-answer-dir>
```

현재 그대로의 cold 경로는 tree historical 24 + tree full 4 + MS full 3 = **31 candidate fits**다. MS의 과거 Q3/Q4 모델 재학습을 여기에 몰래 추가하지 않는다. 과거 고정 e150 proposal 재검증은 이번 선택의 research evidence이고, 배포 학습 입력이 아니다. 최종 portable 패키지에서는 frozen 완료 정책과 원형 학습 절차를 명시하고 기존 OOF/답안을 필수 학습 입력으로 만들지 않아야 한다. fit 수를 줄이는 다른 최종 학습 경로가 필요하면 별도 계약/검증이지 이번 봉인 코드의 조용한 변경이 아니다.

새 단일 `RUN_TRAINING`/`RUN_INFERENCE` wrapper, packaging README, dependency lock, ZIP builder는 현재 이 새 reconstruction 폴더에 없다. 외부 독립 폴더에서도 `02_code/scripts/...`, `02_code/src/...`, `02_code/configs/...`의 상대 구조를 보존해야 기존 `ROOT=...parents[...]` 계산이 로컬 code-root를 가리킨다. 전체 repo의 개인 경로나 Git checkout을 암묵적으로 import하는 상태를 portable이라고 부르면 안 된다.

권장 사용자 구조는 `01_data`(참조만) / `02_code` / `03_model` / `04_logs` / `05_answer` / `06_docs`다. 다만 원 MS runner의 내부 `02_code`/`03_model`/`04_validation` 구조와 tree `full/models` 구조는 새 wrapper가 명시적으로 매핑해야 하며 현재 모델을 움직이거나 덮어쓰는 작업은 하지 않는다.

## 4. 환경과 시간 측정 범위

현재 `.venv-p1` package **metadata만** 조회한 버전: Python 3.12.10; numpy 2.3.5; pandas 3.0.1; scipy 1.18.0; scikit-learn 1.9.0; XGBoost 3.4.0; LightGBM 4.7.0; torch 2.13.0+cu130; joblib 1.5.3; pyarrow 25.0.1; psutil 7.2.2. 이것은 install 가능한 최소 lock/오프라인 wheelhouse 검증이 아니다. tree seal은 numpy/pandas/XGBoost/LightGBM/joblib 버전과 thread environment를 검사하며, MS 결과/replay는 실제 torch/CUDA/GPU/backend 설정을 남긴다. CUDA bf16 미지원 시 fp32/CPU로 자동 대체하지 않는다.

- tree historical 상한 5,400초, 전체 source→answer 상한 21,600초. tree CPU4와 MS CPU2/독점 GPU를 지정된 범위에서 병행하므로 시간은 fit 시간의 단순 합이 아니라 **최초 시작부터 최종 답안 검증 완료까지의 벽시계**다.
- 현재 가장 이른 실제 시작은 tree 2026-09-06 **09:00:49.700005 KST**, MS는 09:01:29.814319 KST다. 현재 실행 전체 deadline은 **15:00:49.700005 KST**다. 원 실행 clock을 새 receipt 생성 시각으로 바꾸면 안 된다.
- 과거 MS 3-seed full fit 실측은 합계 6,584.2288초(109.74분)였지만 fresh feature 준비/QA/최종 추론이 포함되지 않았으므로 이번 완료 시각을 보장하지 않는다. 이번 전체 측정은 아직 끝나지 않았다.
- `materialize.decision`은 tree/MS actual start의 최솟값을 확인하고, 각 stage timer와 저장 후 검사가 같은 최초 6h에 묶인다. 최종 CSV를 별도 PID에서 다시 생성하여 SHA를 비교한다면 그 단계까지 포함해 완료 시간을 보고해야 한다.
- `tree.replay` 자체에는 독립 hard timeout이 없고, 최종 CSV exact replay용 전용 엔트리도 아직 없다. 향후 단일 wrapper는 재생과 QA도 같은 전체 deadline 하에서 감시해야 한다. 이를 위해 현재 실행의 봉인 모델·시계를 바꾸지 않는다.
- 장기 saved-model inference는 이미 종료된 cold run의 오래된 deadline을 reset해서 실행하는 기능이 아니다. `materialize.py`는 그 clock이 6h를 넘으면 의도적으로 거부한다. 후일 재생용에는 별도 saved-inference adapter/짧은 새 실행 clock을 두고 **새로운 6h cold 학습 증거와 구별**해야 한다.

## 5. 아직 충족되지 않은 완료 조건

| 항목 | 이 문서 시점의 상태 / 필요한 증거 |
|---|---|
| 새 MS full3 완료 | 진행 중; 모두 e150/4,050 steps/nonfinite0과 모델 hash가 필요 |
| MS 새 PID 원형 bf16 replay | 대기; 전 seed row/boundary/type exact 및 환경 일치 필요 |
| 새 tree full4와 새 PID replay | 해당 terminal/QA 수집 후 판단; 이 검토는 모델을 열거나 재실행하지 않음 |
| 새 tree + MS 완성 정책 내부 QA | root 소유; 과거 union F1을 승계하지 말고 동일 키 비교·population 차이를 공개 |
| 공식 materialization | 아직 후속 root 결정; 모델/QA 연계 후 원형 그대로 생성 |
| 새 PID 최종 CSV exact replay | 아직 미실측; schema/key/order/finite/binary/hash와 입력 전후 SHA 연결 필요 |
| 전체 6h | 진행 중, 완료시간 미확정; source preparation/fit/QA/inference/answer replay 모두 포함 |
| 빈 폴더 portable cold 실행 | 미실행; 원 repo 밖 새 code-root/빈 model-output에서 실행해 증명 필요 |
| offline/clean environment | 미검증; Python network deny는 실제 OS 차단망 검증과 다름 |
| packaging README/자동 명령/ZIP | 미생성; 학습 코드 ZIP과 저장 모델 재생 ZIP 역할을 구분하고 실제 추출 명령 검증 필요 |
| 최종 모델 잠금·공식 제출 | 이 검토 범위 밖; 로컬 준비나 GitHub push가 공식 최종 제출을 뜻하지 않음 |

따라서 이 단계에서 주장할 수 있는 것은 **새 모델을 명시 로드하도록 설계된 연구 실행/추론 인터페이스와 packaging gap이 확인되었다**는 것이다. 현재 패키지를 “최고점 완전 복원”, “빈 폴더 재학습 exact PASS”, “최종 제출 준비 완료”로 표시해서는 안 된다.
