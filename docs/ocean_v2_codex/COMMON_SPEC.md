# COMMON_SPEC — `src/ocean_v2/common/` (세 문제 공용)

## 목적
세 문제 파이프라인이 공유하는 경로 해석, 해시·매니페스트, 결정론 설정, 런타임 영수증, 제출 검증기, 상수 리터럴 감사, 부트스트랩 유틸을 제공한다. 문제별 패키지는 이 모듈만 의존하고 서로는 import하지 않는다.

## 파일과 API

### `paths.py`
- `resolve_data_dir(problem: Literal["p1","p2","p3"], cli_value: str|None) -> Path`: 우선순위 CLI 인자 > 환경변수(`P1_DATA_DIR` 등) > 오류. 필수 파일 집합(브리프 §1-1) 존재 확인, 없으면 명확한 예외.
- `run_dir(problem, candidate, base="artifacts/ocean_v2") -> Path`, `submission_dir(problem, candidate, base="submissions/claude_v2") -> Path` (생성 포함).

### `hashing.py`
- `sha256_file(path) -> str`(1 MiB 블록), `sha256_bytes`, `input_manifest(data_dir, filenames) -> dict[name -> {bytes, sha256}]`, `stable_hash(obj) -> str`(json.dumps sort_keys 후 sha256; 특징 캐시 키용).

### `determinism.py`
- `set_global_seed(seed: int)`: `random`, `numpy` 전역 seed; torch가 import 가능하면 `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(n)`.
- `lgbm_params(base: dict, seed: int, threads: int) -> dict`: `deterministic=True, force_row_wise=True, num_threads=threads, seed=seed, bagging_seed=seed+1, feature_fraction_seed=seed+2, data_random_seed=seed+3, verbose=-1` 병합.
- `xgb_params(base, seed, threads)`: `tree_method="hist", device="cpu", nthread=threads, random_state=seed`.
- `cat_params(base, seed, threads)`: `task_type="CPU", thread_count=threads, random_seed=seed, allow_writing_files=False, verbose=False`.
- `stable_sort_index(df, keys) -> np.ndarray`: `np.lexsort` 기반 결정론 정렬 인덱스.

### `runtime.py`
- `Receipt` 컨텍스트 매니저: 시작/종료(KST ISO), 초 단위 소요시간, `platform`, 라이브러리 버전(numpy/pandas/lightgbm/xgboost/catboost/torch), 스레드 수, 입력 manifest를 `TRAINING_RECEIPT.json`으로 기록. `@timed("stage")` 데코레이터로 단계별 시간 누적.

### `submission.py` (배포 `score.py`의 입력 검증을 정답 없이 재현)
- `validate_p1(csv_path, sample_submission_path) -> dict`: 열 순서 `station,year,layer,time,label`(+선택 `anomaly_type`), 행 수 169,011, 키 결측·중복 0, 키·순서가 sample과 동일, label이 유한 정수 0/1. 양성 수·계열별 양성률 요약 반환.
- `validate_p2(csv_path, sample_submission_path)`: 열 `station,layer,time,temp`, 26,061행, 키 동일, temp 유한·−5~45. 층별 통계 반환.
- `validate_p3(csv_path, sample_submission_path)`: 열 `case_id,station,lead_h,hs_pred`, 1,200행, lead_h∈{3,6,9,12,18,24}, 키 동일, 0~30 유한. 리드별·정점별 통계, persistence 대비 |Δ| 분위 반환.
- `write_submission(df, path)`: `index=False, encoding="utf-8", lineterminator="\n"`, 문제별 `float_format`(P2 `%.5f`, P3 `%.4f`, P1 정수) 후 `sha256.txt` 동시 기록.

### `audit_constants.py`
- CLI: `python -m ocean_v2.common.audit_constants --package ocean_v2.p3 --allow configs/ocean_v2/p3_base.json`.
- `ast`로 지정 패키지의 `predict.py`, `decode.py`/`postprocess.py`, `features.py`, `ensemble.py`를 순회해 숫자 리터럴(정수 0/1/2, 배열 인덱스, 초·분 환산 상수 제외)을 파일:줄과 함께 나열. 설정 JSON의 `physical_constants` 값 집합과 대조해 목록에 없는 실수 리터럴이 있으면 종료 코드 1. 결과를 `audit_constants.json`으로 저장(패키지 README에 첨부).

### `stats.py`
- `block_bootstrap_delta(err_a, err_b, group_ids, n_boot, seed) -> {delta_mean, ci90, p_improve}`: 그룹(일/episode) 재표집으로 두 후보의 지표 차이 분포. RMSE용(제곱오차 평균 → sqrt)과 F1용(TP/FP/FN 합산) 두 변형.
- `rmse(a, b, w=None)`, `f1_counts(y, p) -> (tp, fp, fn, f1)`.

### `report.py`
- `write_report(json_path, md_path, sections: list[(title, DataFrame|dict|str)])`: JSON과 마크다운 표 동시 저장.

## 테스트 `tests/ocean_v2/test_common.py`
- 각 validator가 잘못된 열 순서·행 수·범위·중복 키를 거부하는지(합성 DataFrame으로).
- `lgbm_params`가 결정론 키를 항상 포함하는지.
- `audit_constants`가 리터럴 `0.834`를 포함한 임시 모듈에서 실패하고 allowlist 값은 통과하는지.
- `block_bootstrap_delta`가 seed 고정 시 재현되는지.
