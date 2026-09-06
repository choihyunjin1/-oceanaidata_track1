# 현재 실행 후속 처리 — 2026-09-06

## 병행 작업 — 12:52 KST 이후 별도 승인

사용자 추가학습 승인으로 [P1/P2/P3 새 코어 학습](../parallel_core_training_20260906_v1/plan.md)을 별도 담당자/경로에서 준비·실행한다. 이 heartbeat의 임무는 기존 b2f17 후보의 제출 후속 처리이며 새 실험을 중복 실행하거나 그 상태를 추측하지 않는다. 기존 MS worker41824 학습은 이미 종료됐다. 새 실험의 진행은 해당 새 ID의 receipt/process로 구분한다.

## 최신 체크포인트 — 10:56 KST, 학습/답안 완료·브라우저 차단

아래의 실행 중/후속 명령은 역사적 순서다. **학습, replay, materialization을 다시 실행하지 않는다.** 현재 상태는 [checkpoint-1056.md](checkpoint-1056.md)를 먼저 읽는다. MS 3-fit 학습, fresh-process replay, 독립 QA168개, 최종 CSV validator 및 별도 전체 추론 replay까지 PASS다. 새 후보 SHA는 `b2f17f5cda8030cb3d97fbb504e6babb6aef8ba7fe555092901479677af0625e`다. 아직 업로드/채점하지 않았다. 파일 선택 대기 timeout 후 Chrome 연결이 `Debugger unattached`를 반환하여 브라우저 단계만 막혔다. 일반 제출 버튼은 클릭하지 않았고 최종 모델 잠금도 하지 않았다.

선택 원칙: 특정 Q 악화만으로 자동 탈락시키지 않는다. 사전 고정한 전체 1차 지표 개선을 우선하고 Q별 악화는 위험으로 보고한다. 공식 점수는 실제 채점 뒤 별도로 비교하며, 내부 개선을 공식 점수로 단정하거나 점수를 역산해 계수/임계값을 바꾸지 않는다.

작업 위치 `C:\Users\cedis\PycharmProjects\PythonProject`; Python `.venv-p1/Scripts/python.exe`. **새 학습/재시작이 아니라 현재 실행을 완료하고 QA한 뒤 새 후보를 채점하는 작업**이다. AGENTS.md, 운영진 규정, P1 계약, 제출 runbook을 따른다. 기존 dirty worktree와 fallback을 보존한다. commit/push/최종 모델 잠금 없음.

후속 heartbeat ID: `p1-28-9`, 이 스레드에 10분 간격 ACTIVE로 등록됨. 기존 진행이 정상일 때는 알림 없이 기록하고, 완료/오류 처리 뒤 PAUSED로 전환한다. 등록 상태와 이후 실행 완료는 별개다.

## 이미 완료 — 재실행 금지

- `artifacts/p1_champion_reconstruction_20260906_v1/tree_historical`: 24fits/657.652s COMPLETE, replay36checks exact, independent QA171checks PASS.
- 같은 root의 `tree_full`: 4fits/166.543s COMPLETE, 새 PID probe replay2checks exact, independent QA31checks PASS. terminal SHA `c8d8f9e6671fd0caf5a325e90a28c4985442006cf3f04a44c3d6f3a2b7d26ad6`.
- `artifacts/p1_champion_reconstruction_20260906_v2_union_evaluation`: 기존 역사적 MS 검증 제안과 새 트리의 287862키 exact 결합, fit0, 독립 count384/bootstrap793 QA PASS. [후보 선택](candidate-selection.md), [결합 결과](union-report.md)를 읽는다. v1검증의119행 달력귀속 오류는 별도 실패 보존, v2가 canonical이다.
- 최종 트리 arm은 `union`으로 선택했으며 MS e150/3seed/.8 OR를 결합한다. 내부1차F1 약 .906966은 반복노출 retrospective이고 공식 점수가 아니다. Q4악화와 불확실성도 함께 보고한다.

## 현재 실행 — 절대 중복 시작하지 말 것

MS 실제 worker PID41824, 시작09:01:29.814319 KST. 진행/터미널:

`artifacts/p1_champion_reconstruction_20260906_v1/mstcn_full3/`

- `progress.json`: 최초 확인은09:24경 첫seed20260827 e90/150, 완료0/3. 이후 반드시 최신값을 읽는다.
- `training-result.json`: 존재하면 학습 종료; status TRAINING_COMPLETE_REPLAY_PENDING이어야 한다.
- `terminal.json`: 오류 또는 fresh replay의 최종 판정. train후 이 파일이 없다는 이유로 재학습하지 않는다.
- `REPLAY_ATTEMPT_LOCK.json`: 이미 있으면 replay도 중복 실행하지 않고 기존 process/terminal을 확인한다.
- `artifacts/p1_champion_reconstruction_20260906_v1_logs/mstcn-full3.stdout.log` / `.stderr.log`: 실행 중에는 끝부분/크기만 검사, 오류 때 read-only 조사한다.

고정 moduleSHA `d46b8a63d12459ef565edb03eb9bc8fe38d745624f192ca502c775a29743bbf8`, own snapshot `mstcn_full3/02_code/mstcn.py`. seed20260827/20260839/20260863, e150, width512, bf16, CPU2/GPU0 soleowner, 원cosine horizon300. 설정/학습강도/epoch/precision 변경0. 과거 모델/답안/patch를 입력으로 쓰지 않는다.

## 학습 종료 후 순서

1. 정상학습 종료 확인 후 새 PID에서 딱 한 번 실행:

```powershell
.venv-p1/Scripts/python.exe -I artifacts/p1_champion_reconstruction_20260906_v1/mstcn_full3/02_code/mstcn.py replay --output artifacts/p1_champion_reconstruction_20260906_v1/mstcn_full3
```

긴 프로세스는 PowerShell Start-Process -WindowStyle Hidden + 별도 stdout/stderr 파일로 실행한다. 오류 시 자동 반복하지 않는다. saved-model replay가 scratch학습 두 번의 결정론 증명은 아니다.

2. `scripts/p1_champion_reconstruction_20260906_v1/qa_mstcn.py`와 전용 테스트/lineage 문서가 완성되어 있는지 확인한다. 별도 독립 QA를 실행해 3fit/e150/165열/4050step/own source·model·encoder SHA와 runtime/access/replay를 대조한다. 없거나 QA에 문제가 있으면 단계 완료로 간주하지 않고 해결한다. 원 frozen module은 수정하지 않는다.

```powershell
.venv-p1/Scripts/python.exe -I scripts/p1_champion_reconstruction_20260906_v1/qa_mstcn.py artifacts/p1_champion_reconstruction_20260906_v1/mstcn_full3 --report artifacts/p1_champion_reconstruction_20260906_v1/mstcn_full3/independent-qa.json
```

3. **모든 QA 뒤에만** 새 post-QA decision JSON을 만든다. `materialize.py:decision`이 exact hash와 상태를 검사하므로 그 스키마를 직접 읽는다. evidence는 internal_qa, tree_historical, tree_training, tree_replay, tree_full_qa, mstcn_training, mstcn_replay, mstcn_qa, union_result, union_qa, union_bootstrap_qa이다. 독립 MS QA도 동일 train/replay SHA와 연결한다. earliest_started_unix는 **1788652849.7000048**이다. source→answer 공통6h 마감은 **15:00:49.700005 KST**이며 시계를 재설정하지 않는다.

4. 공식 materialization 허용 결정 후 `materialize.py`의 `tree`, `mstcn`, `combine`을 각각 새 Python -I CLI로 실행한다(같은 PID에서 혼합 import 금지). README에 명시된 P1_DATA_DIR 원본을 사용하고, 새로운 `candidate/stages/tree`, `candidate/stages/mstcn`, `candidate/05_answer` 경로만 사용한다. stage출력이 이미 있으면 기존receipt부터 확인하고 덮어쓰지 않는다. official169011행, sample은key만, hidden0. 모델 재학습0. train→inference→CSV전체계보와 source-to-answer6h를 검증한다.

5. 파일 schema/unique/exact keys/order/finite/SHA/validator 및 saved model replay가 PASS인지 확인한다. 사용자가 이미 일반 리더보드 업로드를 승인했다. computer-use skill과 실행서를 읽고 로그인 UI에서 현재기한·P1남은횟수·중복을 새로 확인해 **일반 답안 채점만** 한 번 실행한다. 마지막기억 P1 1회는 현재 사실로 가정하지 않는다. 최종 모델 잠금은 클릭하지 않는다. 점수/접수ID/CSV SHA를 연결한 영수증을 새 보고서에 남긴다.

6. 9031 fallback의 공식27.644124점과 새 점수를 비교해 더 좋은 적격본을 유지한다. 과거28.909341점의 자동승계금지. portable packaging은 아직 NOT_READY이며 [readiness](packaging-readiness.md)의 transitive imports/seal/경로/전체cold실행 공백을 구분한다. 실제 완료/실패/다음 행동을 한국어로 보고하고, 이 실행 후속 heartbeat는 PAUSED로 전환한다.

## 경계

진행 중이면 짧은 진행만 기록하며 장기 실행을 건드리지 않는다. 프로세스가 사라졌는데 정상 terminal이 없으면 로그·lock을 읽어 원인을 식별하고 기술 blocker를 보고한다. 외부관측·hidden truth·점수역산계수·수동행patch0. 현재 P2/P3 작업/후보는 변경하지 않는다. 입력·모델·CSV·log·cache·lock은 Git에 올리지 않는다.
