# P1 28.909341점 원본 경로 복원 — 저장 모델 추론 exact PASS

2026-09-06 19:12 KST. **원본 XGBoost O + 원본 LightGBM B 3-seed + 원본 MS-TCN 3-seed를 실제 추론하여, 과거 28.909341점 제출본과 전체 CSV SHA256이 같은 출력을 만들었다.** 과거 답안 CSV와 고정 행 패치 JSON을 예측 입력으로 읽지 않았다. 원형 O/B 셀 조합 및 GI 일반 규칙을 코드로 연결했다. **모델이 소실됐거나 저장 모델 추론조차 불가능한 상태가 아니었다.**

이번 검사는 saved-weight replay다. 캐시 없는 특징 재생성, 빈 모델 폴더부터 전체 재학습, 다른 머신/차단망 최종 검증의 PASS로 확대 해석하지 않는다. 예측 모델 신규 학습 0, 새 CSV 저장 0, 업로드 0, commit/push 0. 최종 CSV 직렬화는 메모리에서 수행했다. 기존 답안은 별도 byte hash 확인만 했으며 추론 입력이나 행별 정답으로 사용하지 않았다.

## 실측 결과

| 단계 | 행 / 양성 | 이번 재생 SHA256 | 당시 기록과 일치 |
|---|---:|---|---|
| XGBoost O | 169,011 / 6,504 | `28243fda9bc56e25a698366823dfab3198cda21bfaec04f30fda6a899eaf0cd3` | PASS |
| LightGBM B 3-seed | 169,011 / 5,856 | `decedb8a9b3df7d955ae9b3848cd8f985c5228e6727accbb514516507755adbf` | PASS |
| 원형 셀 router | 169,011 / 6,061 | `1b04e81c18d5a5cac3115c3a256e8d5a38a9493a32478a184df81fd99f9f6e5f` | PASS |
| router ∪ MS-TCN e150 | 169,011 / 6,394 | `a52dc49c5f522ae92eb67805a8a567dc04d62791725a5de622f544af7a3ce33b` | PASS |
| 위 결과 + GI 일반 spike 규칙 | 169,011 / 6,396 | `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687` | PASS |

- 트리 별도 실행: **13.890초**. 전체 7모델 재생: **37.995초**, 모두 정상 종료.
- 전체 재생에서 트리도 다시 추론했으며 두 실행의 O/B/router SHA가 모두 일치했다. MS 전체 재생은 이번 감사에서 1회이며 반복 학습 결정론 검사가 아니다.
- schema/169,011행/전체 key와 순서/중복/유한 binary label validator PASS.
- 합성 pytest **8 PASS**, 해당 신규 코드 4개 및 tests Ruff PASS. 별도 PID 해시·영수증·OOF 혼동행렬 재계산 **49-check PASS**.
- 핵심 영수증: [full-replay-result.json](full-replay-result.json), [independent-qa.json](independent-qa.json), [saved-replay-result.json](saved-replay-result.json), [pytest.xml](pytest.xml).
- 원래 점수는 기존 `artifacts/official_final_submission_20260905/P1/contract.json`의 제출 기록이다. 이번에 새 업로드/재채점한 수치가 아니다. 실제 보존 답안 `artifacts/official_final_submission_20260905/P1/05_answer/P1_submission.csv`도 위 `57844e…` SHA다.

## 왜 최근에는 복원되지 않았다고 설명했는가

**정확한 복원과 새 설계를 혼동했다.** 최근 `p1_champion_reconstruction_20260906_v1`은 자체 문서에서도 원본 exact 복원이 아니라고 명시한다. 현재 clean 특징(학습 구간 기반 year-depth·spike scale, 168h plateau cap)을 쓰고, 임계값·셀 선택을 새 inner 구간에서 적합했으며, 최종 트리 arm은 원래 router가 아니라 전역 O/B 합집합이었다. 이 변경들의 합계가 다른 모델이다. 새 후보 27.311774점은 원래 28.909341점 모델이 재현 불가능하다는 증거가 아니다. 각 변경이 점수 하락에 기여한 몫을 이번 검사에서 분해한 것은 아니다.

**당시 최종 패키지의 연결도 불완전했다.** 원래 패키지는 MS 실제 가중치를 추론하지만 tree router는 CSV로 공급하고 GI는 고정 두 행 JSON으로 공급했다. TRAIN 진입점은 MS만 학습하고 기본 출력 위치도 PREDICT 기본 가중치 경로와 달랐다. 따라서 '저장 패키지의 답안 replay'와 '빈 03_model에서 모든 구성 요소 재학습'은 서로 다른 검사였다.

**원본 모델 위치를 잘못 좁혀 보았다.** 실제 제출 B는 8월 25일 17:58 fallback 모델이다. 후속 22시 full-deployment-fit 모델과 별도 계보이며, 최신 복원 계획은 후자의 레시피를 참조했다. 실제 제출 B의 저장 모델을 찾아 연결하자 SHA가 정확히 돌아왔다. 후자의 모델이 반드시 다른 답안을 낸다는 비교까지 한 것은 아니다.

## 고정 답안 의존성을 원형 연산으로 바꾼 근거

1. **router**: B를 기본으로, G-ORS/L1·I-ORS/L2의 O-only를 추가하고 S-ORS/L1·L5·L6·I-ORS/L4의 B-only를 제거한다. 8월 26일 업로드 전 동결 소스 및 로컬 OOF 비교 기록이 존재한다. 따라서 이 셀을 곧바로 '리더보드 역산 상수'라고 단정할 근거는 없다.
2. **GI**: 위 추가는 유지하되 B-only 제거를 하지 않은 분기다. 원형 `build_deadline_probe_set_20260828.py`는 `(현재 결과가 0) & (GI가 1) & (GI 유형이 spike)`라는 일반 규칙으로 후보를 만들었다. 원 코드의 숫자 2는 당시 발생 건수/검사값이지 대상 행 key 목록이 아니다.
3. 이번 코드는 0/1/2/7개 spike 등 합성 경우를 통과했고, 실제 원본 모델 출력에서 자연스럽게 **2행**이 추가되어 최종 SHA가 일치했다. `gi_spike2_patch.json`은 읽지 않았다.
4. router의 **원래 셀 선택 알고리즘 전체**는 아직 발견하지 못했다. 원형 셀과 OOF 증거를 회고적으로 복원했을 뿐, '새 OOF에서 셀을 재적합하는 코드까지 복구'라고 부르지 않는다. 같은 OOF에서 선택·평가한 낙관 편향도 남는다. 규정 적격 판정과 운영진 승인은 별도다.

## 과거 내부검증 대조

원래 TRAIN OOF 421,032행의 key/fold 순서가 일치하며, router를 다시 계산한 결과 저장 anchor와 **불일치 0행**이다.

| 모델 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| B | 12,718 | 644 | 3,337 | 0.864670089 |
| 원형 router | 12,756 | 618 | 3,299 | 0.866899997 |
| 원형 O/B 전역 합집합 | 13,094 | 1,230 | 2,961 | 0.862042859 |

router − B = +0.002229908. 원형에서 합집합은 B보다 낮았다. 최신 변경 모델의 합집합 개선과 서로 모순이 아니며, 서로 다른 모델·선택 절차다. 이번 수치는 이미 사용한 개발 OOF의 회고적 재계산으로 독립 holdout 성능이나 새 공식 점수가 아니다. [result.json](result.json)의 최초 O receipt `NOT_YET_LINKED` 항목은 이후 발견된 원본 학습 receipt와 [independent-qa.json](independent-qa.json)의 PASS로 보완한다. 초기 영수증은 덮어쓰지 않았다.

## 입증 수준 / 다음 작업

| 항목 | 상태 |
|---|---|
| 원본 O/B 가중치 존재 및 당시 receipt SHA | PASS |
| 원형 router의 과거 OOF bit 재생 | PASS |
| 원본 7모델 + 일반 규칙 → 원 답안 전체 SHA | PASS |
| 과거 답안 CSV·고정 행 패치 없는 추론 | PASS |
| 원시 배포 데이터 → 캐시 없는 동일 특징 생성 | 이번 검사 미실행 |
| 빈 모델 폴더 → 전체 원형 재학습 → 동일 답안 | 미실행 |
| 원형 셀 선택의 실행 가능한 전체 학습 절차 | 미복구 |
| 새 clean-room 최종 제출 패키지 | 미완료 |

다음은 **원형 설정을 유지한 채** 특징 생성 → O/B 학습 → MS 학습의 연결을 독립 폴더에 구성하고 검증하는 작업이다. 셀 선택 출처/재적합 처리도 명시해야 한다. 그 전에 모델 구조·threshold·router를 바꿔 '복원'이라고 부르지 않는다. 기존 28.9점 파일과 모델, 최근 b2f17 후보, 기존 실패 attempt는 모두 보존했다.

별도 최근 `P1_champion_portable` cold 학습 11fits는 끝났지만 마지막 MS QA가 소유 폴더 밖 모듈을 읽어 기술 실패한 상태다. 이는 **다른 b2f17 후보의 패키지 문제**이며 이번 원본 28.9 saved replay PASS와 섞지 않는다. 해당 실행은 재시작하지 않았다.

## 실행 경계 및 재실행 안내

`scripts/p1_historical_full_replay_20260906_v1.py --root <repo> --data <P1 배포 폴더> --output <새 JSON 경로>`가 전체 저장 모델 진단 진입점이다. `--check-only`는 원본 pin만 검사한다. 원본 파일을 변경하지 않으며 결과 JSON 경로가 존재하면 거부한다. 캐시·가중치를 포함한 원본 로컬 자산이 필요하므로 아직 portable/source-only 진입점이 아니다. root 독점 GPU 배정 후 실행한다.

공식 test 관측 169,011행 및 sample의 schema/dummy 행은 승인된 재생에서 읽었다. TRAIN 776,706행과 train-derived cache로 MS encoder를 1회 적합했다. **예측 모델 fit0과 전처리 encoder fit1을 구분한다.** hidden truth·외부 관측·네트워크·업로드를 사용하는 경로는 없다. 계수나 라벨을 공식 점수로 역산하지 않았다. 0 접근 필드는 명시적 코드 경로/영수증 대조이며 OS 전역 접근 감사를 했다는 주장은 아니다.

진단 어댑터 개발 중 첫 실행은 잘못된 `apply_postprocess` import, 두 번째는 특징-only cache에 year key를 요구해 중단했다. 두 경우 모델 추론/학습 이전이었다. import와 cache 검사를 원형 저장 형식에 맞게 고친 뒤 성공했으며, 모델/threshold/특징 값을 결과에 맞춰 조정한 재시도가 아니다. 초기 Ruff 위반도 신규 진단 코드에서만 수정했다.

근거 추적은 [claim-source-ledger.md](claim-source-ledger.md) 참조. `ocean-experiment` 절차에 따라 계보·내부 지표·저장 모델 replay·scratch/최종 적격성을 분리했다.
