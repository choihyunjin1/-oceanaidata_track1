# 결론과 실행 계약

P1 새 long-flank 표현은 내부 pooled F1에서 대조군보다 낮았고, 다음 binary Viterbi의 사전등록 inner on/off 전략도 낮았다. 두 arm 모두 닫고 재학습 가능한 clean O/B control만 새 배포 후보로 보존한다. 이는 역사적 공식 28.909341점 답안을 재생성하거나 그보다 높음을 보장하는 실행이 아니다.

- Root가 decoder의 421,032행 NPZ 및 수치를 독립 QA한 뒤 본 실행을 승인했다.
- 새 fulltrain: 운영진 train 776,706행, O/XGBoost 1-fit와 B/event-day LightGBM 1-fit. seed 20260813, CPU 4 threads, BLAS 1, CUDA 사용 없음, wall cap 1시간.
- 2025 Q4의 기존 inner 선택 B_union과 O high=0.2 / B high=0.3을 재사용. low ratio=0.5, gap=0, minimum run=12. 추가 캘리브레이션 0.
- Decoder OFF, fulltrain transition 추정 0. raw always-on의 사후 긍정 결과로 정책을 바꾸지 않는다.
- 모델 및 frozen_recipe 생성·해시 확정 뒤에만 별도 predict 과정에서 공식 test 169,011행과 sample의 key 열만 읽는다. sample 예측값/hidden/기존 답안 CSV/외부자료/선행 가중치 입력 0.
- 출력 CSV는 새 output/05_answer 안에 생성하고 별도 프로세스 verify로 저장 모델 재추론 byte equality를 검사한다. 업로드·commit·push는 0.
- 기존 station×year×layer train-depth map은 2026 키가 없어 missing/unknown으로 남는다. 내부 Q2-inner에도 cross-year 조건이 있었지만 Q4-inner와는 공변량 차이가 있다. 이 실행에서 사후 fallback을 넣지 않으며 공식 성능은 미확인으로 표시한다.
- 기존 final 패키지와 독립된 경로를 사용한다. 로컬 코드/모델/답안 패키지 준비와 공식 최종 ZIP 적격성/업로드 검증은 다른 단계다.
