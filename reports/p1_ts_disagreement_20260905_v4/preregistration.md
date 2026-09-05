# P1 T–S 변화 불일치 — 단일 특징 블록 대조

사용자 승인 후 2026-09-05 작성. 공식 역산/외부자료 계보는 제외한다. 제공 문서의 T–S 가설만 채택하며 그 문서의 예상 점수·관측에서 유도한 임계값을 복사하지 않는다.

배포 train.csv → 원래 clean v1의 earlier-inner/outer 분할과 통계 → 기존 80열에 염분 변화 대비 수온 jump/roughness, local correlation/residual을 추가한 B 모델. 비교군은 기존 정확한 SHA의 O/B 모델과 동일 inner-selected 정책이다. 수심 계약과 디코더는 변경하지 않는다. median salinity-step epsilon은 train 관측만으로 적합하며 평가 label/anomaly_type은 특징 함수에 전달하지 않는다.

윈도6/36/144행은 10분 cadence의1/6/24시간 설계값이다. exact segment 내부의 양방향 공개 관측만 사용하며 새 특징의 최대24시간보다 기존21일 purge가 길다. 기존 기저 특징의 의존성을 축소했다고 주장하지 않는다. 3fold × inner/outer × B 1모델 =6학습, CPU3threads, 40분 점검 예산이다. 기저 O/XGB는 새로 학습하지 않고 정확한 모델/의존성 hash를 검증한다.

선택은 이전 inner에서 동일 threshold grid 및4정책으로 수행한다. 주평가는 Q2/Q3/Q4 pooled F1이며 fragmented 스트레스와 같은 retained-row intact 비교, 정점/유형/시기 진단도 남긴다. 평균 개선이면 INTERNAL_CANDIDATE로 보존하고 seed/구간 위험을 별도 표기한다. 악화하면 이 지문을 닫고 창 길이/epsilon을 사후 튜닝하지 않는다. 이 면은 이미 노출된 historical development이지 fresh confirmation이 아니다.

합성 계약검사 후 학습하고, 새로운 독립 산술 QA 및 저장모델 별도프로세스 재생을 수행한다. 공식 입력/CSV/upload/Git0. fulltrain은 이번6fit 비교에 포함하지 않으므로 내부 개선만으로 제출 준비 완료라고 부르지 않는다.
