# P1 v28m1 독립 업로드 판정

## 결론

`P1_submission.csv`는 행·키·허용 스키마·해시·배포 데이터 전용 scratch 계보 검사는 통과하지만, **업로드하면 안 된다**. 169,011행 전부가 같은 양성 라벨인 상수 분류기로 붕괴했다.

원인은 공식 구간의 label-shift EM 목표 유병률이 `0.999999`까지 포화된 뒤, 기존 anchor 밖 162,615행을 모두 추가한 데 있다. 그 결과 음성 예측은 0행이고 일별 추가 비율 최댓값도 `1.0`이다. 이는 내부에서 이미 실패했던 일별 집중도와 station-layer 안정성 gate보다 더 강한 실제 배포 붕괴 신호다. 내부 예상 `+0.227점`은 이 공식 materialization 분포를 정당화하지 못한다.

대회 채점기는 `station,year,layer,time,label`의 5열 최소 스키마를 허용하므로 형식 오류는 아니다. 그러나 형식 통과와 제출 가치는 별개다. v28m1은 `BLOCK_UPLOAD_DEGENERATE_ALL_POSITIVE`로 보존하고, 다음 후보는 별도 사전등록 namespace에서 EM transport 상한 또는 label-free prevalence guard를 둔 뒤 비상수 분포를 pre-upload gate로 강제해야 한다.

외부자료·인터넷 자료·KIOST 원자료·hidden truth·실관측 pretrained weight 접근과 업로드는 모두 0이다. 예측 행 값은 출력하지 않았다.
