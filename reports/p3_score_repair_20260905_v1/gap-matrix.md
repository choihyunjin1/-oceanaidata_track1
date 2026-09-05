# P3 evidence gap

| 질문 | 확인된 근거 | 남은 gap / 행동 |
|---|---|---|
| 새 blockmask가 자체 control을 이기는가 | 12fits, ΔRMSE +0.00888860m | 아니오. 같은 구성 재시도/HPO하지 않음 |
| 약한 baseline만 이긴 결과를 승격했는가 | 강한 clean .77910484까지 동일키 비교, 두 arm/blend 모두 악화 | 신규 LGBM 승격 없음 |
| 6h TabPFN25에 의미 있는 전체 개선이 있는가 | −.00076509m,95%CI [−.00261383,+.00116266],H1/G/I악화 | 작고 사후인 후보로만 유지; 공식 개선 미확인 |
| 기상 파생 결측이 실제 원시가림과 일치하는가 | 합성289행에서 전체1,275요약 일치,330compactweather열 | 구현검증 PASS; 성능개선과 별개 |
| 기존 외부/공식점수 역산 계수가 섞였는가 | source/cache/helper/OOF SHA와 recipe 확인; refined/KMA사용0 | 새deploy는 source-only OOF까지 재생성 |
| 준비된 최종 모델인가 | 후속 source-only8CatBoost+3router+6hTab1fit/별도replay오차0/로컬CSV2개완료 | 로컬준비완료. 운영진 최종ZIP·격리환경 검증은 별개 |
| 6시간 offline 재현 조건을 통과했는가 | 새 prepare/train/replay/predict1,678.413초,weights동봉·local reload·README완료 | RTX5090현재머신한정. OS격리/운영진하드웨어/최종ZIP미검증 |
| 공식입력/업로드를 소비했는가 | 이 screen0. 후속별도승인 context57,800/index1,200행 | hidden/sample/과거CSV입력/업로드0. 공식채점미확인 |
