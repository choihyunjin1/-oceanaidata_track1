# Binary Viterbi decoder1개 — 결과 확인 전 사전계약

root 승인 후 `p1_score_repair_20260905_v1`의 강한 clean O/B control을 그대로 사용한다. backbone 재학습0, train-only 전이통계6회(inner/outer 각3), inner on/off 선택3회다. 공식 입력/CSV/upload0.

조건부 실행 근거는 현재 control의FN3261 중 일부 탐지 사건 내부누락1867(57.25%)이다. 이것은 누락행 패치 권한이 아니다. 새 표현 flank는 이미실패하여 재사용하지 않는다.

decoder는 유형별 기간/복합문법, fixed run extension, CAPA가 아니다. binary 정상/이상2state의 연속10분 전이를 학습split label에서 pooled 추정한다. 정점·층·실제gap·학습경계를 넘는 전이는 세지 않는다. 각 전이count에Laplace1을 더하고 행별정규화, 초기state비율도train only Laplace1을 사용한다. λ=1 고정, HPO없음.

unary는 기존 선택 control의 logit(p)−logit(inner high threshold), OR이면 구성unary의max다. 이는 **calibrated likelihood가 아닌 scoring heuristic**이다. Viterbi전역경로는 각exactgap segment별로 독립 계산한다. 기존control 양성인 plateau/confirmed singleton spike는 삭제하지 않되 모든 기존양성을OR로보존하지는 않는다.

inner에서 기존control보다F1이엄격히높을때만decoder on; 동률이면off. outer는 그on/off와기존threshold를바꾸지않고전이통계만해당outertrain에서추정한다. rawdecoder와선택결과를함께보관한다. outer점수를본후λ·기간·정점규칙을조절하지않는다. 실패하면현cleancontrol보존으로종료한다.

과거typed-duration semi-Markov는 약한typedunary .58104→.58348을기록했지만 spike재현율을훼손했다. 이번은typedunary/기간상한을이어쓰지않고현O/B모델과train추정switchcost·hardspike보존을사용하는구체적대조한개다. 같은family의실패가능성을숨기지않는다.
