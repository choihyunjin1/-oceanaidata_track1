# Claim-to-source ledger — P1 MS-TCN++/ASRF v1

| ID | 주장 | 근거 | 출처 종류 | 확신 | 남은 간극 |
|---|---|---|---|---|---|
| C1 | Round-B의 병목은 정밀도보다 긴 이벤트 재현율이다. | TP/FP/FN `12718/644/3337`; FN 중 3330행이 19행 이상 이벤트 | 로컬 frozen OOF | 높음 | 공식 hidden 분포가 같은지는 모름 |
| C2 | 기존 P1 TCN 실패는 장문맥 MS-TCN++ 공간을 닫지 못했다. | 과거 RF 약 29~31행, width 32~128, 짧은 optimizer budget | 로컬 config/artifact | 높음 | 오래된 Patch Transformer는 긴 창을 봤으므로 문맥만으로는 부족 |
| C3 | multi-stage dilated TCN은 초기 dense prediction을 반복 보정한다. | MS-TCN CVPR 2019, MS-TCN++ TPAMI 2020 | 1차 논문/공식 코드 | 높음 | 영상→센서 전이 미검증 |
| C4 | boundary regression을 분리하면 temporal segment 경계를 보정할 수 있다. | ASRF WACV 2021 | 1차 논문/공식 코드 | 높음 | P1 row-F1 개선량으로 환산 불가 |
| C5 | 최대 epoch를 늘리되 best checkpoint를 써야 한다. | P3 GRU/TCN은 2~4 epoch 최저 후 30 epoch 악화; P1 과거 model도 조기 최저 | 로컬 학습곡선 | 높음 | 새 모델의 최적 epoch는 미지 |
| C6 | P1이 P2/P3보다 첫 고용량 실험의 정보가치가 높다. | P1 identifiable FN headroom; P2/P3 현실적 예상 delta와 계산비용 비교 | 로컬 감사 + 문헌 종합 | 중간 | 문제별 공식 point surface가 완전히 선형이 아님 |
| C7 | local Q2 통과가 공식 +3점을 보장하지 않는다. | P1 official/local magnitude ratio와 P3 sign reversal | 공식 결과 기록 + 로컬 비교 | 높음 | 새 구조의 공식 전이는 제출 전 미지 |
| C8 | 로컬 `+0.066079`는 공식 `+3점`과 동치가 아니다. | 공식 목표 endpoint 0.930749와 로컬 기준 0.864670의 차이일 뿐이며 직접 metric delta는 +0.112876 | 산술/정의 감사 | 높음 | local→official transport 계수 미식별 |

## 1차 출처

1. Farha & Gall, MS-TCN, CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html
2. Li et al., MS-TCN++, TPAMI 2020: https://arxiv.org/abs/2006.09220 ; official code https://github.com/sj-li/MS-TCN2
3. Ishikawa et al., ASRF, WACV 2021: https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html ; official code https://github.com/yiskw713/asrf
4. Cerqueira et al., time-series performance estimation: https://arxiv.org/abs/1905.11744
5. CSDI, NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html
6. SSSD-S4, TMLR: https://openreview.net/forum?id=hHiIbk7ApW
7. PatchTST, ICLR 2023: https://openreview.net/forum?id=Jbdc0vTOcol
