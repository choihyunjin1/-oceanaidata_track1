# P1 v32g 공식 제출 결과

## 결론

`P1_V32G_E150_CATBOOST_PRECISION_UNION`은 Public F1 `0.832905`, `28.892255점`으로 채점됐다. 기존 champion `0.833548 / 28.909341점` 대비 F1 `-0.000643`, 점수 `-0.017086점`이므로 기존 champion을 유지한다.

후보는 champion의 6,396개 양성을 모두 보존하고 CatBoost 고신뢰 23개만 추가한 비중복 probe였다. 파일·키·분포·해시·배포 데이터 전용 scratch 정책 QA는 통과했지만, 제출 전 내부 pooled `ΔF1 -0.000173`과 Q3/Q4 `ΔF1 -0.000239`가 이미 NO_GO 방향이었다. 공식 결과도 같은 방향의 소폭 회귀를 확인했으므로 이 23개 추가 집합은 폐기한다.

제출 CSV는 169,011행, SHA-256 `2f6b115dd706d42b8851685af4c2e4a065f111e2b51b8dee9bc6170396e5e279`다. 대회 플랫폼 업로드는 정확히 1회였고, 외부자료·pretrained weights·인터넷 자료·KIOST 원자료·hidden truth 접근은 0이다.
