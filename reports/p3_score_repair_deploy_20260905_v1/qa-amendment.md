# 학습 전 비모델 QA amendment

원본 feature prepare는 2026-09-05 KST에 한 번 완료했다. 소요 697.8306537000171초, 24,360 anchors, 591 features, 181 station-episode/1,086 검증행, 공식 입력 0행이다. 실행 중 코드는 변경하지 않았다.

- 완료된 prepare runner SHA-256: `5e7a9c78a852259e3078270139e5156e5d7b28075202f2e99740cfbc5f99dc32`.
- 이후 train/predict runner SHA-256: `c4dd2c284dc2be5088b26e5e419f43dabdeda1506d3c092b9e8dd59b27c5d1e6`.
- config SHA-256 (변경 없음): `f8b64d3c2783a3bb11f8bda4b9100032610b08d86122b5d6067530e5e4e5a39f`.

변경은 모든 JSON read의 UTF-8 명시, CSV roundtrip을 공식 index에 맞춰 정렬된 frame 값과 비교, 총 6시간 계산에 fresh-process replay 시간 포함 및 이미 시간을 소진했으면 공식 입력을 읽기 전에 중단하는 QA 항목뿐이다. 모델, seed, feature, split, lead, 혼합비, 후보 선택 및 학습 횟수는 바꾸지 않았다. 원본 prepare receipt/lock/산출물은 보존했다.

amendment 후 screen + deploy focused pytest 14개 PASS, deploy Ruff PASS. 별도 새 train lock으로 고정된 8 CatBoost + 1 TabPFN backbone fit 및 3 router fit을 수행한다. prepare 재실행이나 과거 산출물/OOF 재사용은 없다.
