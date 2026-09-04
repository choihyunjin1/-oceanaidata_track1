# Commit inclusion manifest

## 포함

- P1/P2/P3 사전등록 config
- P1/P2/P3 one-shot runner와 독립 verifier/QA
- focused unit tests
- 문제별 aggregate JSON과 Markdown 보고서
- 본 병렬 사이클 통합 보고서와 QA

## 제외

- 원본 P1/P2/P3 데이터와 README
- 공식 test/sample/submission 파일과 값
- prediction CSV와 submission CSV
- model state와 checkpoint
- P1 replay history, sealed NPZ, attempt lock
- P3 blind prediction parquet, seal, attempt lock
- cache, temporary files, Python bytecode

제외 산출물은 `.gitignore`가 적용되는 local artifact namespace에만 존재한다.
