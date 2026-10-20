# PR 1 기준 전체 재분석 결과

완료일은 2026-10-20, 분석기간은 2016-01-04~2026-10-14이며 거래량 준비기간은 2015-10-01부터다.
[비교 보고서](../../docs/FULL_RERUN_COMPARISON_20261020.md)를 먼저 확인한다.

- baseline: ce87af8 기존 코드를 동일 원자료에 적용한 통계.
- improved: PR 1 병합 0b91b5f 및 군집 분모 후속 수정 통계. 기업행위가 검증된 총수익률은 아니다.
- model_rerun: PR 1 당시 72개 특징 v1 및 고정 bag3 v2. 두 방향 각각 40분기 평가.
- model_default101: PR 1 당시 기본 101개 특징의 별도 평가. 10월20일 최신 모델을 뜻하지 않는다.
- key_return_comparison.csv: 진입·평가기간을 맞춘 평균/중앙값/표본 전후 비교.
- model_historical_comparison.csv: 과거 게시값 대비 변화. 원자료·실행환경 차이가 남아 PR의 인과효과로 볼 수 없다.
- model_same_data_code_comparison.csv: 같은 모델·입력에서 구/신 추론 및 평가 함수 일치.
- model_feature_set_comparison.csv: 같은 표본에서 72개와 101개 특징 성과 비교.
- cluster_outcome_fix_comparison.csv: 군집 유지·관측 분모 수정의 영향.

원자료, API키, 로컬 경로가 포함된 로그, 모델 바이너리는 GitHub에 포함하지 않았다. 전체 상세비교표와 학습모델은 별도 로컬 결과 묶음에 보존했다. 기존 outputs/tables를 덮어쓰지 않았다.
