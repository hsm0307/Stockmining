# 실전 예측 기록

`python -m krxlimit daily` (또는 `predict`)가 이 폴더에 모델별 예측을 저장합니다.
장 시작 전에 커밋해 두면 커밋 시각이 사전 예측의 증거가 됩니다.

- `{모델명}/YYYYMMDD.csv` : 기준일 장 마감 후 계산한 다음 거래일 상위 20개 후보
- `{모델명}/full/YYYYMMDD.csv.gz` : 전 종목 예측 확률 (채점용)
- `{모델명}/scoreboard.csv` : 일별 채점표 (P@5/10/20, AUC, AP)
- `{모델명}/actual_events_rank.csv` : 실제 발생 종목이 예측 순위 몇 위였는지
- `{모델명}/meta.json` : 사용한 모델 정보 (타깃, 학습 기간, 특징 목록)
