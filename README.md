# Seoul LH Jeonse Risk Checker

서울시 주택 전세 보증금의 적정 수준을 예측하고, 입력 매물이 주변 시세 대비 과도하게 높거나 낮은지 확인하는 간단한 웹 애플리케이션입니다.

최종 실행 화면은 사용자가 지역, 계약월, 주택 유형, 면적, 방 개수, 보증금을 입력하면 모델이 추정한 적정 보증금 범위와 위험도를 보여줍니다.

## 실행 방법

```bash
python src/frontend/app.py
```

실행 후 터미널에 표시되는 로컬 주소로 접속하면 됩니다. 기본적으로 `res/final_model` 아래의 저장된 모델 아티팩트를 사용합니다.

## 설치

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```text
src/
  data/       데이터 수집, 정제, 전처리, 피처 생성 코드
  models/     모델 학습, 튜닝, 예측 코드
  utils/      모델링 공통 유틸리티
  frontend/   로컬 웹 앱 실행 코드
res/
  final_model/ 최종 CatBoost 모델과 메타데이터
data/
  modeling/   모델 학습용 데이터셋
```

## 모델 설명

최종 모델은 Optuna로 튜닝한 CatBoost 회귀 모델입니다. 모델은 주택의 지역, 계약월, 주택 유형, 면적, 방 개수, 주변 전월세/매매 통계 등을 사용해 `LH 보증금 / 주변 전세 보증금 중위값` 비율을 예측합니다.

예측 대상은 `log1p_lh_vs_rent_median_ratio_winsor`입니다. 즉, 이상치 영향을 줄이기 위해 비율을 winsorizing한 뒤 `log1p` 변환한 값을 학습합니다. 예측 후에는 다시 원래 비율로 변환하여 적정 보증금과 95% 예측 구간을 계산합니다.

최종 테스트 성능은 `res/final_model/metadata.json` 기준으로 다음과 같습니다.

| 지표 | 값 |
| --- | ---: |
| RMSE | 0.1815 |
| MAE | 0.1308 |
| R2 | 0.9309 |

## 사용 변수

주요 입력 변수는 다음과 같습니다.

| 변수 | 설명 |
| --- | --- |
| `gu_code`, `gu_name` | 서울시 자치구 코드와 이름 |
| `ym` | 계약월 |
| `property_type` | 주택 유형 코드 |
| `주택유형`, `유형` | 원천 데이터의 주택 유형 정보 |
| `area_m2`, `area_m2_clean` | 전용면적 및 정제된 면적 |
| `방갯수`, `room_count_clean` | 방 개수 및 정제된 방 개수 |
| `세대원수`, `household_size_clean` | 세대원수 및 정제된 세대원수 |
| `rent_txn_count`, `jeonse_txn_count` | 지역/월/유형별 전월세 및 전세 거래 건수 |
| `rent_deposit_median`, `rent_deposit_mean` | 주변 전세 보증금 중위값과 평균값 |
| `rent_deposit_per_m2_median`, `rent_deposit_per_m2_mean` | 면적당 전세 보증금 통계 |
| `monthly_rent_median` | 월세 중위값 |
| `sale_txn_count` | 매매 거래 건수 |
| `sale_price_median`, `sale_price_mean` | 주변 매매가 중위값과 평균값 |
| `sale_price_per_m2_median`, `sale_price_per_m2_mean` | 면적당 매매가 통계 |
| `is_ood_property_type`, `is_in_domain_property_type` | 학습 범위 안/밖 주택 유형 여부 |
| `area_*`, `room_*`, `*_missing`, `*_imputed` | 데이터 품질 플래그 및 결측 보정 변수 |

추가로 모델 내부에서는 다음 조합 피처를 사용합니다.

| 변수 | 설명 |
| --- | --- |
| `gu_property` | 자치구 + 주택 유형 |
| `gu_ym` | 자치구 + 계약월 |
| `property_ym` | 주택 유형 + 계약월 |
| `housing_type_ym` | 주택유형 + 계약월 |
| `detail_type_ym` | 유형 + 계약월 |

## 위험도 산출

앱은 예측된 적정 보증금과 사용자가 입력한 보증금을 비교해 가격 위험 점수를 계산합니다. 입력 보증금이 예측 적정가 또는 95% 상단 구간보다 높으면 위험 점수가 높아지고, 너무 낮은 경우도 비정상 매물 가능성으로 반영합니다.

또한 KMeans와 Isolation Forest 기반의 비지도 학습 아티팩트를 함께 사용해 유사 매물 군집, 군집 내 거리, 이상치 점수를 제공합니다. 이 값은 가격 예측 자체보다는 매물이 학습 데이터 분포에서 얼마나 이질적인지 설명하는 보조 지표입니다.

## Git 업로드 예시

```bash
git add README.md
git commit -m "Add project README"
git push
```
