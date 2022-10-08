# XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 예제 분석

여기에서는 [XGBoost와 사이킷런을 활용한 그레이디언트 부스팅](https://github.com/gilbutITbook/080263)의 예제 중심으로 설명합니다. 

## 자전거 대여 횟수 예측

[자전거 대여 데이터셋](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)을 이용하여 어떤 날의 날씨, 주중/주말, 온도, 습도를 이용해 그날의 자동차 대여 숫자를 예측하는 프로그램 입니다.  

#### Linear Regression을 이용한 예측

[linear-regression.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/linear-regression.ipynb)과 같이 RMSE 898.21로 예측되었습니다. 

#### XGBoost를 이용한 예측

[XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)의 XGBRegressor를 이용하여 자전거 대여 횟수를 예측하면 RMSE 

## Reference 

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)