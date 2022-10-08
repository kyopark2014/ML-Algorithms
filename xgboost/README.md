# XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 예제 분석

여기에서는 [XGBoost와 사이킷런을 활용한 그레이디언트 부스팅](https://github.com/gilbutITbook/080263)의 예제 중심으로 설명합니다. 

[XGBoost]()에서 XGBoost에 대해 설명하고 있습니다.

## 자전거 대여 횟수 예측

[Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)의 데이터에 있는 날씨, 주중/주말, 온도, 습도를 이용해 그날의 자동차 대여 숫자를 이용하여 자전거 대여 횟수를 예측합니다.  

1) [wrangling-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-bike.ipynb)와 같이 전처리를 합니다. 

2) [Linear Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/linear-regression.md)을 이용하여 [linear-regression-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/linear-regression-bike.ipynb)와 같이 RMSE 898.21로 예측되었습니다. 

3) [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)의 XGBRegressor를 이용하여 [xgboost-regression-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-regression-bike.ipynb)와 같이 RMSE 705.11로 예측하였습니다. XGBoost는 보통 Linear Regression보다 좋은 결과를 얻을 수 있습니다. 


## Census Income 분류

[Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)을 이용하여 수익이 50K 이상과 이하의 수익을 가진 사람들을 분류합니다.

1) [wrangling-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-census.ipynb)와 같이 전처리를 합니다. 

2) [logistic regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/logistic-regression.md)을 이용하여 [logistic-regression-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/logistic-regression-census.ipynb)와 같이 0.8의 정확도로 예측하였습니다. 

3) [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)의 XGBClassifier를 이용하여 [xgboost-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-census.ipynb)와 같이 0.86의 정확도로 예측하였습니다. XGBoost는 보통 logistic regression보다 좋은 결과를 얻을 수 있습니다. 


## Reference 

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
