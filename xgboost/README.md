# XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 예제 분석

여기에서는 [XGBoost와 사이킷런을 활용한 그레이디언트 부스팅](https://github.com/gilbutITbook/080263)의 예제 중심으로 설명합니다. 

[XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)를 참조하여 아래와 같은 케이스에 대해 예측합니다.

## 자전거 대여 횟수 예측

[Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)의 데이터에 있는 날씨, 주중/주말, 온도, 습도를 이용해 그날의 자동차 대여 숫자를 이용하여 자전거 대여 횟수를 예측합니다.  

![image](https://user-images.githubusercontent.com/52392004/194688735-fc3c810e-a53b-4d2e-8e9a-ac226a4a60ac.png)


1) [wrangling-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-bike.ipynb)와 같이 전처리를 합니다. 

2) [Linear Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/linear-regression.md)을 이용하여 [linear-regression-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/linear-regression-bike.ipynb)와 같이 RMSE 898.21로 예측되었습니다. 

3) [Decision Tree](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)를 이용하여 [decision-tree-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/decision-tree-bike.ipynb)와 같이 RMSE 945로 예측하였으나, HPO를 통해 855까지 개선하였습니다. 

4) [Random Forest](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)를 이용하여 [random-forest-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/random-forest-bike.ipynb)와 같이 RMSE가 689.644로 예측되었으나, HPO를 통해 619.01까지 개선되었습니다. 

5) [Gradient Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#gradient-boosting)을 이용하여 [gradient-boosting-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/gradient-boosting-bike.ipynb)와 같이 RMSE가 648.426로 예측되었으나, HPO를 통해 596.954로 개선되었습니다. 

6) [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)의 XGBRegressor를 이용하여 [xgboost-regression-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-regression-bike.ipynb)와 같이 RMSE 705.11로 예측하였으나, HPO를 통해 584.34로 개선되었습니다. 이와 같이 XGBoost는 일반적으로 다른 방법들보다 더 좋은 성능을 내고 더 빠르게 결과를 얻을 수 있습니다. 




## Census Income 분류

[Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)을 이용하여 수익이 50K 이상과 이하의 수익을 가진 사람들을 분류합니다.

![image](https://user-images.githubusercontent.com/52392004/194688739-66d5a6e5-a211-4db0-8bc2-044b6a25d5ee.png)

1) [wrangling-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-census.ipynb)와 같이 전처리를 합니다. 

2) [logistic regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/logistic-regression.md)을 이용하여 [logistic-regression-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/logistic-regression-census.ipynb)와 같이 0.8의 정확도로 예측하였습니다. 

3) [Decision Tree](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)를 이용하여 [decision-tree-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/decision-tree-census.ipynb)와 같이 0.81로 예측하였으나 HPO를 통해 0.85까지 개선하였습니다. 

4) [Random Forest](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)를 이용하여 [random-forest-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/random-forest-census.ipynb)와 같이 0.84의 정확도로 예측하였으나 HPO를 통해 0.86으로 개선하였습니다.

5) [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)의 XGBClassifier를 이용하여 [xgboost-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-census.ipynb)와 같이 0.86의 정확도로 예측하였습니다. XGBoost는 보통 logistic regression보다 좋은 결과를 얻을 수 있습니다. 


## Heart Disease를 분류

[심장질환 데이터셋 (Heart Disease Data Set)](https://archive.ics.uci.edu/ml/datasets/heart+Disease)을 이용하여 심장질환 여부를 분류합니다. 

![image](https://user-images.githubusercontent.com/52392004/194788662-5456298d-3adf-44db-b28d-c2a681b1ce45.png)

1) [Decision Tree](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)를 이용하여 [decision-tree-heart-disease.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/decision-tree-heart-disease.ipynb)에서는 0.86의 정확도로 예측하였으나, HPO를 통해 0.87로 개선하였습니다. 

## 외계 행성 찾기 분류

[외계 해성 데이터셋은 2017년 캐글에 소개된 데이터셋](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)으로 [XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)에 있는 [exoplanets.csv.zip](https://raw.githubusercontent.com/rickiepark/handson-gb/main/Chapter04/exoplanets.csv.zip)을 이용하여, 어떤 행성이 별을 가지고 있는지 예측합니다. 

![image](https://user-images.githubusercontent.com/52392004/194874087-721d62ee-bd62-4daf-901f-42b1576328d0.png)


1) [Gradient Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#gradient-boosting)을 이용하여 [gradient-boosting-exoplanets.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/gradient-boosting-exoplanets.ipynb)와 같이 구하면, 0.99의 정확도로 215초가 소요됩니다.

2) [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)의 XGBClassifier를 이용하여 [xgboost-exoplanets.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-exoplanets.ipynb)와 같이 구하면, 0.99의 정확도로 8초가 소요됩니다.

## Reference 

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
