# 랜덤 포레스트 (Random Forest)

다수의 [Decision tree](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)의 분류 결과를 취합해서 최종 예측값을 결정하는 [앙상블 학습](https://github.com/kyopark2014/ML-Algorithms/blob/main/ensemble.md)입니다. 랜덤 포레스트의 개별모델은 결정트리로서 최종 예측을 위해 수백 또는 수천개의 결정트리로 구성될 수 있습니다. 

랜덤 포레스트는 32비트 정수 범위에서 난수를 만들어 개별트리의 random_state를 지정하여 관리하기 때문에, 언제든지 손쉽게 각 트리의 bootstraping smaple을 재현할 수 있습니다. 

## 분류 (Classification)

[Ensemble 방식](https://github.com/kyopark2014/ML-Algorithms/blob/main/ensemble.md)으로 <U>다수결 투표(Majority Vote)</U>를 사용합니다. [decision-tree-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/decision-tree-census.ipynb)은 [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)을 이용하여 수입을 에측하는 분류모델입니다. 

## 회귀 (Regression)

[Ensemble 방식](https://github.com/kyopark2014/ML-Algorithms/blob/main/ensemble.md)으로 모델의 예측을 평균하지만 개별 트리를 만들기 위해 Bagging(Boostrap Aggregating)을 사용합니다. [random-forest-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/random-forest-bike.ipynb)은 [Bike Rental 데이터](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)를 이용해 Bike 대여 숫자를 예측하는 회귀모델입니다. 



### Bagging

Bootstraping은 중복을 허용한 셈플링을 의미하는데, 랜덤 포레스트에서 개별 결정트리를 만들때에 Bootstraping을 수행합니다. 모든 결정트리가 다른 샘플을 사용하도록 원본 데이터셋과 동일한 크기의 Bootstraping을 이용합니다. 

## Hyperparameter

아래와 같이 Hyperpameter의 기본값을 확인할 수 있습니다. 

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, random_state=2, n_jobs=-1)
print(rf.get_params())
```
이때의 결과는 아래와 같습니다. 

```python
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 
'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 
'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 
'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 10, 'n_jobs': -1, 
'oob_score': False, 'random_state': 2, 'verbose': 0, 'warm_start': False}
```

- oob_score: Bagging에서 선택되지 않은 샘플을 테스트 샘플로 활용할 수 있습니다. "oob_score=True"로 설정하면, 랜덤 포레스트 모델 훈련후에, 각 트리에서 사용하지 않은 샘플을 사용해 개별 트리의 예측 점수를 누적하여 평균을 냅니다. 회귀 모델은 트리의 predict() mathod 출력을 누적하며, 분류 모델의 경우는 predict_probe() method의 출력을 누적합니다. predict_probe()는 leaf node에 있는 class 비율을 사용해 예측 확률을 반환합니다. 

- n_estimators: 앙상블할때 사용하는 트리의 개수를 지정합니다. v0.22부터 기본값이 10에서 100으로 변경되었습니다. 
- warm_start: "warm_start=True"로 지정하면 트리를 만들때에 이전 모델에 이어서 트리를 추가합니다. warm_start 매개변수를 사용해 n_estimators에 따라 [OOB 점수의 변화를 그래프](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/random-forest-census.ipynb)로 그릴 수 있습니다. 



## 구현방법

1) Bootstrap dataset 생성합니다.

2) Feature중 Random하게 n개 선택 후, 선택한 feature로 결정트리(Decision Tree) 생성후 반복합니다.

   - scikit-learn는 100개의 결정트리를 기본값으로 생성하여 사용 

3) Inference: 모든 Tree를 사용하고 분류한 후 최종 Voting합니다.

4) Validation: Bootstrap을 통해 랜덤 중복추출을 했을 때, Original dataset의 샘플 중에서 Bootstrap 과정에서 선택되지 않은 샘플들을 OOB(Out-of-Bag) 샘플이라 하고, 이 샘플들을 이용하여 Validation 수행합니다.

## 코드분석

[random_forest.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/random_forest.ipynb)에 대해 아래와 같이 설명합니다. 

1) 데이터를 준비합니다.


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
```

여기서 wine은 아래의 feature를 가지고 있습니다.

```python
wine.head()
```

![image](https://user-images.githubusercontent.com/52392004/186914946-170ca7d9-930e-4994-8135-0114537fc98f.png)


2) Random Forest 방식으로 Training을 수행합니다. 

- Train/Test dataset의 결과를 보면 아래와 같이 과대적합이지만, [결정트리로 수행한 결과](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)보다는 좋은 결과를 얻고 있습니다. 

- [k-fold cross validation를 이용한 교차검증](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md#k-fold-cross-validation%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D)을 cross_validate()을 이용해 수행하고 있습니다. n_splits를 지정하고 있지 않으므로 기본값인 5번 수행하고 있습니다. 

- return_train_score은 명시적으로 Train Fold의 점수를 받을지 여부를 설정합니다. 

```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

0.9973541965122431 0.8905151032797809
```

alcohol, sugar, pH의 중요도는 아래와 같이 sugar의 중요도가 더 높습니다. 

```python
rf.fit(train_input, train_target)
print(rf.feature_importances_)

[0.23167441 0.50039841 0.26792718]
```

OOB(Out-of-Bag) 샘플을 이용하여 아래와 같이 validation을 수행합니다. 

```python
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

rf.fit(train_input, train_target)
print(rf.oob_score_)

0.8934000384837406
```

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
