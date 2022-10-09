# Hyperparameter Optimization (HPO)

Hyperparameter optimization은 머신러닝 학습 알고리즘별 최적의 Hyperparameter 조합을 찾아가는 과정을 의미 합니다.

x축은 중요한 파라메터이고 y축은 중요하지 않은 파라메터라면, Grid search와 Random Search를 이용하여 최적의 Hyperparameter를 구할 수 있습니다. Grid Search는 9번 시도 했지만, 3개의 시도를 한 효과를 가지므로, Rnadom search가 일반적으로 Grid search보다 더 좋은 결과를 얻습니다. 그리고 Grid search는 Random search보다 느립니다. 

이것은 GridSearchCV 클래스와 RandomizedSearchCV 클래스를 이용해 구할 수 있습니다.

![image](https://user-images.githubusercontent.com/52392004/186670429-43eae8fc-7bc5-4a46-8ae8-91f827474604.png)


## 일반적인 가이드라인

- 알고리즘 별 hyperparameter를 이해합니다.
- 경험적으로 중요한 hyperparameter를 먼저 탐색하고 값을 고정합니다.
- 덜 중요한 hyperparameter를 나중에 탐색합니다.
- 먼저 넓은 범위에 대해 hyperparameter를 탐색하고 좋은 결과가 나온 범위에서 다시 탐색합니다.
- Random Search가 Grid Search에 더 적은 trial로 더 높은 최적화를 기대할 수 있습니다.
- HPO에 test dataset을 사용하지 않고, validation dataset을 사용합니다. 

## GridSearchCV

GridSearchCV 클래스는 fit() method에서 전달한 훈련 데이터를 사용해 [k-fold 교차 검증](https://github.com/kyopark2014/ML-Algorithms/blob/main/cross-validation.md#k-fold-cross-validation)을 수행합니다. fold 개수를 지정하는 cv의 기본값은 5입니다. [decision-tree-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/decision-tree-bike.ipynb)의 GridSearchCV 예제는 아래와 같습니다. 

```python
params = {'max_depth':[None,2,3,4,6,8,10,20]}

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=2)

from sklearn.model_selection import GridSearchCV
grid_reg = GridSearchCV(dt, params, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1)

grid_reg.fit(X_train, y_train)

best_params = grid_reg.best_params_
print("Best parameters:", best_params)

best_score = np.sqrt(-grid_reg.best_score_)
print("Best score: {:.3f}".format(best_score))

best_model = grid_reg.best_estimator_
y_pred = best_model.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: {:.3f}'.format(rmse_test))
```

## RandomizedSearchCV

탐색할 Hyperparameter가 많을 때 GridSearchCV로 튜닝을 하면 너무 오래시간이 걸릴수 있습니다. RandomizedSearchCV는 GridSearchCV와 동일한 방식으로 동작하지만 모든 Hyperparamter 조합을 테스트하지 않고, 랜덤한 조합을 테스트 하므로, 제한된 시간 안에 최상의 조합을 찾습니다. 

## 구현 예

### Decision Tree를 이용하여 Wine을 구분

Decision Tree를 이용하여 Wine을 구분하는 예제에 대하여, [hyperparameter_tuning.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/hyperparameter_tuning.ipynb)에서는 Grid/Random search를 HPO를 수행하고 있습니다. 

1) 데이터를 준비합니다.

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
print(train_input.shape, test_input.shape)

(5197, 3) (1300, 3)
```

2) Hyperparameter tuning 없이 Training한 경우는 아래와 같고, 과대적합(Overfiting)되고 있습니다.

```
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

0.996921300750433
0.8584615384615385
```

3) min_impurity_decrease(분할로 얻어질 최소한의 불순도 감소량)을 Hyperparameter tuning 한 경우입니다.

GridSearchCV로 min_impurity_decrease을 튜닝시에 이전보다 개선이 있습니다. 

```python
from sklearn.model_selection import GridSearchCV

params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

0.892053107562055
0.8615384615384616
```

최고일때의 min_impurity_decrease의 값은 0.001입니다.

```python
print(gs.best_params_)

{'min_impurity_decrease': 0.0001}
```

이때, 사용된 교차검증 Set들에 대한 결과는 아래와 같습니다. 

```python
print(gs.cv_results_['mean_test_score'])
print(np.max(gs.cv_results_['mean_test_score']))

[0.85780355 0.85799604 0.85799604 ... 0.86126601 0.86165063 0.86357629]
0.8683865773302731
```


4) 여러개 Hyperparameter tuning 한 경우

- min_impurity_decrease: 분할로 얻어질 최소한의 불순도 감소량
- max_depth: 트리의 최대 깊이
- min_samples_split: 분할되기 전에 노드가 가져야 하는 최소 셈플수 
- n_jobs=-1: 전체 리소스 사용

min_impurity_decrease만 tuning 했을때와 거의 동일한 결과를 얻습니다. 

```python
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

0.892053107562055
0.8615384615384616
```

이때의 최적값입니다. 

```python
print(gs.best_params_)

{'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
```

5) Random search를 이용하여 hyperparameter tuning시에 결과입니다.

- min_impurity_decrease: 분할로 얻어질 최소한의 불순도 감소량
- max_depth: 트리의 최대 깊이
- min_samples_split: 분할되기 전에 노드가 가져야 하는 최소 셈플수 
- min_samples_leaf: 리프노드가 가지고 있어야 할 셈플수

Grid Search와 유사한 결과를 얻었습니다. 

```python
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
          }
from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, 
                        n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

0.8928227823744468
0.86
```

이때 사용된 hyperparameter 값은 아래와 같습니다. 

```python
print(gs.best_params_)

{'max_depth': 39, 'min_impurity_decrease': 0.00034102546602601173, 'min_samples_leaf': 7, 'min_samples_split': 13}
```

## Decision Tree를 이용하여 Heart Disease를 예측 (Classification)

[decision-tree-heart-disease.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/decision-tree-heart-disease.ipynb)에서는 RandomizedSearchCV을 이용하여 HPO를 수행합니다. 

## Decision Tree를 이용하여 Census Income를 예측 (Classification)

[Decision-tree-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/decision-tree-census.ipynb)에서는 RandomizedSearchCV을 이용항 HPO를 수행합니다. 

## Decision Tree을 이용한 Bike Rental 수요 예측 (Regression)

[decision-tree-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/decision-tree-bike.ipynb)에서는 GridSearchCV를 이용한 HPO를 수행합니다. 


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)

