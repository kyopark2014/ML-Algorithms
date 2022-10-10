# Boosting

## Boosting 방식의 특징

부스팅(Boosting)은 여러 개의 약한 예측 모델을 순차적으로 구축하여, 반복적으로 오차를 개선하면서, 하나의 강한 예측 모델을 만드는 방식입니다.

- [앙상블 기법](https://github.com/kyopark2014/ML-Algorithms/blob/main/ensemble.md)에 속합니다.
- 부스팅에서는 개별 트리가 이전 트리를 기반으로 만들어집니다. 독립적으로 트리가 동작하지 않으며 다른 트리 위에 만들어집니다. 
- 부스팅은 반복적으로 오류를 고치므로, 약한 학습기를 강력한 학습기로 변환할 수 있습니다. 
- 각 단계에서 만드는 예측 모델은 이전 단계의 예측 모델의 단점을 보완합니다.
- 각 단계를 거치면서 예측 모델의 성능이 좋아집니다.
- Adaboost(Adaptive Boosting), GBM(Gradient Boosting Machines), XGBoost(eXtreme Gradient Boost), LightGBM(Light Gradient Boost Machines), CatBoost…
- 순차적으로 하므로 병렬화 안되므로 느린데, 이를 수정한게 [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)나 LightGBM이 있습니다.


## Gradient Boosting

Gradient Boosting은 타깃(Target)과 모델의 예측 사이에 손실함수를 정의하고, [경사하강법(Gradient desecent)](https://github.com/kyopark2014/ML-Algorithms/blob/main/stochastic-gradient-descent.md#gradient-descent)을 사용하여 잔차(Residual)가 최소화하는 방향으로 결정트리를 추가하는 앙상블 방법입니다. Scikit-learn의 Gradient Boosting Regression 모델의 기본 손실함수는 제곱 오차이고 분류 모델의 기본 손실 함수는 로지스틱 손실함수 입니다. 경사하강법으로 두 함수를 미분하면 y - y(sub)pred(/sub)을 얻을 수 있습니다. 

- 깊이가 얕은 트리를 사용하여 이전 트리의 오차를 보완하는 방식의 앙상블 방법입니다. 따라서, 과대 적합에 강하고 높은 일반화 성능을 기대할 수 있습니다.
- 이전 트리의 예측 오차를 기반으로 완전히 새로운 트리를 훈련합니다. 여기서 새로은 트리는 올바르게 예측된 값에는 영향을 받지 않습니다. 
- 결정트리 개수를 늘려도 과대 적합에 강하므로, 트리의 개수를 늘리거나 학습률을 증가시킬 수 있습니다. (n_estimators , learning_rate)
- 일반적으로 Random forest보다 나은 성능을 기대하지만 순차적으로 계산하여야 하므로 느립니다. 

### Basic Learner

Gradient boosting의 기본 학습기(Basic learner)는 [결정트리(Decision tree)](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)로서, 높은 정확도로 튜닝을 하지 않습니다. 이것은 Gradient boosting이 기본학습기에 의존하는 모델이 아니라 오차에서 학습하는 모델을 원하기 때문입니다. 따라서, max_depth가 1인 decision stump나 max_depth가 2나 3인 결정트리를 사용합니다. 



### Residual

Boosing에서 정확한 최종 예측을 만들기 위해 오차를 계산할 수 있어야 합니다. 잔차(Residual)는 관측된 데이터값(observed data value)인 타겟(Target)과 예측(predicted data value) 사이의 차이입니다. 잔차는 모델 예측이 정답에서 얼마나 떨어져있는지 알려주며 양수 또는 음수 입니다. 

[선형 회귀(Linear Regression)](https://github.com/kyopark2014/ML-Algorithms/blob/main/linear-regression.md)에서는 데이터에 얼마나 잘 맞는지를 평가하기 위하여 잔차의 제곱인 [결정계수(Coefficient of determination)](https://github.com/kyopark2014/ML-Algorithms/blob/main/evaluation.md#coefficient-of-determination)을 사용합니다.

### Classificlation

[GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)은 클래스별로 각 트리의 예측을 더한 후 [시그모이드 함수](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md#sigmoid)을 적용하여 예측 확율을 계산합니다. 

### Regression

[GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html?highlight=gradientboostingregressor#sklearn.ensemble.GradientBoostingRegressor)



### Hyperparameter

기본 Hyperparameter는 아래와 같습니다.

```python
{'alpha': 0.9, 'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1, 
'loss': 'squared_error', 'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 
'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 
'n_estimators': 100, 'n_iter_no_change': None, 'random_state': 2, 'subsample': 1.0, 'tol': 0.0001, 
'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
```

- learning_rate: 기본 학습기(Basic learner)의 오차를 기반으로 전체 앙상블을 만들면 모델에 처음 추가된 트리의 영향이 너무 크게 되므로, 모델 구축에 대한 영향을 조절하여 개별 트리의 기여를 줄입니다 (축소: shrinkage). 일반적으로 트리 개숫(n_estimators)를 늘리면 learning_rate를 줄여야 합니다. 

-  subsample: 기본 학습기에 사용될 셈플의 비율을 지정합니다. 기본값인 1.0보다 작으면 훈련할때 샘플의 일부만 사용하게 됩니다. 0.8이면 80%만 사용합니다. subsample이 1보다 작을때 확률적 경사 부스팅이라고 부릅니다. 확률적이라는 말은 모델에 무작위성이 주입된다는 뜻입니다. 



### Early stopping

Gradient Boosting에서는 일정한 수준 이상 손실 함수가 향상되지 않으면, 훈련을 종료할 수 있는 조기 종료(early stopping)를 제공합니다. 

validation_fraction (기본값 0.1)만큼 훈련 세트에서 검증 데이터를 덜어낸 다음 n_iter_no_change 반복 횟수동안 검증 점수가 tol(기본값 1e-4)만큼 향상되지 않으면 훈련을 종료합니다. n_iter_no_change의 기본값은 None으로 조기 종료를 수행하지 않습니다.  


### 추가된 결정트리의 갯수 

앙상블에 추가된 트리의 개수는 "estimator_"속성에서 아래처럼 확인 할 수 있습니다. 

```python
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(max_depth=depth, n_estimators=300, random_state=2)
len(gbr.estimators_)
```

## Case of Gradient Boosting

[gradient_boosting.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/gradient_boosting.ipynb)에 대해 설명합니다. 

아래와 같이 GradientBoostingClassifier을 이용하여 n_estimators의 기본값인 100을 사용할때 아래와 같은 결과를 얻을 수 있습니다. 

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

0.8881086892152563 0.8720430147331015
```

아래와 같이 n_estimators(Tree의 갯수)을 500으로 설정시에 과대정합(Overfit)된 결과를 얻을수 있습니다. 또한, learning_rate(학습륜)을 이용하여 학습속도를 조정할 수 있습니다. 

```python
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

0.9464595437171814 0.8780082549788999
```


## Histogram-based Gradient Boosting

- Gradient Boosting의 속도와 성능을 개선합니다. 
- 입력 특성을 256개로 나눔니다. 여기서 256개의 구간 중 하나를 떼어 놓고 누락된 값을 위해 사용합니다.
- 트리의 갯수 (n_estimator)를 지정하지 않고 부스팅 반복 횟수(max_iter)를 지정
- 특성 중요도를 확인하기 위하여 permutation_importance를 사용합니다. permutation_importance() 함수가 반환하는 객체는 반복해서 얻은 특성중요도, 평균, 표준편차를 담고 있습니다. 


[historam_gradient_boosting.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/historam_gradient_boosting.ipynb)에 대해 설명합니다.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate

hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

0.9321723946453317 0.8801241948619236
```

중요도를 확인하기 위하여 permutation_importance()을 이용합니다. 

```python
from sklearn.inspection import permutation_importance

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10,
                                random_state=42, n_jobs=-1)
print(result.importances_mean)

[0.05969231 0.20238462 0.049     ]
```


## XGBoost (eXtreme Gradient Boost)

[XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)에서는 [Kaggle](https://www.kaggle.com/)에서 유례없이 성공한 XGBoost 알고리즘에 대해 설명합니다.

## LightGBM

Microsoft에서 만든 LightGBM 으로 [Histogram-based Gradient Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#histogram-based-gradient-boosting) 알고리즘입니다. 


[light_gbm.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/light_gbm.ipynb)에 대해 설명합니다.

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate

lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

0.935828414851749 0.8801251203079884
```



## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
