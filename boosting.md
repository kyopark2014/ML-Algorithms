# Boosting

## Boosting 방식의 특징

부스팅(Boosting)은 여러 개의 약한 예측 모델을 순차적으로 구축하여, 반복적으로 오차를 개선하면서, 하나의 강한 예측 모델을 만드는 방식입니다.

- [앙상블 기법](에 속합니다.
- 부스팅에서는 개별 트리가 이전 트리를 기반으로 만들어집니다. 독립적으로 트리가 동작하지 않으며 다른 트리 위에 만들어집니다. 
- 부스팅은 반복적으로 오류를 고치므로, 약한 학습기를 강력한 학습기로 변환할 수 있습니다. 

- 각 단계에서 만드는 예측 모델은 이전 단계의 예측 모델의 단점을 보완합니다.
- 각 단계를 거치면서 예측 모델의 성능이 좋아집니다.
- Adaboost(Adaptive Boosting), GBM(Gradient Boosting Machines), XGBoost(eXtreme Gradient Boost), LightGBM(Light Gradient Boost Machines), CatBoost…
- 순차적으로 하므로 병렬화 안되므로 느린데, 이를 수정한게 XGBoost나 LightGBM이 있습니다.


## Gradient Boosting

Gradient Boosting은 [경사하강법(Gradient desecent)](https://github.com/kyopark2014/ML-Algorithms/blob/main/stochastic-gradient-descent.md#gradient-descent)을 사용하여 잔여오차를 최소화하는 방향으로 트리를 앙상블에 추가합니다. 결정트리를 계속 추가하면서 가장 낮은 곳을 찾아 이동합니다.

- 깊이가 얕은 트리를 사용하여 이전 트리의 오차를 보완하는 방식의 앙상블 방법입니다. 따라서, 과대 적합에 강하고 높은 일반화 성능을 기대할 수 있습니다.
- 결정트리 개수를 늘려도 과대 적합에 강하므로, 트리의 개수를 늘리거나 학습률을 증가시킬 수 있습니다. (n_estimators , learning_rate)
- 일반적으로 Random forest보다 나은 성능을 기대하지만 순차적으로 계산하여야 하므로 느립니다. 


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
