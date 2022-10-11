# XGBoost Algorithm

XGBoost는 여러개의 머신러닝 모델(basic learner)를 연결하여 사용하는 [앙상블 학습(Ensemble method)](https://github.com/kyopark2014/ML-Algorithms/blob/main/ensemble.md)입니다. 

- [Gradient Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#gradient-boosting)를 개량하여, 보다 정규화된 형태가 되었으며, 대규모 dataset을 처리할 수 있습니다.
- 분류/회귀/Rank문제 모두 사용 가능합니다.
- L1, L2 regularization을 사용하여 overfitting을 막고 모델을 일반화 합니다.
- 병렬 학습이 가능하여 빠른 학습 가능합니다.
- 학습데이터셋의 missing value도 처리 가능합니다.
- 높은 성능을 보여줍니다. 
- scikit-learn에서 지원하지 않습니다. 
- XGBClassifier 클래스에서 tree_method=‘hist’로 지정하여 히스토그램 기반 그레이디언트 부스팅 알고리즘을 사용합니다. 

## 속도 향상

XGBoost는 속도에 주안점을 두어 설계되었습니다. 

### 근사 분할 탐색 알고리즘

XGBoost가 사용하는 근사 분할 탐색 알고리즘은 데이터를 나누는 퍼센트인 분위수(Quantile)을 사용하여 후보 분할을 제안합니다. global proposal에서는 동일한 분위수가 전체 훈련에 사용되고, local proposal에서는 각 분할 마다 새로운 분위수를 제안합니다. Quantile sketch algorithm은 가중치가 균일한 dataset에서 잘동작합니다. XGBoost는 이론적으로 보장된 병합과 가지치기를 기반으로 새로운 가중 퀀타일 스케치를 사용합니다. 

### 희소성 고려 분할 탐색

Dataset이 주로 누락된 값으로 구성되거나, One-hot encoding되어 있는 경우에 대부분의 원소가 0이거나 NULL인 희소 데이터 형태를 가집니다. 범주형 특성을 수치형 특성으로 만들기 위해서 pandas의 get_dummies()를 사용할때에도 one-hot encoding이 이용되어 집니다. XGBoost는 희소한 행렬을 탐색할때 매우 빠릅니다. 

### 병렬 컴퓨팅 



## Basic learner

XGBooster는 기본 학습기(Basic leaner)로 가장 많이 사용되는 방식은 [결정트리(Decision Tree)](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)입니다. 

[xgboost.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/xgboost.ipynb)에 대해 설명합니다. 

```python
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate

xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

0.9555033709953124 0.8799326275264677
```

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
