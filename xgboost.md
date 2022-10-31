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

### 속도 향상

XGBoost는 속도에 주안점을 두어 설계되었습니다. 

- 근사 분할 탐색 알고리즘: XGBoost가 사용하는 근사 분할 탐색 알고리즘은 데이터를 나누는 퍼센트인 분위수(Quantile)을 사용하여 후보 분할을 제안합니다. global proposal에서는 동일한 분위수가 전체 훈련에 사용되고, local proposal에서는 각 분할 마다 새로운 분위수를 제안합니다. Quantile sketch algorithm은 가중치가 균일한 dataset에서 잘동작하는데 XGBoost는 이론적으로 보장된 병합과 가지치기를 기반으로 새로운 가중 퀀타일 스케치를 사용합니다. 

- 희소성 고려 분할 탐색: Dataset이 주로 누락된 값으로 구성되거나, One-hot encoding되어 있는 경우에 대부분의 원소가 0이거나 NULL인 희소 데이터 형태를 가집니다. 범주형 특성을 수치형 특성으로 만들기 위해서 pandas의 get_dummies()를 사용할때에도 one-hot encoding이 이용되어 집니다. XGBoost는 희소한 행렬을 탐색할때 매우 빠릅니다. 

- 병렬 컴퓨팅: XGBoost는 데이터를 Block 단위로 정렬하고 압축하여 여러대의 머신이나 외부 메모리에 분산될 수 있습니다. 분할 탐색 알고리즘은 블록의 장점을 사용해 분위수(Quantile) 탐색을 빠르게 수행합니다. 

- 캐시 고려 접근: XGBoost는 캐시를 고려한 Prefetching을 사용합니다. 내부 버퍼를 할당하고 Gradient 통계를 가져와 미니배치 방식으로 누적을 수행합니다. Prefetching은 읽기/쓰기 의존성을 느슨하게 만들고 많은 샘플을 가진 dataset에서 실행 부하를 50% 절감할 수 있습니다. 

- 블록 압축과 샤딩: 블록 압축(block compression)은 열을 압축하여 디스크를 읽는데 도움이 되며, 블록 샤딩 (block sharding)은 번갈아 가며 여러 디스크로 데이터를 샤딩하기 때문에 데이터를 읽는 시간을 줄여줍니다. 


### 정확도 향상 

XGBoost는 Gradient Boosting이나 Random Forest와 달리 학습하려는 목적 함수의 일부로 규제를 포함하고 있습니다. 규제(Regulaization)를 추가하여 분산을 줄이고 과대적합을 방지합니다. 


## XGboost Basic learner

XGBoost는 [Basic Learner](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#basic-learner)를 이용하여 오류로부터 
합습을 수행합니다.

### gbtree 

XGBooster는 기본 학습기(Basic leaner)로 가장 많이 사용되는 방식은 [결정트리(Decision Tree)](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)입니다. 데이터는 일반적으로 비선형이기 때문에 필요한만큼 데이터를 분할하여 셈플에 도달하는 결정트리는 비선형 데이터에 좋은 선택입니다.

### gblinear

데이터가 선형이라면 결정트리보다 선형모델을 기본학습기로 쓰는것을 바람직합니다. 선형 부스팅 모델에서는 앙상븗되는 각 모델이 선형일뿐, 기본 동작은 트리 부스팅 모델과 동일합니다. 기본 모델을 만들고 이어지는 후속 모델의 [잔차(Residual)](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#residual)를 바탕으로 학습하고, 개별 모델을 합하여 최종 결과를 만듭니다. 

gblinear는 선형 회귀에 규제항을 추가하는데, 여러번 부스팅하면 라소 회귀가 됩니다. 

또한, gblinear는 [로지스틱 회귀](https://github.com/kyopark2014/ML-Algorithms/blob/main/logistic-regression.md)로 분류문제에 사용할 수 있습니다. 

### DART

DART(Dropouts meet Multiple Additive Regression)은 결정트리의 한 형태이지만, 규제 방법으로 [신경망의 드롭아웃(Dropout)](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md#dropout) 기법을 사용합니다. 

새로운 부스팅 단계마다 새로운 모델을 만들기 위하여 이전 트리의 잔차를 더하지 않고 이전 트리를 랜덤하게 선택하고 1/k 배율로 리프 노드를 정규화합니다. 여기서 k는 드롭아웃된 트리의 개숫입니다. 

## Hyperparameters

#### booster

기본학습기(base learner)를 지정합니다. gbtree(default), gblinear, dart를 지정할 수 있습니다. 

#### objective

손실함수(Loss function)을 지정합니다. 

- reg:logistic은 decision을 위한 분류 문제에 사용합니다.
- binary:logistic은 probability로 분류 문제를 해결할때 사용합니다.
- reg:linear: linear한 회귀 문제에 사용
- reg:squarederror: 회귀(regression)에서 사용합니다. 


#### n_estimators

앙상불의 트리 개수를 의미합니다. [잔차(Residual)](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#residual)에 훈련되는 트리 개수로서 n_estimators를 늘리면 learning_rate을 줄여야 합니다. 

- 범위: [1, inf], 기본값: 100
- 값을 늘리면 대용량 데이터에서 성능을 높일 수 있습니다.

#### learning_rate

부스팅의 각 단계에서 [기본학습기(base learner)](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#basic-learner)의 기여도를 줄입니다. eta로도 불립니다. 기본학습기의 영향도를 줄이는것을 축소(shrinkage)라고 부르는데, 기본 학습기의 영향이 너무 크지 않도록 조정합니다. 



- 범위: [0, 1], 기본값: 0.3
- 값을 줄이면 좋은 성능을 위해 더 많은 트리가 필요하지만, 과대적합을 방지합니다. 
- learning_rate이 1이면 어떤 어떤 조정도 하지 않는다는 의미입니다. 

아래는 [xgboost-heart-desease-hpo.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-heart-desease-hpo.ipynb)와 같이 learning_rate으로 HPO하는 것을 보여줍니다. 

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

learning_rate_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

import time
start = time.time()

for value in learning_rate_values:
    xgb = XGBClassifier(booster='gbtree', objective='binary:logistic', 
                        random_state=2, verbosity=0, use_label_encoder=False, learning_rate=value)
    
    xgb.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    score = accuracy_score(y_pred, y_test)
    print('Accuracy:', np.round(score, 3))    
print('\nElased time: %0.2fs' % (time.time()-start))    
```




#### max_depth

분할 회수에 해당하는 트리의 깊이를 결정합니다. 개별 트리가 max_depth까지만 성정할 수 있기 때문에 max_depth를 제한하면 과대적합을 방지합니다. 

- 범위: [0, inf], 기본값: 6
- 값을 줄이면 과대적합을 방지합니다. 
- 0은 tree_method='hist', grow_policy='Lossguide'일때 선택할수 있으며 깊이에 제한이 없다는 의미입니다. 


#### gamma

라그랑주 승수(Lagrange multiplier) 또는 min_split_loss로도 불립니다. gamma는 노드 분할을 위한 최소 손실 감소를 지정합니다. 10이면 매우 높은 값이므로 이보다 작은 값으로 설정합니다. gamma를 높이면 보수적인 모델이 만들어집니다. 

- 범위: [0, inf], 기본값: 0
- 값을 늘리면 과대적합을 방지합니다. 


#### min_child_weight

노드 분할을 위해 필요한 최소 가중치 합입니다. 샘플 자중치 합이 min_child_weight보다 작으면 더이상 분할하지 않습니다. 

- 범위: [0, inf], 기본값: 1
- 값을 늘리면 과대적합을 방지합니다. 

#### subsample

부스팅 단계마다 사용할 훈련 샘플 개수의 비율입니다. 

- 범위: [0, 1], 기본값: 1
- 값을 줄이면 과대적합을 방지합니다. 

#### colsample_bytree

부스팅 단계마다 사용할 특성의 비율입니다. subsample과 비슷하게 colsample_bytree는 각 부스팅 단계마다 사용할 특성의 비율을 제한합니다. colsample_bytree는 특성의 영향을 제한하고 분산을 줄이는데 유용합니다. 

- 범위: [0, 1], 기본값: 1
- 값을 줄이면 과대적합을 방지합니다. 

#### colsample_bylevel

트리 깊이마다 사용할 특성 개수의 비율입니다.

- 범위: [0, 1], 기본값: 1
- 값을 줄이면 과대적합을 방지합니다. 

#### colsample_bynode

노드를 분할할 때마다 사용할 특성 개수의 비율입니다. 

- 범위: [0, 1], 기본값: 1
- 값을 줄이면 과대적합을 방지합니다. 

#### scale_pos_weight

불균형한 데이터에 사용합니다. 분류에만 사용합니다.

- 범위: [0, inf], 기본값: 1
- sum(음성 샘플)/sum(양성 샘플)로서 데이터의 불균형을 제어합니다. 

#### max_delta_step

불균형이 매우 심한 데이터셋에만 권장 됩니다. 

- 범위: [0, inf], 기본값: 0
- 값을 늘리면 과대적합을 방지합니다. 

#### reg_lambda

가중치 L2 규제입니다. 

- 범위: [0, inf], 기본값: 1
- 값을 늘리면 과대적합을 방지합니다. 

#### reg_alpha

가중치 L1 규제입니다. 

- 범위: [0, inf], 기본값: 0
- 값을 늘리면 과대적합을 방지합니다. 

#### missing

누락된 값을 -999.0과 같은 수치로 대체합니다. 

- 범위: [-inf, inf], 기본값: None
- 누락된 값을 자동으로 처리합니다. 

#### 기본값

[XGBClassifier 기본 파라메터](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/xgboost.ipynb)는 아래와 같습니다.
 
 ```java
{'objective': 'binary:logistic',
 'use_label_encoder': False,
 'base_score': None,
 'booster': None,
 'callbacks': None,
 'colsample_bylevel': None,
 'colsample_bynode': None,
 'colsample_bytree': None,
 'early_stopping_rounds': None,
 'enable_categorical': False,
 'eval_metric': None,
 'gamma': None,
 'gpu_id': None,
 'grow_policy': None,
 'importance_type': None,
 'interaction_constraints': None,
 'learning_rate': None,
 'max_bin': None,
 'max_cat_to_onehot': None,
 'max_delta_step': None,
 'max_depth': None,
 'max_leaves': None,
 'min_child_weight': None,
 'missing': nan,
 'monotone_constraints': None,
 'n_estimators': 100,
 'n_jobs': None,
 'num_parallel_tree': None,
 'predictor': None,
 'random_state': None,
 'reg_alpha': None,
 'reg_lambda': None,
 'sampling_method': None,
 'scale_pos_weight': None,
 'subsample': None,
 'tree_method': None,
 'validate_parameters': None,
 'verbosity': None}
 ```
 
 [XGBRegressor의 기본 파라메터](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-diabetes.ipynb)는 아래와 같습니다.
 
 ```java
 {'objective': 'reg:squarederror',
 'base_score': None,
 'booster': None,
 'colsample_bylevel': None,
 'colsample_bynode': None,
 'colsample_bytree': None,
 'enable_categorical': False,
 'gamma': None,
 'gpu_id': None,
 'importance_type': None,
 'interaction_constraints': None,
 'learning_rate': None,
 'max_delta_step': None,
 'max_depth': None,
 'min_child_weight': None,
 'missing': nan,
 'monotone_constraints': None,
 'n_estimators': 100,
 'n_jobs': None,
 'num_parallel_tree': None,
 'predictor': None,
 'random_state': None,
 'reg_alpha': None,
 'reg_lambda': None,
 'scale_pos_weight': None,
 'subsample': None,
 'tree_method': None,
 'validate_parameters': None,
 'verbosity': None}
 ```

## Model Saving
[How to save and load Xgboost in Python?](https://mljar.com/blog/xgboost-save-load-python/)와 같이 model.save_model()로 'json' 또는 'txt'를 사용할 수 있습니다.

또한, pickle.dump()로 memory snapshop을 만들어서 학습을 Resume 할 수 있습니다. 하지만 이 경우에 version에 주의하여야 합니다. 

## 기본 예제

아래에서는 [xgboost.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/xgboost.ipynb)에 대해 설명합니다. 추가적인 예제는 [XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 예제 분석](https://github.com/kyopark2014/ML-Algorithms/tree/main/xgboost)을 참조합니다. 

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

[Using XGBoost in Python Tutorial](https://www.datacamp.com/tutorial/xgboost-in-python)

[How to save and load Xgboost in Python?](https://mljar.com/blog/xgboost-save-load-python/)
