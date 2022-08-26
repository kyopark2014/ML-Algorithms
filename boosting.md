# Boosting

## Gradient Boosting

- 경사하강법을 사용하여 트리를 앙상블에 추가합니다. 결정트리를 계속 추가하면서 가장 낮은 곳을 찾아 이동합니다.

- 깊이가 얕은 트리를 사용하여 이전 트리의 오차를 보완하는 방식의 앙상블 방법입니다. 따라서, 과대 적합에 강하고 높은 일반화 성능을 기대할 수 있습니다.

- 결정트리 개수를 늘려도 과대 적합에 강하므로, 트리의 개수를 늘리거나 학습률을 증가시킬 수 있음 (n_estimators , learning_rate)


## Histogram-based Gradient Boosting

- Gradient Boosting의 속도와 성능을 개선합니다. 
- 입력 특성을 256개로 나눔니다. 여기서 256개의 구간 중 하나를 떼어 놓고 누락된 값을 위해 사용합니다.
- 트리의 갯수 (n_estimator)를 지정하지 않고 부스팅 반복 횟수(max_iter)를 지정

## XGBoost (eXtreme Gradient Boost)

- “대규모의 데이터셋을 어떻게 더 빠르게 학습할 수 있을까?” 의 해결책입니다.
- 분류/회귀/Rank문제 모두 사용 가능합니다.
- Gradient Boosting 을 개량, Gradient Boosting의 보다 정규화된 형태입니다. 
- L1, L2 regularization을 사용하여 overfitting을 막고 모델을 일반화 합니다.
- 병렬 학습이 가능하여 빠른 학습 가능합니다.
- 학습데이터셋의 missing value도 처리 가능합니다.
- 높은 성능을 보여줍니다. 


XGBClassifier 클래스에서 tree_method=‘hist’ 로 지정하여 히스토그램 기반 그레이디언트 부스팅 알고리즘을 사용


## LightGBM

Microsoft에서 만든 LightGBM 으로 히스토그램 기반 그레이디언트 부스팅 알고리즘 사용
 

