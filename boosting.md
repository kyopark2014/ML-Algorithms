# Boosting

## Boosting 방식의 특징

- 여러 개의 약한 예측 모델을 순차적으로 구축하여 하나의 강한 예측 모델을 만듬니다.
- 앙상블 기법에 속합니다.
- 각 단계에서 만드는 예측 모델은 이전 단계의 예측 모델의 단점을 보완합니다.
- 각 단계를 거치면서 예측 모델의 성능이 좋아집니다.
- Adaboost(Adaptive Boosting), GBM(Gradient Boosting Machines), XGBoost(eXtreme Gradient Boost), LightGBM(Light Gradient Boost Machines), CatBoost…
- 순차적으로 하므로 병렬화 안되므로 느린데, 이를 수정한게 XGBoost나 LightGBM이 있습니다.


## Gradient Boosting

- 경사하강법을 사용하여 트리를 앙상블에 추가합니다. 결정트리를 계속 추가하면서 가장 낮은 곳을 찾아 이동합니다.
- 깊이가 얕은 트리를 사용하여 이전 트리의 오차를 보완하는 방식의 앙상블 방법입니다. 따라서, 과대 적합에 강하고 높은 일반화 성능을 기대할 수 있습니다.
- 결정트리 개수를 늘려도 과대 적합에 강하므로, 트리의 개수를 늘리거나 학습률을 증가시킬 수 있음 (n_estimators , learning_rate)
- 일반적으로 Random forest보다 나은 성능을 기대하지만 순차적으로 계산하여야 하므로 느립니다. 

### 코드 분석 





## Histogram-based Gradient Boosting

- Gradient Boosting의 속도와 성능을 개선합니다. 
- 입력 특성을 256개로 나눔니다. 여기서 256개의 구간 중 하나를 떼어 놓고 누락된 값을 위해 사용합니다.
- 트리의 갯수 (n_estimator)를 지정하지 않고 부스팅 반복 횟수(max_iter)를 지정
- 특성 중요도를 확인하기 위하여 permutation_importance를 사용합니다. permutation_importance() 함수가 반환하는 객체는 반복해서 얻은 특성중요도, 평균, 표준편차를 담고 있습니다. 




## XGBoost (eXtreme Gradient Boost)

- Gradient Boost를 개량하여 대규모 dataset을 처리합니다. 
- 분류/회귀/Rank문제 모두 사용 가능합니다.
- Gradient Boosting 을 개량, Gradient Boosting의 보다 정규화된 형태입니다. 
- L1, L2 regularization을 사용하여 overfitting을 막고 모델을 일반화 합니다.
- 병렬 학습이 가능하여 빠른 학습 가능합니다.
- 학습데이터셋의 missing value도 처리 가능합니다.
- 높은 성능을 보여줍니다. 
- scikit-learn에서 지원하지 않습니다. 
- XGBClassifier 클래스에서 tree_method=‘hist’로 지정하여 히스토그램 기반 그레이디언트 부스팅 알고리즘을 사용합니다. 



## LightGBM

Microsoft에서 만든 LightGBM 으로 히스토그램 기반 그레이디언트 부스팅 알고리즘입니다. 
 


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
