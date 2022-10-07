# XGBoost Algorithm

XGBoost는 여러개의 머신러닝 모델(basic learner)를 연결하여 사용하는 앙상블 학습(Ensemble method)입니다. 

- Gradient Boost를 개량하여 대규모 dataset을 처리합니다. 
- 분류/회귀/Rank문제 모두 사용 가능합니다.
- Gradient Boosting 을 개량, Gradient Boosting의 보다 정규화된 형태입니다. 
- L1, L2 regularization을 사용하여 overfitting을 막고 모델을 일반화 합니다.
- 병렬 학습이 가능하여 빠른 학습 가능합니다.
- 학습데이터셋의 missing value도 처리 가능합니다.
- 높은 성능을 보여줍니다. 
- scikit-learn에서 지원하지 않습니다. 
- XGBClassifier 클래스에서 tree_method=‘hist’로 지정하여 히스토그램 기반 그레이디언트 부스팅 알고리즘을 사용합니다. 


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
