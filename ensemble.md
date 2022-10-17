# 앙상블 학습

앙상블 학습(Ensemble learning)은 여러 개의 분류기를 생성하고, 그 예측을 결합함으로써 보다 정확한 예측을 도출하는 기법을 의미합니다. 여러 모델의 결과를 연결하기 때문에 오차를 줄이고 더 나은 성능을 내는 경향이 있습니다. [Random Forest](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)는 Bagging(Bootstrap Aggregating)을 사용하고, [Gradient Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#gradient-boosting)이나 [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)는 Boosting을 사용합니다.

## Bagging

Bagging(Bootstrap Aggregating)은 개별 트리를 생성할때 부트스트랩 샘플을 사용합니다. 여기서 Boostraping은 중복을 허용하는 샘플링을 의미하는데, 원본 데이터셋과 동일한 크기의 Bootstraping을 이용합니다.

Bagging은 개별 모델로 예측한 결과를 결합해 최종 예측을 정하는 기법으로 개별 모델이 서로 다른 샘플링 데이터를 활용하는 특징이 있습니다. Bagging은 원리가 간단하면서도 성능을 높일수 있는 효과적인 방법입니다. [랜덤포레스트](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)의 회귀문제가 Bagging을 이용합니다.

배깅 모델에서는 새로운 트리가 이전 트리에 주의를 기울이지 않습니다. 또한 새로운 트리는 부트스트래핑을 사용해 처음부터 훈련되며, 최종 모델은 모든 개별 트리의 결과를 합칩니다. 

## Boosting

[부스팅(Boositng)](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md)에서는 이전 트리의 오차를 기반으로 새로운 트리를 훈련합니다. 가중치를 활용하여 분류 성능이 약한 모델을 강하게 만드는 기법입니다. 


## Ensemble Classification

다수결 투표(Majority vote)를 이용합니다. [Scikit-learn의 VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)처럼 사용자가 선택한 여러 종류의 머신러닝 모델을 연결하거나, [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)나 [랜덤 포레스트](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)처럼 같은 종류의 모델을 여러거 합치는 방식으로 구현합니다. 


## Reference 

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
