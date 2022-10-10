# 앙상블 학습

앙상블 학습(Ensemble learning)은 여러 개의 분류기를 생성하고, 그 예측을 결합함으로써 보다 정확한 예측을 도출하는 기법을 의미합니다. 여러 모델의 결과를 연결하기 때문에 오차를 줄이고 더 나은 성능을 내는 경향이 있습니다. 

## Bagging

Bagging(Bootstrap Aggregating)은 개별 트리를 생성할때 부트스트랩 샘플을 사용합니다. [랜덤포레스트](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)의 회귀문제가 Bagging을 이용하여 회귀 문제를 풉니다. 여기서 Bootstraping은 중복을 허용한 셈플링을 의미합니다.


## Ensenble Classification

다수결 투표(Majority vote)를 이용합니다. [Scikit-learn의 VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)처럼 사용자가 선택한 여러 종류의 머신러닝 모델을 연결하거나, [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)나 [랜덤 포레스트](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)처럼 같은 종류의 모델을 여러거 합치는 방식으로 구현합니다. 


## Reference 

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
