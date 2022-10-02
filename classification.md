# 분류 알고리즘 (Classification)

여기에서는 Sample을 몇개의 Class중에 하나로 분류하는 문제를 다릅니다.

## 로지스틱 회귀 (Logistic Regression)

선형방정식을 사용한 분류 알고리즘으로 시그모이드 함수나 소프트맥스 함수를 사용하여 클래스 확률(0~1)을 출력할 수 있습니다. 

[Logistic Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/logistic-regression.md)에서는 상세한 로지스틱 회귀에 대해 설명합니다. 
 
## Decision Tree

[결정트리 (Decision Tree)](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)를 이용하여, 데이터에 내재되어 있는 패턴을 변수의 조합으로 나타내는 회귀/분류 모델을 Tree형태로 만들 수 있습니다. 

## Random Forest

[Random Forest](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)에서는 앙상블학습(Ensemble learning)의 하나인 Random Forest 방식에 대해 설명합니다. 

## Extra Trees

[Extra Trees](https://github.com/kyopark2014/ML-Algorithms/blob/main/extra-trees.md)에 대해 설명합니다. 

## Boosting

여러 개의 약한 예측 모델을 순차적으로 구축하여 하나의 강한 예측 모델을 만들수 있습니다. 여기서는 [Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md)을 사용하는 [Gradient Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#gradient-boosting), [Histogram-based Gradient Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#histogram-based-gradient-boosting), [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#xgboost-extreme-gradient-boost), [LightGBM](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#lightgbm)에 대해 설명합니다. 


## Activation Function

활성화함수(Activation Function)는 입력신호가 일정 기준 이상이면 출력신호로 변환하는 함수를 의미 합니다. [Activation Function](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md)에서는 linear, sigmoid, relu, tanh, softmax 등의 activation function에 대해 설명합니다.


## Loss Function

손실함수(Loss Function)는 예측값과 실제 정답간의 차이를 표현하는 함수입니다. 

- Regression: MSE(MeanSquaredError,평균제곱오차)
   
- Logistic Regression: Logistic loss function (Binary cross entropy loss function)
   
- MulticlassClassification: Cross entropy loss function

[Neural Network - Loss Function](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md#loss-function)에서는 신경망에서 손실함수를 사용하는 예를 보여주고 있습니다.

## Stochastic Gradient Descent

[확률적경사하강법 (Stochastic Gradient Descent)](https://github.com/kyopark2014/ML-Algorithms/blob/main/stochastic-gradient-descent.md)은 Train set에서 샘플을 무작위로 하나씩 꺼내 손실 함수의 경사를 계산하고 손실이 작아지는 방향으로 파라미터를 업데이트하는 알고리즘입니다. 

Hyperper Parameter로 learning rate(step size)와 epoch가 있습니다.


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[소프트맥스 회귀(Softmax Regression)](https://wikidocs.net/35476)
