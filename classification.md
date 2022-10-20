# 분류 (Classification)

분류 (Classification)은 어떤 대상을 정해진 범주(Class)에 구분해 넣는 작업을 뜻합니다. 머신러닝에서 분류는 주어진 Feature에 따라 어떤 대상을 유한한 범주(categorical variable)로 구분하는 방법입니다. 여기서 **타깃값은 범주형 데이터** 점이 중요합니다. 범주형 데이터는 객관식 문제와 같이 선택지가 있는 값입니다. 개와 고양이를 구분하는 문제, 스팸 메일과 일반 메일을 구분하는 문제, 질병 검사 결과가 양성인지 음성인지 구분하는 문제등이 있습니다. 

Classifier에서 score() method 호출시, 분류(Classification)에서는 [정확도(Accuracy)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#accuracy)을 의미합니다. 반면에, [회귀 (Regression)](https://github.com/kyopark2014/ML-Algorithms/blob/main/regression.md)에서의 score()는 [결정계수(Coefficient of determination)](https://github.com/kyopark2014/ML-Algorithms/blob/main/evaluation.md#coefficient-of-determination)을 나타냅니다. 분류에서는 Accuracy 이외에도 [Confusion Matrix (오차행렬)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md)에 있는 [정밀도(Precision)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#precision), [재현율(Recall)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#recall), [F1 Score](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#f1-score)도 함께 활용합니다. 


## 분류의 종류

분류 알고리즘에는 퍼셉트론, Logistic Regression, 서포트 벡터 머신(SVM), 신경망, k-최근접 이웃(k-NN), 결정트리, 랜덤 포레스트, GBDT (Gradient Boost Decision Tree)가 있습니다. 


## k-최근접 이웃(k-NN)

[kNN(k-Nearest Neighbors)을 이용한 binary classification](https://github.com/kyopark2014/ML-Algorithms/blob/main/helloworld.md)에서는 기본 이진분류를 노트북으로 구현합니다. 


### 로지스틱 회귀 (Logistic Regression)

선형방정식을 사용한 분류 알고리즘으로 시그모이드 함수나 소프트맥스 함수를 사용하여 클래스 확률(0~1)을 출력할 수 있습니다. 

[Logistic Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/logistic-regression.md)에서는 상세한 로지스틱 회귀에 대해 설명합니다. 
 
### Decision Tree

[결정트리 (Decision Tree)](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)를 이용하여, 데이터에 내재되어 있는 패턴을 변수의 조합으로 나타내는 회귀/분류 모델을 Tree형태로 만들 수 있습니다. 

### Random Forest

[Random Forest](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)에서는 앙상블학습(Ensemble learning)의 하나인 Random Forest 방식에 대해 설명합니다. 

### Extra Trees

[Extra Trees](https://github.com/kyopark2014/ML-Algorithms/blob/main/extra-trees.md)에 대해 설명합니다. 

### Boosting

여러 개의 약한 예측 모델을 순차적으로 구축하여 하나의 강한 예측 모델을 만들수 있습니다. 여기서는 [Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md)을 사용하는 [Gradient Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#gradient-boosting), [Histogram-based Gradient Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#histogram-based-gradient-boosting), [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#xgboost-extreme-gradient-boost), [LightGBM](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#lightgbm)에 대해 설명합니다. 




## Activation Function

활성화함수(Activation Function)는 입력신호가 일정 기준 이상이면 출력신호로 변환하는 함수를 의미 합니다. [Activation Function](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md)에서는 linear, sigmoid, relu, tanh, softmax 등의 activation function에 대해 설명합니다.


## Loss Function

[손실함수(Loss Function)](https://github.com/kyopark2014/ML-Algorithms/blob/main/loss-function.md)는 예측값과 실제 정답간의 차이를 표현하는 함수입니다. 
 
## Stochastic Gradient Descent

[확률적경사하강법 (Stochastic Gradient Descent)](https://github.com/kyopark2014/ML-Algorithms/blob/main/stochastic-gradient-descent.md)은 Train set에서 샘플을 무작위로 하나씩 꺼내 손실 함수의 경사를 계산하고 손실이 작아지는 방향으로 파라미터를 업데이트하는 알고리즘입니다. 

Hyperper Parameter로 learning rate(step size와 epoch가 있습니다.


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[소프트맥스 회귀(Softmax Regression)](https://wikidocs.net/35476)

[Deep Learning with TensorFlow2 and Keras - Packt - images](https://static.packt-cdn.com/downloads/9781838823412_ColorImages.pdf)

[Deep Learning with TensorFlow2 and Keras - Packt - github](https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-Keras)

[딥러닝 텐서플로 교과서 - 서지영, 길벗](https://github.com/gilbutITbook/080263)

[머신러닝·딥러닝 문제해결 전략 - 신백균, 골든래빗](https://github.com/BaekKyunShin/musthave_mldl_problem_solving_strategy)
