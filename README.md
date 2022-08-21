# ML-Algorithms

## AI / ML

- 인공지능(AI, Artificial Intelligence): 사람처럼 학습하고 추론할 수 있는 지능을 가진 시스템을 만드는 기술

- 머신러닝(ML, Machine Learning): 규칙을 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘 연구, 예: Sciket-learn

- 딥려닝(DL, Deep Learning): 인공 신경망, 예: TensorFlow, PyTorch



## Binary Classification (HelloWorld)

[KNN을 이용한 binary classification](https://github.com/kyopark2014/ML-Algorithms/blob/main/helloworld.md)에서는 기본 이진분류를 노트북으로 구현합니다. 


## 데이터전처리 

특성값을 일정한 기준으로 맞추는 작업이 필요합니다. 이때 Z점수(표준점수, standard score)를 사용하여 각 데이터가 원점에서 몇 표준편차만큼 떨어져 있는지 나타내므로, 특성값의 크기와 상관없이 동일한 조건으로 비교가 가능합니다. 

<img width="230" alt="image" src="https://user-images.githubusercontent.com/52392004/185773237-2190452a-5aaf-40e7-af6f-08fbb618a524.png">

x: 대상데이터, x bar: 평균(mean), s: 표준편차(std)




## 용어 정리

#### Regression

- 회귀 (Regression): 예측하고 싶은 종속변수가 숫자일때 사용하는 머신러닝 방법입니다.

- 선형회귀 (LinearRegression): 특성(feature)와 Target 사이의 관계를 가장 잘 나타내는 아래와 같은 선형 방정식입니다. 여기서 a는 기울기, 계수(coefficient), 가중치(weight)이고, b는 절편입니다. 

```c
y=ax+b 
```
<img width="331" alt="image" src="https://user-images.githubusercontent.com/52392004/185773282-73e5dd34-6a64-4c8d-87a2-0261dc4053b7.png">

- 다항회귀 (PolynomialRegression): 다항식을 사용하여 특성(feature)와 Target사이의 관계를 표현, 비선형
- 다중회귀 (MultipleRegression): 여러개의 특성을 사용하는 회귀모델, 소프트맥스함수사용
- 로지스틱 회귀 (LogisticRegression):선형방정식을 사용한 분류 알고리즘.시그모이드 함수나 소프트맥스 함수를 사용하여 클래스 확률(0~1)을 출력할 수 있음
- 시그모이드함수 (Sigmoid Function): 선형방정식의 출력을 0에서 1사이의 확률로 압축
- 소프트맥스함수 (SoftmaxFunnction): 다중분류에서 각클래스별 예측출력값을 0에서 1사이의 확률로 압축하고 전체 합이 1이 되도록 변환
- 손실함수(LossFunction): 예측값과 실제 정답간의 차이를 표현하는 함수
   ◇ Regression: MSE(MeanSquaredError,평균제곱오차)
   ◇ LogisticRegression: Logisticlossfunction(Binarycrossentropylossfunction)
   ◇  MulticlassClassification: Crossentropylossfunction
- 확률적경사하강법 (StochasticGradientDescent): 훈련세트에서 샘플을 무작위로 하나씩 꺼내 손실 함수의 경사를 계산하고 손실이 작아지는 방향으로 파라미터를 업데이트하는 알고리즘, 하이퍼파라미터인 learning rate(step size)와 epoch를 조정

#### 모델 평가

일반적으로 train set의 score가 test set보다 조금 높음

- 과대적합(Overfitting): 모델의 train set 성능이 test set보다 훨씬 높은 경우 
- 과소적합(Underfitting): train set와 test set 성능이 모두 낮거나, test set 성능이 오히려 더 높은 경우
- 특성공학(Feature Engineering): 주어진 특성을 조합하여 새로운 특성을 만드는 과정

#### Scaler

아래와 같이 scikit-learn을 이용하여 표준점수로 변환할 수 있습니다. 

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)    

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

#### 규제 (Regularization)

모델이 과적합 되게 학습하지 않고 일반성을 가질 수 있도록 파라미터값에 제약을 주는것을 말합니다. L1 규제(Lasso), L2 규제(Ridge), alpha 값으로 규제량을 조정합니다. 

<img width="423" alt="image" src="https://user-images.githubusercontent.com/52392004/185773329-8b542165-3c41-42d9-ba0f-e437a2f9f811.png">


- Ridge: 계수를 제곱한 값을 기준으로 규제를 적용

```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
```

- Lasso: 계수의 절대값을 기준으로 규제를 적용 

```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
```

## [Amazon SageMaker Built-in Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)

### Predict if an item belongs to a category: an email spam filter

. Supervised Learning

. Problem type: Binary/multi-class classification

. 데이터는 table 형태로 제공  


#### Classification Algorithms 

. Factorization Machines Algorithm, 

. [kNN (K - Nearest Neighbor)](https://github.com/kyopark2014/ML-Algorithms/blob/main/KNN.md)은 Euclidean distance를 이용하여 가장 가까운 k개의 sample을 선택하여, 해당 sample의 결과값으로 예측할 수 있습니다. 

. Linear Learner Algorithm

. XGBoost Algorithm


. [Machine Learning의 Classification 방법에 따른 특징](https://en.wikipedia.org/wiki/MNIST_database)은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/162556347-9d57ea09-1741-4645-a785-82b27466e8a2.png)




