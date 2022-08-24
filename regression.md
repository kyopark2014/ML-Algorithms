# 회귀 (Regression)

회귀 (Regression)는 예측하고 싶은 종속변수가 숫자일때 사용하는 머신러닝 방법입니다.

## 선형회귀 (Linear Regression)

특성(feature)와 Target 사이의 관계를 "y = ax + b"와 같은 선형 방정식으로 표시합니다. 여기서 a(coefficient)는 기울기, 계수(coefficient), 가중치(weight)의 의미이고, b는 절편(intercept, constant)입니다. 

<img width="331" alt="image" src="https://user-images.githubusercontent.com/52392004/185773282-73e5dd34-6a64-4c8d-87a2-0261dc4053b7.png">

[Linear Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/linear-regression.md)에서는 농어의 길이/무게 데이터를 가지고 길이에 대한 무게를 예측하는것에 대해 설명합니다. 

## 다항회귀 (Polynomial Regression)

다항식을 사용하여 특성(feature)와 Target사이의 관계를 표현합니다.

[Polynomial Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/polynomial-regression.md)에서는 Polynomial Regression을 이용합니다.


## 다중회귀(Multiple Regression)

여러개의 특성을 사용하는 회귀모델, 소프트맥스함수사용

[다중회귀(Multiple Regression)](https://github.com/kyopark2014/ML-Algorithms/blob/main/multiple-regression.md)에서는 농어(perch)의 길이, 두께, 높이를 가지고 예측을 수행하는 방법에 대해 설명합니다. 


## 로지스틱 회귀 (LogisticRegression)

선형방정식을 사용한 분류 알고리즘으로 시그모이드 함수나 소프트맥스 함수를 사용하여 클래스 확률(0~1)을 출력할 수 있습니다. 

[Logistic Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/logistic-regression.md)에서는 상세한 로지스틱 회귀에 대해 설명합니다. 
 

## 특성공학(Feature Engineering)

[특성공학(Feature Engineering)](https://github.com/kyopark2014/ML-Algorithms/blob/main/feature-enginnering.md)이용하여 농어의 두께를 좀더 정확하게 예측할 수 있습니다. 이때, 과대적합을 방지하기 위하여 릿지와 라쏘로 규제(Regularization)을 수행합니다. 

## 관련 용어 

- 시그모이드함수 (Sigmoid Function): 선형방정식의 출력을 0에서 1사이의 확률로 압축합니다.

![image](https://user-images.githubusercontent.com/52392004/185773923-7ca38926-f792-46c6-b339-f8459c2fea8c.png)

- 소프트맥스함수 (Softmax Funnction): 다중분류에서 각클래스별 예측출력값을 0에서 1사이의 확률로 압축하고 전체 합이 1이 되도록 변환합니다.

![image](https://user-images.githubusercontent.com/52392004/186540209-23c813e4-9806-4fb3-ae07-1d3e673bdb94.png)

![image](https://user-images.githubusercontent.com/52392004/186540141-a25f1eaa-c287-4b30-8c58-8a63eb9cac29.png)


- 손실함수(Loss Function): 예측값과 실제 정답간의 차이를 표현하는 함수

   ◇ Regression: MSE(MeanSquaredError,평균제곱오차)
   
   ◇ Logistic Regression: Logistic loss function (Binary cross entropy loss function)
   
   ◇ MulticlassClassification: Cross entropy loss function

- 확률적경사하강법 (StochasticGradientDescent): 훈련세트에서 샘플을 무작위로 하나씩 꺼내 손실 함수의 경사를 계산하고 손실이 작아지는 방향으로 파라미터를 업데이트하는 알고리즘입니다. ,하이퍼파라미터인 learning rate(step size)와 epoch를 조정합니다.

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
