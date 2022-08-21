# Regression

## Basic

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

![image](https://user-images.githubusercontent.com/52392004/185773923-7ca38926-f792-46c6-b339-f8459c2fea8c.png)


- 소프트맥스함수 (SoftmaxFunnction): 다중분류에서 각클래스별 예측출력값을 0에서 1사이의 확률로 압축하고 전체 합이 1이 되도록 변환
- 손실함수(LossFunction): 예측값과 실제 정답간의 차이를 표현하는 함수
   ◇ Regression: MSE(MeanSquaredError,평균제곱오차)
   ◇ LogisticRegression: Logisticlossfunction(Binarycrossentropylossfunction)
   ◇  MulticlassClassification: Crossentropylossfunction
- 확률적경사하강법 (StochasticGradientDescent): 훈련세트에서 샘플을 무작위로 하나씩 꺼내 손실 함수의 경사를 계산하고 손실이 작아지는 방향으로 파라미터를 업데이트하는 알고리즘, 하이퍼파라미터인 learning rate(step size)와 epoch를 조정

## Linear Regression

[Linear Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/linear-regression.md)에서는 농어의 길이/무게 데이터를 가지고 길이에 대한 무게를 예측하는것에 대해 설명합니다. 

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
