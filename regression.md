# 회귀 (Regression)

회귀 (Regression)는 예측하고 싶은 종속변수가 숫자일때 사용하는 머신러닝 방법입니다.

## k-최근접 이웃 회귀(k-Nearest Neighbors)

[kNN (k-Nearest Neighbors)](https://github.com/kyopark2014/ML-Algorithms/blob/main/KNN.md)은 가장 가까운 이웃 샘플을 찾고 이 샘플들의 타깃 값을 평균하여 예측값으로 삼습니다. 


## 선형회귀 (Linear Regression)

특성(feature)와 Target 사이의 관계를 "y = ax + b"와 같은 선형 방정식으로 표시합니다. 여기서 a(coefficient)는 기울기, 계수(coefficient), 가중치(weight)의 의미이고, b는 절편(intercept, constant)입니다. 

<img width="331" alt="image" src="https://user-images.githubusercontent.com/52392004/185773282-73e5dd34-6a64-4c8d-87a2-0261dc4053b7.png">

[Linear Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/linear-regression.md)에서는 농어의 길이/무게 데이터를 가지고 길이에 대한 무게를 예측하는것에 대해 설명합니다. 

## 다항회귀 (Polynomial Regression)

다항식을 사용하여 특성(feature)와 Target사이의 관계를 표현합니다.

[Polynomial Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/polynomial-regression.md)에서는 Polynomial Regression을 이용합니다.


## 다중회귀 (Multiple Regression)

여러개의 특성을 사용하는 회귀모델, 소프트맥스함수사용

[다중회귀(Multiple Regression)](https://github.com/kyopark2014/ML-Algorithms/blob/main/multiple-regression.md)에서는 농어(perch)의 길이, 두께, 높이를 가지고 예측을 수행하는 방법에 대해 설명합니다. 

## 특성공학 (Feature Engineering)

[특성공학(Feature Engineering)](https://github.com/kyopark2014/ML-Algorithms/blob/main/feature-enginnering.md)이용하여 농어의 두께를 좀더 정확하게 예측할 수 있습니다. 이때, 과대적합을 방지하기 위하여 릿지와 라쏘로 규제(Regularization)을 수행합니다. 


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)


