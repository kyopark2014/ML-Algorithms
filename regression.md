# 회귀 (Regression)

회귀(Regression)는 어떤 입력 데이터로부터 연속값을 에측합니다. 즉, 회귀(Regression)는 예측하고 싶은 종속변수(devendent variable)가 숫자일때 사용하는 머신러닝 방법으로, 독립변수(independent variable)와 종속변수 간의 관계를 모델링하는 방법입니다. 회귀가 [분류(Classification)](https://github.com/kyopark2014/ML-Algorithms/blob/main/regression.md)와 다른점은 종속변수(타깃값)가 범주형(Categorical) 데이터가 아니라는 사실입니다. 회귀에서 종속변수는 수치형 데이터입니다. 독립변수와 종속변수의 관계에 따라 선형회귀(Linear regression)과 다중 선형 회귀(multiple linear regression)등으로 구분 됩니다. 

회귀의 대표적인 평가지표는 [평균제곱근(RMSE)](https://github.com/kyopark2014/ML-Algorithms/blob/main/evaluation.md#rmse)입니다. 

## k-Nearest Neighbors

[k-최근접 이웃 회귀 (k-Nearest Neighbors)](https://github.com/kyopark2014/ML-Algorithms/blob/main/KNN.md)은 가장 가까운 이웃 샘플을 찾고 이 샘플들의 타깃 값을 평균하여 예측값으로 삼습니다. 


## Linear Regression

선형회귀 (Linear Regression)은 특성(feature)과 Target 사이의 관계를 가중치(Weight)를 이용하여 선형 방정식으로 표시합니다. [Linear Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/linear-regression.md)에서는 농어의 길이/무게 데이터를 가지고 길이에 대한 무게를 예측하는 예제를 가지고 설명합니다. 

## Polynomial Regression

다항회귀(Polynomial Regression)은 다항식을 사용하여 특성(feature)와 Target사이의 관계를 표현합니다.

[Polynomial Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/polynomial-regression.md)에서는 Polynomial Regression을 이용합니다.


## Multiple Regression

다중회귀 (Multiple Regression)은 여러개의 특성을 사용하는 회귀 모델로서, [softmax](https://github.com/kyopark2014/ML-Algorithms/blob/main/classification.md#softmax)함수를 사용합니다. 

[다중회귀(Multiple Regression)](https://github.com/kyopark2014/ML-Algorithms/blob/main/multiple-regression.md)에서는 농어(perch)의 길이, 두께, 높이를 가지고 예측을 수행하는 방법에 대해 설명합니다. 

## Feature Engineering

[특성공학(Feature Engineering)](https://github.com/kyopark2014/ML-Algorithms/blob/main/feature-enginnering.md)이용하여 농어의 두께를 좀더 정확하게 예측할 수 있습니다. 이때, 과대적합을 방지하기 위하여 릿지와 라쏘로 규제(Regularization)을 수행합니다. 


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[머신러닝·딥러닝 문제해결 전략 - 신백균, 골든래빗](https://github.com/BaekKyunShin/musthave_mldl_problem_solving_strategy)

