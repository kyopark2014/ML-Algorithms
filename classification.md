# 분류 알고리즘 (Classification)

## 로지스틱 회귀 (Logistic Regression)

선형방정식을 사용한 분류 알고리즘으로 시그모이드 함수나 소프트맥스 함수를 사용하여 클래스 확률(0~1)을 출력할 수 있습니다. 

[Logistic Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/logistic-regression.md)에서는 상세한 로지스틱 회귀에 대해 설명합니다. 
 
## 관련 용어 

### Sigmoid

시그모이드함수 (Sigmoid Function)는 선형방정식의 출력을 0에서 1사이의 확률로 압축합니다.

![image](https://user-images.githubusercontent.com/52392004/185773923-7ca38926-f792-46c6-b339-f8459c2fea8c.png)

### Softmax

소프트맥스함수 (Softmax Funnction)는 다중분류에서 각클래스별 예측출력값을 0에서 1사이의 확률로 압축하고 전체 합이 1이 되도록 변환합니다.

k차원의 벡터에서 i번째 원소를 z_i, i번째 클래스가 정답일 확률을 p_i로 나타낸다고 하였을 때, 소프트맥스 함수는 를 다음과 같이 정의합니다.

![image](https://user-images.githubusercontent.com/52392004/186542833-891b29e9-c112-42eb-ba1a-d3023753ccb5.png)

만약 k=3이라면, 소프트맥스 함수는 아래와 같은 출력을 리턴합니다. 여기서, p1은 1번 class일 확율을 나타내고, 전체 확율의 합은 1입니다.

![image](https://user-images.githubusercontent.com/52392004/186542970-f41721df-7539-4424-a922-1e375859e889.png)

[Logistric regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/logistic-regression.md)의 다중분류를 한 예로 보면, 아래와 같이 Fish 데이터가 어떤 class일 확율을 sikit-learn의 LogisticRegression을 이용하여 softmax 함수로 구하고 있음을 알수 있습니다. 


```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

decision = lr.decision_function(test_scaled[:5])

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
```

이때의 결과는 아래와 같습니다. 여기서, Fish0의 데이터는 Perch인 확율이 0.841임을 알수 있습니다.

![image](https://user-images.githubusercontent.com/52392004/186540141-a25f1eaa-c287-4b30-8c58-8a63eb9cac29.png)

### Loss Function

손실함수(Loss Function)는 예측값과 실제 정답간의 차이를 표현하는 함수입니다. 

- Regression: MSE(MeanSquaredError,평균제곱오차)
   
- Logistic Regression: Logistic loss function (Binary cross entropy loss function)
   
- MulticlassClassification: Cross entropy loss function

### Stochastic Gradient Descent

[확률적경사하강법 (Stochastic Gradient Descent)](https://github.com/kyopark2014/ML-Algorithms/blob/main/stochastic-gradient-descent.md)은 Train set에서 샘플을 무작위로 하나씩 꺼내 손실 함수의 경사를 계산하고 손실이 작아지는 방향으로 파라미터를 업데이트하는 알고리즘입니다. 

Hyperper Parameter로 learning rate(step size)와 epoch가 있습니다.


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[소프트맥스 회귀(Softmax Regression)](https://wikidocs.net/35476)
