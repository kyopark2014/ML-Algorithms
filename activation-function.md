# 활성화함수 (Activation Function)

활성화함수(Activation Function)는 입력신호가 일정 기준 이상이면 출력신호로 변환하는 함수를 의미 합니다. [Nueral Network - Activation Function](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md#activation-function)에서는 신경망에서 Activation Function에 대해 예제를 보여주고 있습니다. 

### Linear


### Sigmoid

시그모이드함수 (Sigmoid Function)는 선형방정식의 출력을 0에서 1사이의 확률로 압축합니다.

![image](https://user-images.githubusercontent.com/52392004/185773923-7ca38926-f792-46c6-b339-f8459c2fea8c.png)

### Softmax

소프트맥스함수 (Softmax Funnction)는 다중분류에서 각클래스별 예측출력값을 0에서 1사이의 확률로 압축하고 전체 합이 1이 되도록 변환합니다.

k차원의 벡터에서 i번째 원소를 z_i, i번째 클래스가 정답일 확률을 p_i로 나타낸다고 하였을 때, 소프트맥스 함수는 를 다음과 같이 정의합니다.

![image](https://user-images.githubusercontent.com/52392004/186542833-891b29e9-c112-42eb-ba1a-d3023753ccb5.png)

만약 k=3이라면, 소프트맥스 함수는 아래와 같은 출력을 리턴합니다. 여기서, p1은 1번 class일 확율을 나타내고, 전체 확율의 합은 1입니다.

![image](https://user-images.githubusercontent.com/52392004/186542970-f41721df-7539-4424-a922-1e375859e889.png)

[Logistric regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/logistic-regression.md)의 다중분류를 한 예로 보면, 아래와 같이 Fish 데이터가 어떤 class일 확율을 scikit-learn의 Logistic Regression을 이용하여 softmax 함수로 구하고 있음을 알수 있습니다. 


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
