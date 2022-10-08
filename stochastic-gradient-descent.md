# 확률적 경사 하강법 (Stochastic Gradient Descent)

Loss를 줄이기 위해서 반복적으로 기울기를 계산하여 변수의 값을 변경해 나가는 과정을 수행하여 정확도를 높입니다. [gradient-descent.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/gradient-descent.ipynb)의 물고기 데이터에 확율적 경사하강법을 적용한 예를 아래에서 설명합니다. 

## Gradient Descent

경사하강법(Gradient Descent)은 오차를 최소화하기 위한 방법입니다. 

아래와 같이 경사하강법을 수행할 수 있습니다. 

1) 임의의 시작점을 Random하게 선택합니다.
2) 현재 위치의 경사를 구합니다.
3) 기울기의 반대방향으로 조금 이동합니다.
4) 기울기가 0인 곳에 도달 할 때 까지 반복합니다. 

Error 값은 아래처럼 MSE로 정의할 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/187074635-7f25adde-6ce3-42ac-b7b5-d78a2efa2948.png)

Weight는 아래처럼 Error의 미분으로 표현됩니다. 

![image](https://user-images.githubusercontent.com/52392004/187074671-d487c330-6f1e-4745-985e-5f3c813799a6.png)


이때의 Error값은 아래처럼 변화시키면서 Weight의 Error를 개선합니다. 

![image](https://user-images.githubusercontent.com/52392004/187074618-9b5ab505-fd28-4de7-a75f-4af062c7e95c.png)


아래 그림은 gradient descent를 최적화하는 예제입니다. 

![image](https://user-images.githubusercontent.com/52392004/193482799-0e09acbe-1abd-4ebe-bc64-0179709ff158.png)


## Gradient vanishing & exploding

(t+1)의 Weight는 이전 Weight에서 learning rate와 gradient의 곲을 뺀것입니다. 

![image](https://user-images.githubusercontent.com/52392004/187074828-fe5c8079-49fa-4cfe-bad3-56f35a1231a9.png)

[activation function](https://github.com/kyopark2014/ML-Algorithms/blob/main/perceptron.md#%ED%99%9C%EC%84%B1%ED%95%A8%EC%88%98-activation-function)으로 signoid를 쓰면 미분값이 0.25가 됩니다. 

![image](https://user-images.githubusercontent.com/52392004/187074885-56f2531e-628a-4b55-8170-7714a17a011e.png)

[Deep network](https://github.com/kyopark2014/ML-Algorithms/blob/main/neural-network.md#deep-network)와 같이 layer가 증가하면, weight의 변화가 줄어서 더이상 학습이 안되어버리므로 signoid를 activation function으로 사용할 수 없게 됩니다. 이것을 Gradient vanishing problem이라고 합니다. 

![image](https://user-images.githubusercontent.com/52392004/187074794-34a16dd4-83e4-4d23-9f8f-9cac583b8133.png)

### ReLU (Rectified Linear Unit)

Activation function으로 [ReLU (Rectified Linear Unit)](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md#relu)를 사용하여 Gradient vanishing 문제를 해결할 수 있습니다. 


## 적용 예

1) 데이터를 준비합니다. 여기서 얻어온 데이터는 [혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)를 참조합니다.

```python
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)
    
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)    
```

2) 아래와 같이 scikit-learn의 SGDClassifier를 이용하여 결정계수를 구하면 아래와 같습니다. 

```python
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', random_state=42)

train_score = []
test_score = []

import numpy as np
classes = np.unique(train_target)

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

import matplotlib.pyplot as plt

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```

이때 결과는 아래와 같습니다. 여기서 iteration은 epoch를 의미하는데, Test set에 대한 정확도는 epoch가 100이상에서는 동일함을 알 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/186546899-321e805b-e5a1-4e2e-8ce9-150189f3402f.png)

아래와 같이 epoch가 100인 경우에 대해서 결정계수를 구하면 아래와 같이 구할 수 있습니다. 

```python
sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

0.957983193277311
0.925
```

손실함수를 SGDClassifier의 기본값인 hinge를 사용하였을때도 동일한 결과를 얻을 수 있습니다. 

```python
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

0.9495798319327731
0.925
```

손실함수를 hinge로 했을때에 epoch의 증가에 따른 정확도 그림은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/186547410-282dee58-c8eb-422f-a398-4f66ed2444d5.png)


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)


[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)


