# Logistic Regression

여기의 데이터는 [혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)을 이용합니다. [상세한 코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/logistic_regression.ipynb)에서 아래의 예제를 jupyter notebook으로 확인할 수 있습니다.

표준점수(z)는 "z = a * 무게 + b * 길이 + c * 두께 + d * 대각선 + e * 높이 + f"와 같이 표현됩니다. 


## 데이터 준비

아래와 같이 pandas를 통해 fish data를 로딩합니다.

```python
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
```

아래와 같은 fish 데이터를 로드하였습니다.

![image](https://user-images.githubusercontent.com/52392004/186283925-8861fcfc-2d94-43ee-a61d-404e44e1baba.png)

이때 읽어온 fish 데이터는 "Bream", "Roach", "Whitefish", "Parkki", "Perch", "Pike", "Smelt"이 있습니다.

Numpy포맷으로 변환합니다. 

```python
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
print(fish_input[:5])

[[242.      25.4     30.      11.52     4.02  ]
 [290.      26.3     31.2     12.48     4.3056]
 [340.      26.5     31.1     12.3778   4.6961]
 [363.      29.      33.5     12.73     4.4555]
 [430.      29.      34.      12.444    5.134 ]]
 
fish_target = fish['Species'].to_numpy()
print(fish_target)

['Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream'
 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream'
 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream'
 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Bream' 'Roach'
 'Roach' 'Roach' 'Roach' 'Roach' 'Roach' 'Roach' 'Roach' 'Roach' 'Roach'
 'Roach' 'Roach' 'Roach' 'Roach' 'Roach' 'Roach' 'Roach' 'Roach' 'Roach'
```

Train, Test Set을 준비합니다.

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)
```    

StandardScaler을 이용하여 정규화를 합니다.

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

## K 최근접 분류

Logistic regression과 비교하기 위하여 K 최근접 분류를 수행해 보았습니다. 

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

0.8907563025210085
0.85
```

## 이진 분류

이진분류에서는 표준점수(z)을 확율로 바꾸기 위하여 Sigmoid 함수를 사용합니다. 

데이터를 준비하고, 이진 로지스틱 회귀를 수행합니다. 

```python
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(kn.score(train_bream_smelt, target_bream_smelt))
0.9696969696969697
```

아래와 같이 분류 항목을 classes로 확인하고, 계수(coeffcient)들과 절편(intercept)을 확인할 수 있습니다. 이때, Bream, Smelt의 계산된 값을 proba로 찍어보면 0-1의 확률로 아래와 같이 계산됨을 알 수 있습니다. 

```python
print(lr.classes_)
print(lr.coef_, lr.intercept_)
print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))

['Bream' 'Smelt']
[[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']
[[0.99759855 0.00240145]
 [0.02735183 0.97264817]
 [0.99486072 0.00513928]
 [0.98584202 0.01415798]
 [0.99767269 0.00232731]]
```

표준점수를 decision_function으로 구하고, 이를 Sigmoid 함수에 해당하는 expit()로 구하면, 상기의 확율과 같은 값임을 알수 있습니다. 

```python
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit
print(expit(decisions))

[-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]
[0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]
```

## 다중 분류 

표준점수(z)을 확율로 바꾸기 위하여 Softmax 함수를 사용합니다. 다중분류를 쓰는 로지스틱 회귀에서는 C를 이용해 규제 (L2 규제를 기본적용)를 하는데, C 값이 클수록 규제가 약해집니다. 아래와 같이 다중분류로 Logistric regression을 수행합니다. K 최근접 분류보다 좋은 결과를 얻고 있습니다. 

```python
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

0.9327731092436975
0.925
```

이때 내부에서 게산된 값을 아래와 같이 predict와 predict_proba로 확인 할 수 있습니다. 

```python
print(lr.classes_)
print(lr.coef_.shape, lr.intercept_.shape)

proba = lr.predict_proba(test_scaled[:5])

import numpy as np
print(np.round(proba, decimals=3))

['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
(7, 5) (7,)
[[0.    0.014 0.841 0.    0.136 0.007 0.003]
 [0.    0.003 0.044 0.    0.007 0.946 0.   ]
 [0.    0.    0.034 0.935 0.015 0.016 0.   ]
 [0.011 0.034 0.306 0.007 0.567 0.    0.076]
 [0.    0.    0.904 0.002 0.089 0.002 0.001]]
```

마찬가지로 표준점수를 계산하고, softmax로 확율을 계산하면 predict_proba로 얻어진 결과와 같습니다. 

```python
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

from scipy.special import softmax

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))

[[ -6.5    1.03   5.16  -2.73   3.34   0.33  -0.63]
 [-10.86   1.93   4.77  -2.4    2.98   7.84  -4.26]
 [ -4.34  -6.23   3.17   6.49   2.36   2.42  -3.87]
 [ -0.68   0.45   2.65  -1.19   3.26  -5.75   1.26]
 [ -6.4   -1.99   5.82  -0.11   3.5   -0.11  -0.71]]
 
[[0.    0.014 0.841 0.    0.136 0.007 0.003]
 [0.    0.003 0.044 0.    0.007 0.946 0.   ]
 [0.    0.    0.034 0.935 0.015 0.016 0.   ]
 [0.011 0.034 0.306 0.007 0.567 0.    0.076]
 [0.    0.    0.904 0.002 0.089 0.002 0.001]]
```

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
