# 다중회귀(Multiple Regression)

## 다중회귀를 이용한 농어 예측

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)에는 여러개의 특성(길이, 두께, 높이)을 사용하여 예측할 때 쓸수 있는데 다중회귀 예제를 아래처럼 제공하고 있습니다. 

[Multiple Regression 상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/multiple_regression.ipynb)에서는 농어의 무게를 예측하는 예제입니다. 


1) 데이터를 아래와 같이 pandas로 읽어올 수 있습니다. [CSV 파일](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/perch_full.csv)에는 length, height, width로 도미(perch) 데이터가 정리되어 있습니다.

```python
import pandas as pd

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)
```

이때 읽어진 데이터의 형태는 아래와 같습니다. 

```java
[[ 8.4   2.11  1.41]
 [13.7   3.53  2.  ]
 [15.    3.82  2.43]
 [16.2   4.59  2.63]
 [17.4   4.59  2.94]
 [18.    5.22  3.32]
 [18.7   5.2   3.12]
 [19.    5.64  3.05]
```

2) Weight에 대한 데이터를 준비하고, Train/Test Set을 준비합니다.

```python
import numpy as np
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )

from sklearn.model_selection import train_test_split

# 훈련 세트와 테스트 세트로 나눕니다
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42)
```

3) 다중회귀를 수행합니다. 

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)  # default degree=2

poly.fit(train_input)

train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
```

4) Ridge로 규제를 수행합니다.

```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_poly, train_target)
print(ridge.score(train_poly, train_target))
print(ridge.score(test_poly, test_target))
```

이때의 결과는 아래와 같습니다. 

```c
0.9894023149360563
0.9853164821839827
```

alpha로 규제롤 조정합니다. 

```python
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델을 만듭니다
    ridge = Ridge(alpha=alpha)
    # 릿지 모델을 훈련합니다
    ridge.fit(train_poly, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(ridge.score(train_poly, train_target))
    test_score.append(ridge.score(test_poly, test_target))

import matplotlib.pyplot as plt

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/186055845-bc87bbce-3feb-4d43-9a9d-4b56992b63b5.png)

5) Lasso로 규제를 수행합니다.

```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_poly, train_target)
print(lasso.score(train_poly, train_target))
print(lasso.score(test_poly, test_target))
```

결과는 아래와 같습니다. 

```c
0.9886724935434511
0.9851391569633392
```

alpha로 규제의 정도를 조정해봅니다.

```python
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 라쏘 모델을 만듭니다
    lasso = Lasso(alpha=alpha, max_iter=10000)
    # 라쏘 모델을 훈련합니다
    lasso.fit(train_poly, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(lasso.score(train_poly, train_target))
    test_score.append(lasso.score(test_poly, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/186056113-2e16cb7d-84c0-4d25-998e-7b106b932d0d.png)


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
