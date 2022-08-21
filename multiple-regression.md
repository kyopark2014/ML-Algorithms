# 다중회귀(Multiple Regression)와 특성공학(Feature Engineering)


## 특성공학을 이용한 농어 예측

1) [Polynomial Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/polynomial-regression.md)에서 이용한 데이터를 로드하고, train/test set을 분리합니다. 

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
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
```

2) feature engineering을 통해 기존 특성에 새로운 특성을 추가합니다. 

```java
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)  # default degree=2

poly.fit(train_input)

train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
```

3) StandardScaler을 이용하여 표준점수로 변환합니다.

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```

4) 과대적합을 방지하기 위하여 규제(Reguarization)중에 Ridge를 적용합니다. 

Ridge로 결정계수를 구하면 아래와 같습니다. Ridge는 계수를 제곱한 값을 기준으로 규제를 적용합니다. 

```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
```

이때 얻어진 결정계수는 [linear regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/linear-regression.md)보다 개선됩니다. 

```java
0.9857915060511934
0.9835057194929057
```

Ridge의 alpha를 이용하여 규제의 강도를 조절 할수 있습니다. alpha값이 크면 규제의 강도가 커지는데, 계수값을 더 줄이고 좀 더 과대적합을 해소할 수 있습니다. 

```python
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델을 만듭니다
    ridge = Ridge(alpha=alpha)
    # 릿지 모델을 훈련합니다
    ridge.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
```

이것을 그래프로 그려보면 아래와 같습니다. 

```python
import matplotlib.pyplot as plt

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

이때 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/185815725-f43ed47d-c08c-424a-ba3f-02807631e91b.png)


5) 과대적합을 방지하기 위하여 Lasso를 적용하여 Ridge와 비교해 봅니다. Lasso는 계수의 절대값을 기준으로 규제를 적용하는데, 계수를 0으로 만들 수도 있습니다. 

```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```

이때, 얻어진 결정계수는 아래와 같이 과대적합이 아닙니다. 

```java
0.986591255464559
0.9846056618190413
```

Ridge처럼 Rasso도 alpha로 규제의 강도를 조절할 수 있습니다. 

```python
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 라쏘 모델을 만듭니다
    lasso = Lasso(alpha=alpha, max_iter=10000)
    # 라쏘 모델을 훈련합니다
    lasso.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

import matplotlib.pyplot as plt
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

이때 얻어진 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/185816002-0ed4e806-536d-43ee-800e-73ec07c96f25.png)


## 다중회귀 (Multiple Regression)

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)에는 여러개의 특성(길이, 두께, 높이)을 사용하여 예측할 때 쓸수 있는데 다중회귀 예제를 아래처럼 제공하고 있습니다. 

데이터를 아래와 같이 pandas로 읽어올 수 있습니다. [CSV 파일](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/perch_full.csv)에는 length, height, width로 도미(perch) 데이터가 정리되어 있습니다.

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

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
