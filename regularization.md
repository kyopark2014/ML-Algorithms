# 규제 (Regularization)


모델이 과적합 되게 학습하지 않고 일반성을 가질 수 있도록 파라미터값에 제약을 주는것을 말합니다. L1 규제(Lasso), L2 규제(Ridge), alpha 값으로 규제량을 조정합니다. 

<img width="423" alt="image" src="https://user-images.githubusercontent.com/52392004/185773329-8b542165-3c41-42d9-ba0f-e437a2f9f811.png">

상세 예제는 [다중회귀(Multiple Regression)](https://github.com/kyopark2014/ML-Algorithms/blob/main/multiple-regression.md)을 참조합니다.

- Ridge: 계수를 제곱한 값을 기준으로 규제를 적용합니다. 

```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

0.9894023149360563
0.9853164821839827
```

alpha를 매개변수로 규제의 강도를 조절할 수 있습니다. 이때, alpha값이 크면 규제 강도가 세짐으로 계수값을 더 줄이고 좀 더 과대적합 해소가 가능합니다. 

```python
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델을 만듭니다
    ridge = Ridge(alpha=alpha)
    # 릿지 모델을 훈련합니다
    ridge.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

import matplotlib.pyplot as plt

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

이때의 결정계수는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/186549328-8373f016-8cf1-4099-9bca-8f11ed8359fc.png)

alpha=1일때의 결정게수를 아래처럼 구할 수 있습니다. 

```python
ridge = Ridge(alpha=1)
ridge.fit(train_poly, train_target)

print(ridge.score(train_poly, train_target))
print(ridge.score(test_poly, test_target))

0.9894023149360563
0.9853164821839827
```


아래는 릿지 회귀의 한 예입니다. 

<img width="284" alt="image" src="https://user-images.githubusercontent.com/52392004/185773607-69cefcfb-e931-47c6-b9ff-6f2045015674.png">


- Lasso: 계수의 절대값을 기준으로 규제를 적용합니다.

```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

0.9886724935434511
0.9851391569633392
```

alpha를 매개변수로 규제의 강도를 조절할 수 있습니다. 이때, alpha값이 크면 규제 강도가 세짐으로 계수값을 더 줄이고 좀 더 과대적합 해소가 가능합니다. 

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
    
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()    
```

이때 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/186549504-d05a8c42-2bdd-45ea-b6b5-5367241d1e00.png)

릿지처럼 동일하게 alpha=1일때를 구하면 아래와 같습니다. 

```python
lasso = Lasso(alpha=1, max_iter=10000)
lasso.fit(train_poly, train_target)

print(lasso.score(train_poly, train_target))
print(lasso.score(test_poly, test_target))

0.9889627859751756
0.9857778187186969
```
