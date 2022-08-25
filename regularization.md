# 규제 (Regularization)


모델이 과적합 되게 학습하지 않고 일반성을 가질 수 있도록 파라미터값에 제약을 주는것을 말합니다. L1 규제(Lasso), L2 규제(Ridge), alpha 값으로 규제량을 조정합니다. 

<img width="423" alt="image" src="https://user-images.githubusercontent.com/52392004/185773329-8b542165-3c41-42d9-ba0f-e437a2f9f811.png">


- Ridge: 계수를 제곱한 값을 기준으로 규제를 적용

```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
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
```

아래는 릿지 회귀의 한 예입니다. 

<img width="284" alt="image" src="https://user-images.githubusercontent.com/52392004/185773607-69cefcfb-e931-47c6-b9ff-6f2045015674.png">


- Lasso: 계수의 절대값을 기준으로 규제를 적용 

```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
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
```
