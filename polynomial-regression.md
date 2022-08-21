# 다항회귀 (Polynomial Regression)

## 농어의 길이를 무게로 예측 

1) 아래와 같이 데이터를 준비합니다. 

```python
import numpy as np

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
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
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

# 훈련 세트와 테스트 세트를 2차원 배열로 바꿉니다
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

2) 아래와 같이 feature engineering을 합니다. 

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=5, include_bias=False)

poly.fit(train_input)

train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
```

3) Linear regression을 수행합니다. 

```python
lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]]))
````

이때의 50cm인 농어의 예측 무게는 [1573.98423528]으로 [linear regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/linear-regression.md)에서 얻어진 값보다 훨씬 큰값으로 예측됩니다.  


4) 그래프인 y = ax^2 + bx + c에 대한값은 아래와 같이 구할 수 있습니다.

```python
print(lr.coef_, lr.intercept_)
a = lr.coef_[0]
b = lr.coef_[1]
c = lr.intercept_
```

5) 그래프를 그려보면 아래와 같습니다. 

```python
import matplotlib.pyplot as plt

# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다
point = np.arange(15, 50)

plt.scatter(train_input, train_target)
plt.plot(point, a*point**2 + b*point + c)

# 50cm 농어 데이터
plt.scatter([50], [1574], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

얻어진 그래프는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/185797100-ea3dfb8e-03a7-4317-8729-e406c14d1248.png)


## 결과 

아래와 같이 

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))



## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
