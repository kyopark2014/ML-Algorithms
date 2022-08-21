# Linear Regression

여기에서는 [혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)에 있는 농어의 길이/무게 데이터를 가지고 길이에 대한 무게를 예측하는것에 대해 설명합니다. 자세한 코드는 [상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/linear_regression.ipynb)을 참조합니다. 


## 동작 설명 

1) 아래와 같이 데이터를 입력합니다. 

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
```     

2) Train과 Test Set을 만듧니다.

```python
from sklearn.model_selection import train_test_split

# 훈련 세트와 테스트 세트로 나눕니다
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

# 훈련 세트와 테스트 세트를 2차원 배열로 바꿉니다
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

3) Linear Regression으로 모델을 생성합니다.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()  # 선형 회귀 모델 훈련
lr.fit(train_input, train_target)
```


4) 길이가 50cm인 농어의 무계를 예측합니다. 

```python
print(lr.predict([[50]]))
```

이때의 결과는 아래와 같습니다. 

```python
[1241.83860323]
```

5) 그래프를 그리면 아래와 같이 데이터 영역 밖의 값을 예측할때에 Linear Regression으로 예측한것을 알 수 있습니다. 

```python
import matplotlib.pyplot as plt

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)
# 15에서 50까지 1차 방정식 그래프를 그립니다
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_],'r')
# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

이때의 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/185791485-e57c716a-31fc-4453-a76d-f9d0aa93f6bb.png)


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
