# 과일사진 이미지 

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)의 과일사진을 로드하고 구조를 확인하고자 합니다. [상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/fruits.ipynb)에 아래의 Sample Code가 있습니다. 

아래와 같이 과일사진인 fruits_300.npy를 로드하면 100x100 크기의 이미지 300개를 로드할 수 있습니다. 

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

import numpy as np
fruits = np.load('fruits_300.npy')
print(fruits.shape)

(300, 100, 100)
```

해당 이미지 셈플은 'apple', 'pineapple', 'banana' 입니다. 

```python
import matplotlib.pyplot as plt

plt.imshow(fruits[0], cmap='gray_r')
plt.show()

fig, axs = plt.subplots(1, 3)
axs[0].imshow(fruits[0], cmap='gray_r')
axs[1].imshow(fruits[100], cmap='gray_r')
axs[2].imshow(fruits[200], cmap='gray_r')
plt.show()
```

![image](https://user-images.githubusercontent.com/52392004/187010777-a14d0388-9b27-49ef-a6e2-3ccdbdcd79bc.png)

이미지 전체를 그려보면 아래와 같습니다. 

```python
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[i*10 + j], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()
```

![image](https://user-images.githubusercontent.com/52392004/187010803-545c5f99-d171-4130-9991-71c9f90d5a84.png)

```python
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[i*10 + j + 100], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()
```

![image](https://user-images.githubusercontent.com/52392004/187010825-d82577fc-f683-4929-84e6-0b798d320cc6.png)

```python
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[i*10 + j + 200], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()
```

![image](https://user-images.githubusercontent.com/52392004/187010844-d4d1b030-6933-422d-81d7-693d827d5821.png)

## 이미지 평균

```python
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()
```

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/187010891-f5e441e1-bf73-4dea-8852-010a9acf659e.png)


## Fruit에서 바나나와 평균값이 비슷한 이미지를 찾기 

```python
abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))

banana_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[banana_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()
```

아래와 같이 일부 사과 이미지가 바나나로 분류가 됩니다. 단순히 이미지 평균값의 차이만 가지고도 어느정도 분류가 가능합니다.

![image](https://user-images.githubusercontent.com/52392004/187010940-c59b491e-66cc-4eb0-a674-18979e5950b9.png)


## Logistic Regression을 이용해 분류해보기 

dataset을 train, test으로 분리후 logistic regression으로 분류시에 아래처럼 매우 좋은 결과를 얻습니다. 

```python
import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
target = np.array([0] * 100 + [1] * 100 + [2] * 100)

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fruits_2d, target, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(train_input, train_target)

print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

1.0
0.9833333333333333
```

[교차검증(Cross Validation)](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md#k-fold-cross-validation%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D)을 이용해서 구하면 조금 더 정확한 값을 구할수 있습니다. 

```python
from sklearn.model_selection import cross_validate

scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

0.9966666666666667
0.6095898628234864
```



## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
