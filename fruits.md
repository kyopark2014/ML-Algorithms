# 과일사진 이미지 

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)의 과일사진을 로드하고 구조를 확인하고자 합니다. 

과일사진인 fruits_300.npy를 로드하면 100x100 크기의 이미지 300개를 로드할 수 있습니다. 

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

import numpy as np
fruits = np.load('fruits_300.npy')
print(fruits.shape)

(300, 100, 100)
```

해당 이미지 셈플은 사과, 파인에플, 바나나 입니다. 

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

일부 사과 이미지가 바나나로 분류가 됩니다. 

![image](https://user-images.githubusercontent.com/52392004/187010940-c59b491e-66cc-4eb0-a674-18979e5950b9.png)




## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
