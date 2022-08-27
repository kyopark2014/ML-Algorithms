# PCA (Principal Component Analysis)

PCA는 데이터의 분산(variance)을 최대한 보존하면서 서로 직교하는 새 기저(축)를 찾아, 고차원 공간의 표본들을 선형 연관성이 없는 저차원 공간으로 변환하는 기법입니다. 


## 특징

#### 장점

- Correlated feature를 제거할 수 있습니다.
- 차원 축소된 데이터를 학습데이터로 활용 했을 때 모델의 성능이 좋아집니다. (Overfitting 방지합니다.)
- 시각화하기 용이합니다.

#### 단점 

- PCA전 Standardization 필요합니다.
- 정보량 손실이 있습니다.
- Feature의 해석력이 약화됩니다.
- 많은 계산량을 필요로 합니다. 


## 구현 예제 

1) 이미지 데이터 로드하기

[과일사진 이미지](https://github.com/kyopark2014/ML-Algorithms/blob/main/fruits.md)를 아래처럼 로드합니다. 

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```

2) PCA를 적용합니다.

아래와 같이 PCA로 50 차원(dimensions)으로 데이터를 축소합니다. 이 경우에 아래처럼 10000이 50으로 축소가 됩니다.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

fruits_pca = pca.transform(fruits_2d)

print(fruits_2d.shape)
print(fruits_pca.shape)

(300, 10000)
(300, 50)
```

50개의 pca component를 이해를 위해 그려보면 아래와 같습니다. 

```python
import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다. 
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, 
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()
    
draw_fruits(pca.components_.reshape(-1, 100, 100))    
```    

이때 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/187012277-5abcd4e1-08af-46d8-b963-14ac4443c3a8.png)


원본으로 재구성 가능한지 아래처럼 시험하면 유사한 원본 이미지를 얻을 수 있습니다.

```python
fruits_inverse = pca.inverse_transform(fruits_pca)
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")
```

![image](https://user-images.githubusercontent.com/52392004/187012324-52e32d0c-7a30-4eec-bfc2-40a674aa20b7.png)

![image](https://user-images.githubusercontent.com/52392004/187012329-f518a486-b46b-47fd-8312-529e99876854.png)

![image](https://user-images.githubusercontent.com/52392004/187012336-a164cc2c-4396-4dce-bc34-ab72026e1938.png)


## Logistic Regression으로 분류하기 

[PCA 적용전의 fruits_2d를 가지고 Logistic regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/fruits.md#logistic-regression%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%B4-%EB%B6%84%EB%A5%98%ED%95%B4%EB%B3%B4%EA%B8%B0)을 수행한 결과는 정확도는 0.9966666666666667이고, 계산시간은 0.6095898628234864 입니다.

PCA를 이용해 마찬가지로 Logistic Regression을 수행합니다. 아래처럼 계산시간이 개선됨을 알 수 있습니다. 

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

fruits_pca = pca.transform(fruits_2d)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

target = np.array([0] * 100 + [1] * 100 + [2] * 100)

from sklearn.model_selection import cross_validate

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

1.0
0.03617258071899414
```



## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
