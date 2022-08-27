# PCA(Principal Component Analysis)

PCA는 데이터의 분산(variance)을 최대한 보존하면서 서로 직교하는 새 기저(축)를 찾아, 고차원 공간의 표본들을 선형 연관성이 없는 저차원 공간으로 변환하는 기법입니다. 

## 이미지 데이터 로드하기

[과일사진 이미지](https://github.com/kyopark2014/ML-Algorithms/blob/main/fruits.md)를 아래처럼 로드합니다. 

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```


https://github.com/kyopark2014/ML-Algorithms/blob/main/fruits.md#logistic-regression%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%B4-%EB%B6%84%EB%A5%98%ED%95%B4%EB%B3%B4%EA%B8%B0

0.9966666666666667
0.6095898628234864

PCA를 이용해 

from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

