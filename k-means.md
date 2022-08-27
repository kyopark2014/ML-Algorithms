# K-Means

관측한 데이터를 유사도를 기준으로 k개의 군집으로 분류하는 클러스터링 알고리즘입니다.

아래는 k=5으로 분류한 결과입니다. 

<img width="484" alt="image" src="https://user-images.githubusercontent.com/52392004/187023362-03548fc7-2535-4ca0-b567-f444dcf8c3b7.png">

## 동작

- Step1: 임의로 뽑은 K개의 데이터를 클러스터 𝑖, 𝑗 의 중심으로 지정 (𝜇_𝑖 , 𝜇_𝑗)를 구합니다.

- Step2: 각 데이터와 클러스터간 거리를 계산하여 거리가 짧은 클러스터에 데이터 할당합니다.

- Step3: 𝑥는 𝜇_𝑖에 가까우므로 클러스터 𝑖에 할당하고 클러스터의 중심을 갱신합니다.

- Step4: 모든 데이터포인트에 대해 Step2, Step3을 반복

## Elbow

Elblow Method: 클러스터 중심과 클러스터 샘플사이의 거리의 제곱합(Inertia)가 빠르게 감소하다가 감소 추이가 작아지는 지점을 최적의 클러스터 수로 결정합니다. (heuristic) 

<img width="864" alt="image" src="https://user-images.githubusercontent.com/52392004/187023432-16da2010-ad2a-45a9-9e4a-6917f5c73a88.png">

## 특징

#### 장점

- 이해하기 쉬운 Clustering 기법입니다.
- Clustering 속도가 빠름니다.
- 쉽게 구현 가능합니다.

#### 단점

- 최적의 클러스터 수를 결정해야 합니다.
- 최초 결정한 임의의 클러스터 중심의 위치에 따라 성능이 달라집니다.   
- Outliner에 민감하여 Standardization 필요합니다.

## Scikit-learn의 k-means++ 

```python
class sklearn.cluster.KMeans ( 
  n_clusters=8, 
  *, 
  init='k-means++', 
  n_init=10, 
  max_iter=300, 
  tol=0.0001, 
  precompute_distances='deprecated', 
  verbose=0, 
  random_state=None, 
  copy_x=True, 
  n_jobs='deprecated',
  algorithm='auto')
```

### 주요 Parameters

- n_clusters: 클러스터의 수. Default: 8
- init:  초기 중심점의 선택 방식 (Ramdom or k-means++). Default: k-means++
- n_init: 클러스터 중심의 초기화 횟수. Default: 10 (K-Means를 10번 수행)
- max_iter: 중심점의 최대 업데이트 수. Default: 300
- tol: tolerance. 예전 중심점과 갱신된 중심점의 차이가 tol보다 작을 경우 갱신 종료. Default: 0.0001

### 주요 Attributes
- cluster_centers: 중심점 센터의 좌표
- labels_: 각 샘플의 클러스터 id
- lnertia_: 각 샘플과 그 샘플이 속하는 클러스터의 중심까지 거리의 총합 (Elbow method에 사용)

## 코드 예제

[상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/kmeans.ipynb)을 아래에서 설명합니다.

1) 데이터를 준비합니다.

[PCA로 데이터를 축소](https://github.com/kyopark2014/ML-Algorithms/blob/main/pca.md)합니다. 

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

fruits_pca = pca.transform(fruits_2d)
print(fruits_2d.shape)
print(fruits_pca.shape)

(300, 10000)
(300, 50)
```

2) k-Means를 이용하여 아래와 같이 3개의 군집으로 분류합니다. 

아래와 같이 pineapple로 111개, banana로 98개, apple로 91개가 분류되었습니다.

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)

print(np.unique(km.labels_, return_counts=True))

(array([0, 1, 2], dtype=int32), array([111,  98,  91]))
```

이것을 보기쉽게 그리면 아래와 같습니다. 

```python
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")
```    

![image](https://user-images.githubusercontent.com/52392004/187024353-312790c5-dfca-427a-b9ad-d6183dbb4f43.png)

![image](https://user-images.githubusercontent.com/52392004/187024360-dca1cca2-6e97-436a-8e28-a6f555389ef5.png)

![image](https://user-images.githubusercontent.com/52392004/187024367-ddad9100-3ea8-4bb8-90ee-d5921bfb1bb9.png)


3) 전체적인 분포도를 그리면 아래와 같습니다.

```python
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
```

결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/187024410-475bad1d-1c7e-4785-bc8c-20ed5e0c8071.png)

k-Means는 비지도학습(Unsupervised Learning)으로 정답 labdel이 없는 데이터를 이용하여 상기와 같이 3개의 카테고리로 분류할 수 있습니다. 


## Reference

[sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)


[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
