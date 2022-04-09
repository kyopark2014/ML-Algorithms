# K - Nearest Neighbor(kNN) Classification

## Machine Learning at Work (O'REILLY)

. 새로운 데이터의 클래수를 다수결로 결정한다. 학습된 데이터 중 새로 입력된 데이터와 거리가 가장 가까운 k개를 선택하고, k개 데이터를 확인해 가장 많은 데이터가 속한 클래스를 찾는다. 그리고 이 클래스를 새로운 데티어의 클래스로 삼는다. 원리가 단순하여 그대로 사용하는 경우도 많고, 비슷한 항목을 탐색하는데 사용되기도 한다. 

### kNN의 특징 

. 데이터를 하나씩 순차적으로 학습한다.

. 기본적으로 모든 데이터와의 거리를 계산해야 하므로 예측에 시간이 걸린다.

. k값에 따라 편차는 있지만 예측 성능은 괜찮은 편이다. 

. 데이터의 스케일을 맞추기 위해 normalization이 필요할 수 있다. 
 

. hyperparameter인 k값 설정이 결정경계(Decision boundary)를 좌우한다. k값이 커질수록 경정경계가 매끈해지는 경향이 있으나 처리시간이 길어진다. 

. 두 데이터 사이의 멀고 가까움을 측정하는 거리 측정 알고리즘에 같은 클래스에 속하는 데이터들의 평균을 이용하는 Euclidean distance가 있고, 데이터가 분포하는 방향을 고려하는 Mahalanobis distance를 사용하기도 한다. 

. 자연어 처리처럼 높은 차원의 희소한 데이터를 다루는 경우에는 성능이 잘 나오지 않는다. 

. 차원 축소 기법을 이용하여 차원을 줄여주면 성능이 개선되기도 한다. 

. elastic search의 점수를 distance로 간주하여 kNN을 적용하기도 한다. 

. 계산 시간이 오래 걸리는 문제를 해결하기 위해, 근사적인 이웃탐색 방법을 쓰기도 한다. 




## How KNN algrorithm works with example : K - Nearest Neighbor
https://www.youtube.com/watch?v=2YQHPfwVuF8

### Lazy learner 방식

. Training dataset is stored

. On Querying similarity between test data and training set recodes is calculated to predict the class of test data

. kNN


### k-NN 알고리즘 특징 
. Non-parametric method used for classification

. Prediction for test data is done on the basis of its neighbour

. k is an interger (small), if k=1, k is assigned to the class of single nearest neighbour


### Example 

1) 문제 

acid durability = 3, strength = 7인 셈플이 Good/Bad인지 찾으려고 함 

![image](https://user-images.githubusercontent.com/52392004/162555422-93ffc044-712b-4471-af1b-f6a9d873239e.png)

2) Similarity를 Euclidean으로 정의 

![image](https://user-images.githubusercontent.com/52392004/162555450-d0e531ad-2617-46c0-9a63-584ca2a6ac37.png)

3) 계산된 Distance 

![image](https://user-images.githubusercontent.com/52392004/162555465-9e9c3272-4fd1-4ebc-a7b4-83544386797a.png)

Type2:  sqrt((7-3)^2 + (4-7)^2) = sqrt(16+9) = 5

4) k=1인 경우 

아래와 같이 가장 가까운 Type3가 Good이므로 Good으로 예측 

![image](https://user-images.githubusercontent.com/52392004/162555514-1f2c2d96-d543-41db-88d3-28d85ef3970a.png)


5) k=2인 경우 

아래와 같이 Type3, Type4가 가까운 셈플인데 둘다 Good 이므로 Good임 

![image](https://user-images.githubusercontent.com/52392004/162555585-3e4ba508-7516-4c44-bd83-413c3dc3e12e.png)

6) k=3인 경우 

아래와 같이 2개가 Good이고 한개가 Bad이므로 Good으로 예측 

![image](https://user-images.githubusercontent.com/52392004/162555608-c0022169-a8cd-43ec-9b23-588c298ce083.png)




## 머신 러닝, KNN 알고리즘으로 간단히 맛보기
https://www.youtube.com/watch?v=g5CXjAjIKoE

타이타닉 탑습자의 나이와 티켓값으로 생존/사망 여부에 대한 데이터가 아래와 같이 있다고 가정합니다. 

![image](https://user-images.githubusercontent.com/52392004/162555658-b937b42e-1ddf-47a3-ba12-001be845a535.png)

노란색으로 표시된 위치의 탑승자의 생존/사망 여부를 예측하고자 합니다. 

![image](https://user-images.githubusercontent.com/52392004/162555693-b52028c5-011d-408d-a092-13a66977ac1f.png)

k=5인 경우에 노란색과 가장 가까운 5개를 뽑아서 생존이 더 많으므로 생존할 가능성이 많다고 생각할 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/162555728-2666aa43-0077-40bb-898c-e9d5d0d56659.png)

아래와 같이 하단의 탑승자가 있을때 k=3으로 예측한다면 사망이 더 많으므로 사망으로 예측합니다. 

![image](https://user-images.githubusercontent.com/52392004/162555737-c0bfcece-b58f-4e27-a79f-220221a6cf3c.png)

데이터가 많을 수록 정확도 높아지며, 많은 경험이 성능 향상으로 이어지므로 머신러닝으로 볼수 있다고 합니다. 



