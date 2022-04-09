## kNN Classification

## How KNN algrorithm works with example : K - Nearest Neighbor
https://www.youtube.com/watch?v=2YQHPfwVuF8

#### Lazy learner 방식

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



