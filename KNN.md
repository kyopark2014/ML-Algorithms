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




