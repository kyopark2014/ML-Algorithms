# 인공신경망 설계

## 지도학습 문제유형 

### Regression

Regression 모델에서 값의 범위는 -inf레서 inf이어야 하므로, 출력층(Output)에서는 activation function을 제거하여 net값이 출력이 되도록 합니다. 이와같이 출력레이어에서 activation function없이 사용하는것을 linear로 사용했다고 얘기합니다. 

![image](https://user-images.githubusercontent.com/52392004/187060927-41d2dc0e-fc4d-4e91-b975-9648ae5c9328.png)

### Two Class Classification

아래와 같이 입력값에 2개의 label을 가지는 모델을 만들고자 합니다. 

<img width="454" alt="image" src="https://user-images.githubusercontent.com/52392004/187060971-1e7f7a6b-2a58-45b9-ae1d-3091c97a7832.png">

이때, Red는 1, Black은 0으로 치환할 수 있는데, activation function으로 sigmoid를 사용하여 구현할 수 있습니다. 이때의 loss function은 cross entropy를 사용하면 됩니다. 

<img width="317" alt="image" src="https://user-images.githubusercontent.com/52392004/187061056-2fb47fbe-0865-45ec-8884-5f5123feb154.png">

판별기준을 아래와 같이 출력값이 0.5보다 큰 경우에 1 (Red)이고 작은 경우에 0 (Black)으로 표현할 수 있습니다.

![image](https://user-images.githubusercontent.com/52392004/187061053-0ae1250d-09b6-4854-bd1e-ac4cedde5301.png)



