# Neural Network

## Perceptron

Artificial Neural Network의 기본인 [Perceptron의 선형분류모델과 Multi Layer Perceptron](https://github.com/kyopark2014/ML-Algorithms/blob/main/perceptron.md)에 대해 설명합니다. 

## Neural Network 학습방법 

#### 학습준비물

- 입력값과 출력값
- 네트워크구조: Node의 개수와 Hidden layer의 개수 등
- 랜덤하게 초기화된 가중치 (Weight)

![image](https://user-images.githubusercontent.com/52392004/187053154-be27fb5d-8321-4fb7-aeeb-c2805ebd6093.png)

### 문제의 정의 

![image](https://user-images.githubusercontent.com/52392004/187053314-5ce8d8cc-8569-49a7-9cc7-5a3793938eb2.png)

여기서 에러는 아래와 같이 적의 할 수 있습니다. 이것은 MSE Loss를 의미합니다. 

![image](https://user-images.githubusercontent.com/52392004/187053252-e341f296-587f-4947-8fdb-c4ea75664848.png)

1957년에는 Error 함수를 구할 수 없었으나, Paul Werbos (1974년), Geoffrey Hinton (1986년)에 의해 Error Back Propagation을 통해 해결할 수 있게 됩니다. 

### Error Back Propagation

- Neural Network을 학습시키기 위해 Gradient Descent 모델을 사용합니다.
- Error Function의 기울기를 구하여 Weight를 업데이트 하는 과정이 에러값을 역으로 전파하는것과 같다고하여 Error Back Propagation 기법이라고 부릅니다.
- 미분을 하기 위해 Activation 함수를 Hard Limit 대신에 Sigmoid를 사용합니다. 
- Feed forward와 Back progration을 반복함으로써 최적의 Weight 값을 찾을 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/187053501-b317063c-fec9-4f63-8593-c865d521be64.png)





