# 신경망 (Neural Network)

신경망은 다중 퍼셉트론이라고 불리며, [퍼셉트론](https://github.com/kyopark2014/ML-Algorithms/blob/main/perceptron.md)을 하나의 노드로 하여 계층적으로 쌓아 놓은 구조입니다. 뇌의 신경세포(뉴런)들이 시냅스로 연결되어 전기신호를 통해 정보를 주고받는 모습에서 착안하였기 때문에 이런 이름이 붙였다고 합니다. 

## Perceptron

Artificial Neural Network의 기본인 [Perceptron의 선형분류모델과 Multi Layer Perceptron](https://github.com/kyopark2014/ML-Algorithms/blob/main/perceptron.md)에 대해 설명합니다. 

## Neural Network 특징 

- 신경망 모델은 네트워크로 표현된 프로그램입니다.
- 각 노드의 Net값은 각 입력(Input)이 가중치(Weight)합으로 표현합니다.
- 각 노드의 출력(Output)은 노드의 Net값에 Activation Function을 적용한 값입니다.
- 각 노드의 출력은 다음 레이어의 입력입니다.
- 레이어를 추가하여 비선형 모델링이 가능합니다.
- 문제의 복잡도가 증가함에 따라 필요한 노드의 개수도 증가합니다.
- Overfitting 보다는 Generalized 된 모델이 좋습니다. Tranin dataset에 많아질수록 Loss 함수가 작아지는데, validation dataset의 Loss가 줄다가 증가하기 시작하면 모델학습을 중지합니다. 

### Shallow Network

노드의 개수가 늘어나면 가중치 개수가 기하급수적으로 늘어나게 됩니다. 

<img width="255" alt="image" src="https://user-images.githubusercontent.com/52392004/187060127-33725f3a-40d1-4842-b36c-913a0e67430e.png">

### Deep Network

- 가중치의 개수가 급증하는 것을 피할 수 있습니다.
- 보다 복잡한 문제를 풀 수 있습니다. 층이 늘어날수록 비선형이 늘어나면서 복잡한 문제도 풀수 있습니다. (Fine Transform)
- 최적화하기 어렵고, 과적합되기 쉽고, [내부공변량 변화 (Internal Covariate Shift)](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md#batch-normalization)가 커서 학습하기 어려습니다.

<img width="574" alt="image" src="https://user-images.githubusercontent.com/52392004/187060185-99e36ba2-f90d-4b51-91c8-e0f8ddcb05b8.png">



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


## 모델 학습 방법 

- 뉴런 개수 최적화
- 훈련 반복횟수 최적화
- 정규화 방법 사용
- 트레이닝 데이터 확보

아래는 itteration을 통해 Error의 변화를 보여줍니다. Train dataset의 에러는 계속 주는데, Validation dataset은 어느정도까지만 줄어듭니다. (Overfitting)

![image](https://user-images.githubusercontent.com/52392004/187053619-9ad1fe1c-a4b8-469c-b32d-0cfebb277e25.png)

## Neural Network 설계 방법

[Neural Network 설계](https://github.com/kyopark2014/ML-Algorithms/blob/main/neural-network-design.md)에서는 Regression, Two class classification, Multi class classification에 대해 설명합니다. 


## Reference 

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[Machine Learning at Work - 한빛미디어]
