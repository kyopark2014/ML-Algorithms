# Perceptron

## 선형분류모델

뉴런의 수학적모델을 위해 Rosenblatt이 1957년에 개발한 **선형분류모형**로서 인식의 기본단위를 의미합니다. 

아래에서 x1, x2, 1은 입력으로서, 가중치 w1, w2를 가지고 있습니다. 여기서, 가중치는 수상돌기의 발달정도를 의미하고, [활성함수(Activation Function)](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md) f에 의해 어떤 임계치를 넘으면 출력으로 y를 만들게 됩니다. y의 범위는 0에서 1인데, 0보다 큰 값을 받으면 1을 출력하므로 hard limit이라고 불립니다. 

![image](https://user-images.githubusercontent.com/52392004/187052605-4935035d-5faf-4a66-b326-87affa297063.png)

연속된 perceptron은 아래와 같은 형태로 구성되는데, 이를 Artificial Neural Network라고 부릅니다. 

![image](https://user-images.githubusercontent.com/52392004/187052824-3ce286c3-a2dd-498e-8396-12d9aca31455.png)

선형분류모델은 아래처럼 표시될수 있는데, 입력값이 직선보다 위에 있으면 1이고 아래있으면 0을 의미합니다. 

![image](https://user-images.githubusercontent.com/52392004/187052865-db5a5eaf-bfa2-49cc-bcc8-54c9e702ac69.png)

Perceptron을 이용해 AND, OR, NOT을 표현하면 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/187052902-01df2b3c-5e68-41a6-928a-ca3755f28261.png)

또한, Artificial Neural Network를 이용해 XOR를 표현할수 있습니다.

![image](https://user-images.githubusercontent.com/52392004/187052913-3f7a55e3-c494-426f-b478-290ad9a1ba91.png)


### 활성함수 (Activation Function)

[Activation Function](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md)에서는 신경망에서 사용하고 있는 다양한 Activation function에 대해 설명하고 있습니다.

## Multi Layer Perceptron

Input (입력층), Hidden (은닉층), Output (출력층)의 3종류의 Layer들를 만들 수 있습니다. 

<img width="551" alt="image" src="https://user-images.githubusercontent.com/52392004/187053103-cf7ffa03-cf40-4d1f-b513-223c923e6dcc.png">


Layer 구조를 통해 비선형 모델링이 가능합니다. 



#### Universal Approximation Theorem

- Boolean Function: 2 Layers
- Continuous Function: 3 Layers
- Arbitrary Function: 3 Layers
