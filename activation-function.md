# 활성화함수 (Activation Function)

활성화함수(Activation Function)는 입력신호가 일정 기준 이상이면 출력신호로 변환하는 함수를 의미 합니다. 아래는 activation fuction의 한 예입니다. 여기서 x는 input, w은 weight, b는 bias를 의미합니다. 

<img width="535" alt="image" src="https://user-images.githubusercontent.com/52392004/193482564-69ff8f1a-66e9-48cc-a714-fd5b4bd43203.png">



Activation Function에는 Hard Limit, [Sigmoid](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md#sigmoid), [ReLU](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md#relu), [Leaky ReLU](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md#leaky-relu)가 있으며, 아래와 같이 여러종류의 Activation Function이 Deep Learning등에서 활용되고 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/187075789-d80d2ec9-f16e-4fbe-90f8-16ebaca88752.png)

[Neural Network - Activation Function](https://github.com/kyopark2014/deep-learning-algorithms/blob/main/deep-learning.md#activation-function-1)에서는 신경망에서 Activation Function에 대해 예제를 보여주고 있습니다. 


### Linear

CNN의 Conv2D에서 activation function의 기본값입니다. 입력 뉴런과 가중치로 계산된 결과값이 그대로 출력됩니다. 

### Sigmoid

시그모이드함수 (Sigmoid Function)는 선형방정식의 출력을 0에서 1사이의 확률로 압축합니다. [Logistic regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/logistic-regression.md)과 같은 분류 문제를 확률적으로 표현하는데 사용됩니다. 

![image](https://user-images.githubusercontent.com/52392004/185773923-7ca38926-f792-46c6-b339-f8459c2fea8c.png)

sigmoid를 쓰면 미분값이 0.25이 되므로 [Gradient vanishing problem](https://github.com/kyopark2014/ML-Algorithms/blob/main/stochastic-gradient-descent.md#gradient-vanishing--exploding)이 발생하는데 이를 해결하기 위해 [ReLU](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md#relu)등이 활용됩니다. 

### tanh

Hyperbolic Tangent(tanh)는 선형함수의 결과를 -1에서 1사이에서 비선형형태로 변형하여 줍니다. 시그모이드에서 결과값의 평균이 0이 아닌 양수로 편향된 문제를 해결하는데 사용했지만, 기울기 소멸(Gradient vanishing)문제는 여전히 발생합니다. 

<img width="541" alt="image" src="https://user-images.githubusercontent.com/52392004/193482437-47f06467-747b-4ac9-86a3-db4a565bdd24.png">


### ReLU

- Activation function으로 ReLU (Rectified Linear Unit)를 사용하여 Gradient vanishing 문제를 해결할 수 있습니다. 

- 입력(x)이 음수일때는 0을 출력하고, 양수일때는 x를 출력합니다. 

- 경사 하강법(Gradent descent)에 영향을 주지 않아서 학습 속도가 빠릅니다. Hyperbolic Tangent(tanh)에 비하여 속도가 6배 빠릅니다. 

- 음수값을 입력 받으면 항상 0을 출력하기 때문에 학습능력이 감소하므로 [Leaky ReLU](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md#leaky-relu) 함수를 사용하기도 합니다. 

ReLU의 그래프는 아래와 같습니다.

<img width="317" alt="image" src="https://user-images.githubusercontent.com/52392004/187075181-69d7c063-b725-4ace-a6f7-50d0341dff58.png">

수식은 아래와 같습니다. Net값이 음수일때는 0이고, 양수일때는 입력을 그래도 사용하므로, 중첩이 되어도 원래값을 유지합니다. 

![image](https://user-images.githubusercontent.com/52392004/187075198-08d51814-6e66-4ba0-a5ab-5f125fbfe951.png)

ReLU의 경우에 Sigmoid보다 약 6배 빠름지만, 학습데이터를 다 썼는데 한번도 사용되지 않은 노드가 있다면 Weight가 빠지면서 노드가 죽어버리는 문제점이 발생 할 수 있습니다. 이를 위해서, Learning rate를 작게 설정하거나 Leaky ReLU를 사용합니다. 

### Leaky ReLU

Leaky ReLU는 아래와 같은 그래프입니다.

<img width="370" alt="image" src="https://user-images.githubusercontent.com/52392004/187075354-be598ec4-4fc1-47a8-b236-4657127e82b8.png">

입력 값이 음수일때 0이 아닌 0.001처럼 매우 작은 수를 반환하면, 입력값이 수렴하는 구간이 제거되어 ReLU 함수를 사용할때 생기는 학습 능력 감소 문제를 해결할 수 있습니다. 

이것은 아래처럼 정의합니다. 

![image](https://user-images.githubusercontent.com/52392004/187075372-2a8d4197-a86d-4485-b6fb-f054f38becdf.png)

### ELU

<img width="459" alt="image" src="https://user-images.githubusercontent.com/52392004/193482494-bcac05ec-8200-4875-82eb-fc0d7bbbcc3d.png">






## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[딥러닝 텐서플로 교과서 - 서지영, 길벗](https://github.com/gilbutITbook/080263)
