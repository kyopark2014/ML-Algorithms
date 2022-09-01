# 순환신경망 (Recurrent Neural Network)

여기서는 순환신경망에 대해 설명하고, 영화리뷰(Text)에 순환신경망을 적용하는 예제에 대해 설명합니다. 

## Recurrent Neural Network

순환신경망(RNN)은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보냅니다. 이것은, [은닉층에서 활성화 함수를 지난 값은 오직 출력층 방향으로만 향햐는 신경망(Feed Forward Neural Network)](https://github.com/kyopark2014/ML-Algorithms/blob/main/neural-network.md)과 다릅니다. 아래 그림이 CELL이 hidden state를 저장하는것을 보여주고 있습니다. 하나의 노드는 자신뿐 아니라 같은 Layer의 다른 노드와도 순환하므로 셀이라는 용어로 아래처럼 표현합니다. 

![image](https://user-images.githubusercontent.com/52392004/188026554-8ea74097-a8e7-45c4-a44a-979fa85c6681.png)

[Activation function](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md#activation-function)으로 아래와 같은 tanh을 사용합니다. activation function로 미분이 가능한 signoid를 쓰면, 층이 깊어질수록 미분값이 계속 작아지므로(gradient vanishing), ReLU를 사용하여 0보다 큰 경우에 값이 작아지는 문제를 해결할수 있으나, 값이 계속 커지면, [gradient exploding](https://wikidocs.net/61375)의 문제가 발생할 수 있습니다. 순환신경망은 Time step(입력의 길이)이 길어지면 Layer가 깊어진 것과 같은 효과가 발생하고, 순환시 동일한 가중치를 사용하므로 잘못된 가중치에 의해 학습이 잘 이루어지지 않을 가능성이 더 높습니다. 따라서 ReLU 대신 tanh 함수를 사용하여 gradient exploding 방지하고, gradient vanishing의 영향을 줄입니다. 

![image](https://user-images.githubusercontent.com/52392004/188026759-662c74eb-6add-426c-b0e4-4ca7d494bd74.png)



![image](https://user-images.githubusercontent.com/52392004/188027491-3123d09f-9bef-44bf-b9a0-099d68ef643e.png)






## Reference

[순환 신경망(Recurrent Neural Network, RNN)](https://wikidocs.net/22886)


