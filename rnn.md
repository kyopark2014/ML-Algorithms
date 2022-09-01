# 순환신경망 (Recurrent Neural Network)

여기서는 순환신경망에 대해 설명하고, 영화리뷰(Text)에 순환신경망을 적용하는 예제에 대해 설명합니다. 

## Recurrent Neural Network

순환신경망(RNN)은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보냅니다. 이것은, [은닉층에서 활성화 함수를 지난 값은 오직 출력층 방향으로만 향햐는 신경망(Feed Forward Neural Network)](https://github.com/kyopark2014/ML-Algorithms/blob/main/neural-network.md)과 다릅니다. 아래 그림이 CELL이 hidden state를 저장하는것을 보여주고 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/188026554-8ea74097-a8e7-45c4-a44a-979fa85c6681.png)

Activation function으로 아래와 같은 tanh을 사용합니다. 

![image](https://user-images.githubusercontent.com/52392004/188026759-662c74eb-6add-426c-b0e4-4ca7d494bd74.png)



![image](https://user-images.githubusercontent.com/52392004/188027491-3123d09f-9bef-44bf-b9a0-099d68ef643e.png)






## Reference

[순환 신경망(Recurrent Neural Network, RNN)](https://wikidocs.net/22886)


