# 순환신경망 (Recurrent Neural Network)

여기서는 순환신경망에 대해 설명하고, 영화리뷰(Text)에 순환신경망을 적용하는 예제에 대해 설명합니다. 

## Recurrent Neural Network

순환신경망(RNN)은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보냅니다. 이것은, [은닉층에서 활성화 함수를 지난 값은 오직 출력층 방향으로만 향햐는 신경망(Feed Forward Neural Network)](https://github.com/kyopark2014/ML-Algorithms/blob/main/neural-network.md)과 다릅니다. 아래 그림이 CELL이 hidden state를 저장하는것을 보여주고 있습니다. 하나의 노드는 자신뿐 아니라 같은 Layer의 다른 노드와도 순환하므로 셀이라는 용어로 아래처럼 표현합니다. 

![image](https://user-images.githubusercontent.com/52392004/188026554-8ea74097-a8e7-45c4-a44a-979fa85c6681.png)

[Activation function](https://github.com/kyopark2014/ML-Algorithms/blob/main/activation-function.md)으로 아래와 같은 tanh을 사용합니다. activation function로 미분이 가능한 signoid를 쓰면, 층이 깊어질수록 미분값이 계속 작아지므로(gradient vanishing), ReLU를 사용하여 0보다 큰 경우에 값이 작아지는 문제를 해결할수 있으나, 값이 계속 커지면, [gradient exploding](https://wikidocs.net/61375)의 문제가 발생할 수 있습니다. 순환신경망은 Time step(입력의 길이)이 길어지면 Layer가 깊어진 것과 같은 효과가 발생하고, 순환시 동일한 가중치를 사용하므로 잘못된 가중치에 의해 학습이 잘 이루어지지 않을 가능성이 더 높습니다. 따라서 ReLU 대신 tanh 함수를 사용하여 gradient exploding 방지하고, gradient vanishing의 영향을 줄입니다. 

![image](https://user-images.githubusercontent.com/52392004/188026759-662c74eb-6add-426c-b0e4-4ca7d494bd74.png)

아래의 샘플의 text는 아래와 같이 셈플의 수, 시퀀스 길이, 단어표현을 위한 길이로 나타낼 수 있습니다. 이것을 순환층에 넣게 되면 결과를 아래처럼 마지막 타임스텝에 대한 뉴런갯수 만큼의 출력이 나옵니다.

![image](https://user-images.githubusercontent.com/52392004/188027491-3123d09f-9bef-44bf-b9a0-099d68ef643e.png)

- 샘플마다 시퀀스 길이가 다를 수 있으므로 적당한 시퀀스 길이를 선택하여야 합니다. 시퀀스의 평균길이와 같은것이 쓰일 수 있습니다. 

- 시퀀스 길이 보다 긴 시퀀스는 시퀀스를 앞 또는 뒤에서 잘라서 사용합니다. 보통은 텍스트가 긴 경우에 앞보다는 뒤에 더 중요한 정보가 있다고 하여 앞을 자릅니다. (keras의 기본값도 앞을 자름)

- 시퀀스 길이 보다 짧은 시퀀스는 0으로 빈칸을 채우게 됩니다.(패딩)

- 병렬처리를 위해서는 가능한 시퀀스 길이등 입력데이터를 맞추어 주어야 합니다. 


## NLP (Natural Language Processing)

자연어 처리는 음성인식, 기계번역, 감성 분석에 필요합니다.

- 말뭉치는 자연어 훈련 데이터를 의미합니다.

- 토큰은 의미 단위로서, 영어에서는 단어를 쓰고, 한국어는 형태소를 사용합니다. 

- 어휘 사전은 사용할 고유한 토큰의 집합을 의미하는데, 많이 등장한 단어 위주로 선정합니다. 


## Embedding

단어의 표현은 Fashion MNIST처럼 [One-Hot Encoding](https://github.com/kyopark2014/ML-Algorithms/blob/main/embedding.md#one-hot-encoding)을 사용할 수 있습니다. 하지만 이 경우에 어휘 사전의 Token수 만큼 배열의 크기를 정해야해서 데이터가 커지며. 의미 유사성, Collocation(같이 쓰임) 등을 포함 할 수 없습니다. 이를 개선하기 위해 [Embedding](https://github.com/kyopark2014/ML-Algorithms/blob/main/embedding.md)이 사용할 수 있습니다. 



## 실습예제

[Simple Neural Network로 영화리뷰(IMDB)를 이진분류](https://github.com/kyopark2014/ML-Algorithms/blob/main/rnn-simple.md)로 분석합니다. 



## Reference

[순환 신경망(Recurrent Neural Network, RNN)](https://wikidocs.net/22886)


[영화 리뷰 데이터](https://www.imdb.com/)
