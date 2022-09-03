# 순환신경망 - LSTM

Simple RNN은 시퀀스가 길어질수록 학습이 어렵다. 

[Simple RNN 이용한 영화 리뷰](https://github.com/kyopark2014/ML-Algorithms/blob/main/rnn.md)에서 하나의 시퀀스안에는 여러개의 단어가 있고, 시퀀스의 길이는 time step의 길이와 같습니다. 병렬처리를 위해서는 시퀀스의 길이를 제한하게 되는데, 이때 시퀀스 길이보다 더 긴 시퀀스는 앞단을 자르고(기본), 작은 시퀀스는 0으로 채웁니다(Padding). 이와같이 긴 리뷰는 긴 시퀀스를 가지고, 마지막 time step의 정보는 앞단의 정보를 얕은 수준으로 갖게 되는데, text 전체에 대한 이해도가 낮아지므로 LSTM, GRU가 개발되었습니다. 

## LSTM 

Gradient Update Rule은 아래와 같습니다. 

```python
new weight = weight - learning rate * gradient
```




LSTM(Long Short-Term Memory)은 단기 기억을 오래 기억하기 위해 고안된 인공 신경망입니다. 입력 게이트, 삭제 게이트, 출력 게이트 역할을 하는 셀로 구성됩니다. 단일 데이터 포인트(이미지)뿐 아니라 전체 데이터 시퀀스(음성, 비디오) 처리 가능을 제공합니다. 필기 인식, 음성인식, 기계 번역, 로봇 제어등이 이용됩니다. 아래에는 [LSTM의 구조](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)를 보여줍니다. 


![image](https://user-images.githubusercontent.com/52392004/188254856-6a9a5b90-3b8c-4e16-9a95-1dee75821930.png)

여기서, x_t는 t번째의 입력이고, h_t는 hidden state의 sequence를 의미하는데 neural network의 메모리에에 저장된 이전 값을 access 합니다. c_i은 cell state를 의미합니다. 현재의 입력값(x_t), 이전 Hidden state(h_t), 이전 Cell state(c_t)를 가지고 현재의 Hidden state와 Cell state을 구한후 업데이트하게 됩니다. 여기서, activation function으로서 빨간색으로 표시된 것은 signoid이고, 파란색은 tanh 입니다. 

- forget gate (삭제 게이트): 입력과 이전 time step의 은닉상태(hidden state)의 합을 signoid 사용하여 일정값 이하는 삭제합니다.
- input gate (입력 게이트): 입력이 각각 sigmoid와 tanh activation function을 지나 multiplication 되며 cell state에 더해집니다. 
- output gate (출력 게이트): 입력과 이전 time step의 은닉상태(hidden state)의 합을 signoid 한것과 현재의 cell state의 값을 tanh한 값을 multiplication하여 현재의 hidden state를 구합니다. 
- cell state (셀상태): 이전 cell state와 input gateway와 output gateway의 값을 합하여, 현재의 cell state를 정의합니다. Long term memory의 역할을 합니다. 


## Reference

[Illustrated Guide to LSTM’s and GRU’s: A step by step explanation](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
