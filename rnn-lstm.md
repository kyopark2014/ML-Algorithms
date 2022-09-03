# 순환신경망 - LSTM

Simple RNN은 시퀀스가 길어질수록 학습이 어렵다. 

[Simple RNN 이용한 영화 리뷰](https://github.com/kyopark2014/ML-Algorithms/blob/main/rnn.md)에서 하나의 시퀀스안에는 여러개의 단어가 있고, 시퀀스의 길이는 time step의 길이와 같습니다. 병렬처리를 위해서는 시퀀스의 길이를 제한하게 되는데, 이때 시퀀스 길이보다 더 긴 시퀀스는 앞단을 자르고(기본), 작은 시퀀스는 0으로 채웁니다(Padding). 이와같이 긴 리뷰는 긴 시퀀스를 가지고, 마지막 time step의 정보는 앞단의 정보를 얕은 수준으로 갖게 되는데, text 전체에 대한 이해도가 낮아지므로 LSTM, GRU가 개발되었습니다. 

## LSTM 

LSTM(Long Short-Term Memory)은 단기 기억을 오래 기억하기 위해 고안된 인공 신경망입니다. 입력 게이트, 삭제 게이트, 출력 게이트 역할을 하는 셀로 구성됩니다. 단일 데이터 포인트(이미지)뿐 아니라 전체 데이터 시퀀스(음성, 비디오) 처리 가능을 제공합니다. 필기 인식, 음성인식, 기계 번역, 로봇 제어등이 이용됩니다. 

![image](https://user-images.githubusercontent.com/52392004/188253685-5f11a035-2637-4b25-854b-ffb794a1874f.png)

여기서 activation function으로서 빨간색으로 표시된것은 signoid이고, 파란색은 tanh 입니다. 



## Reference

[Illustrated Guide to LSTM’s and GRU’s: A step by step explanation](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
