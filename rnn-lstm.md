# 순환신경망 - LSTM

[Simple RNN 이용한 영화 리뷰](https://github.com/kyopark2014/ML-Algorithms/blob/main/rnn.md)에서 하나의 시퀀스안에는 여러개의 단어가 있고, 시퀀스의 길이는 time step의 길이와 같습니다. 병렬처리를 위해서는 시퀀스의 길이를 제한하게 되는데, 이때 시퀀스 길이보다 더 긴 시퀀스는 앞단을 자르고(default), 작은 시퀀스는 0으로 채웁니다(Padding). 이와같이 긴 리뷰는 긴 시퀀스를 가지고, 마지막 time step의 정보는 앞단의 정보를 얕은 수준으로 갖게 되는데, text 전체에 대한 이해도가 낮아지므로 LSTM, GRU가 개발되었습니다. 

## LSTM 

LSTM(Long Short-Term Memory)은 단기 기억을 오래 기억하기 위해 고안된 인공 신경망입니다. 입력 게이트, 삭제 게이트, 출력 게이트 역할을 하는 셀로 구성됩니다. 단일 데이터 포인트(이미지)뿐 아니라 전체 데이터 시퀀스(음성, 비디오) 처리 가능을 제공합니다. 필기 인식, 음성인식, 기계 번역, 로봇 제어등이 이용됩니다. 아래에는 [LSTM의 구조](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)를 보여줍니다. 

![image](https://user-images.githubusercontent.com/52392004/188254856-6a9a5b90-3b8c-4e16-9a95-1dee75821930.png)

여기서, x_t는 t번째의 입력이고, h_t는 hidden state의 sequence를 의미하는데 neural network의 메모리에에 저장된 이전 값을 access 합니다. c_i은 cell state를 의미합니다. 현재의 입력값(x_t), 이전 Hidden state(h_t), 이전 Cell state(c_t)를 가지고 현재의 Hidden state와 Cell state을 구한후 업데이트하게 됩니다. 여기서, activation function으로서 빨간색으로 표시된 것은 signoid이고, 파란색은 tanh 입니다. 

- forget gate (삭제 게이트): 입력과 이전 time step의 은닉상태(hidden state)의 합을 signoid 사용하여 일정값 이하는 삭제합니다.
- input gate (입력 게이트): 입력이 각각 sigmoid와 tanh activation function을 지나 multiplication 되며 cell state에 더해집니다. 
- output gate (출력 게이트): 입력과 이전 time step의 은닉상태(hidden state)의 합을 signoid 한것과 현재의 cell state의 값을 tanh한 값을 multiplication하여 현재의 hidden state를 구합니다. 
- cell state (셀상태): 이전 cell state와 input gate와 output gate의 값을 합하여, 현재의 cell state를 정의합니다. Long term memory의 역할을 합니다. 

## LTSM Sample

[LSTM 상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/rnn-ltsm.ipynb)를 아래에서 설명합니다.

1) 데이터를 준비합니다. 

```python
import tensorflow as tf

from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)

from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)
```

pad_sequences는 sample의 수가 maxlen(100)보다 작으면 0으로 padding을 추가합니다. 




2) LTSM 모델을 만듧니다. 

[Embedding](https://github.com/kyopark2014/ML-Algorithms/blob/main/rnn.md#embedding)적용후 8개의 neuron으로 LTSM을 정의합니다. 이진 분류이므로 output layer의 activation function으로 signoid를 사용합니다. keras.layers.Embedding (단어사전크기, 특징백터 사이즈,...,input_legth=입력 시퀀스 길이)로 표현합니다. 

```python
from tensorflow import keras

model = keras.Sequential(name='LTSM')

model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation='sigmoid', name='output'))

model.summary()
```



이때의 결과는 아래와 같습니다. 여기서 LSTM의 파라메터는 (16 X 8) + (8 X 8) + 8 = 800개가 됩니다. 특징백터크기(16)이고 neuron의 수는 8인데, 1개의 셀은 8개의 neuron을 순환하므로 8x8을 더해줍니다. 마지막 8은 bias(절편)입니다. 


```python
Model: "LTSM"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 100, 16)           8000      
_________________________________________________________________
lstm (LSTM)                  (None, 8)                 800       
_________________________________________________________________
output (Dense)               (None, 1)                 9         
=================================================================
Total params: 8,809
Trainable params: 8,809
Non-trainable params: 0
_________________________________________________________________
```

3) LTSM을 훈련시킵니다. 

[Optimizer](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md#optimizer-%EA%B0%9C%EC%84%A0%EB%90%9C-gradient-descent-method)는 RMSprop이 기본적으로 사용되는 여기서는 leaning_rate를 1e-4를 쓰고 있습니다. (기본은 1.e-3)



```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', 
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_seq, train_target, epochs=100, batch_size=64,
                    validation_data=(val_seq, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```                    

아래와 같이 epoch가 45에서 훈련이 종료됩니다. 

```python
Epoch 1/100
313/313 [==============================] - 8s 27ms/step - loss: 0.6926 - accuracy: 0.5401 - val_loss: 0.6918 - val_accuracy: 0.5872
Epoch 2/100
313/313 [==============================] - 8s 25ms/step - loss: 0.6901 - accuracy: 0.6211 - val_loss: 0.6879 - val_accuracy: 0.6440
Epoch 3/100
313/313 [==============================] - 8s 25ms/step - loss: 0.6791 - accuracy: 0.6594 - val_loss: 0.6629 - val_accuracy: 0.6740
...
Epoch 43/100
313/313 [==============================] - 8s 25ms/step - loss: 0.4032 - accuracy: 0.8192 - val_loss: 0.4351 - val_accuracy: 0.8012
Epoch 44/100
313/313 [==============================] - 8s 25ms/step - loss: 0.4025 - accuracy: 0.8198 - val_loss: 0.4363 - val_accuracy: 0.8002
Epoch 45/100
313/313 [==============================] - 8s 25ms/step - loss: 0.4013 - accuracy: 0.8206 - val_loss: 0.4352 - val_accuracy: 0.8014
```

이때의 결과는 아래와 같습니다. 

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

아래와 같은 Loss 그래프를 확인할 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/188255580-72a6c0a9-b4d5-4237-a902-b361a2ed653e.png)



과적합을 방지하기 위하여 drop out적용할 하고 2개의 LSTM을 아래처럼 적용할 수 있습니다. 여기서, "return_sequence"을 true로 설정하여, time step의 마지막 상태뿐 아니라, 이전 모든 time step의 결과를 LSTM으로 전달할 수 있습니다. 

```python
model3 = keras.Sequential(name='LTSM with dropout')

model3.add(keras.layers.Embedding(500, 16, input_length=100))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model3.add(keras.layers.LSTM(8, dropout=0.3))
model3.add(keras.layers.Dense(1, activation='sigmoid', name='output'))

model3.summary()
```



이때의 결과는 아래와 같습니다. 

```python
Model: "LTSM with dropout"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 100, 16)           8000      
_________________________________________________________________
lstm_1 (LSTM)                (None, 100, 8)            800       
_________________________________________________________________
lstm_2 (LSTM)                (None, 8)                 544       
_________________________________________________________________
output (Dense)               (None, 1)                 9         
=================================================================
Total params: 9,353
Trainable params: 9,353
Non-trainable params: 0
_________________________________________________________________
```

이때의 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/188255943-423d4235-1caa-43fb-8b78-2cc3aa5721b2.png)




## Reference

[Illustrated Guide to LSTM’s and GRU’s: A step by step explanation](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
