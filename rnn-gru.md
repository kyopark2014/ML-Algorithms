# 순환신경망 - GRU

## GRU 

GRU(Gated Recurrent Unit)는 [LSTM](https://github.com/kyopark2014/ML-Algorithms/blob/main/rnn-lstm.md)과 흡사하지만 경량화된 모델입니다. 아래에서는 GRU의 셀의 구조를 보여주고 있습니다. 여기서, Activation function으로 빨간색은 sigmoid이고 파란색은 tanh 입니다. 

- Update Gate는 LSTM의 Forget Gate와 Input Gate 처럼 동작합니다. 
- Reset Gate는 얼마나 많은 과거 정보를 삭제할지를 결정할 수 있습니다. 


![image](https://user-images.githubusercontent.com/52392004/188256227-e21516f9-456b-424d-b23c-02c4794e12b1.png)

이것은 [LSTM](https://github.com/kyopark2014/ML-Algorithms/blob/main/rnn-lstm.md)의 Cell state가 없으며, update와 reset gate를 가지고 있습니다. 




## GRU Sample

[GRU 상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/rnn-gru.ipynb)를 아래에서 설명합니다. 

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

2) GRU model을 만듧니다. 

Embedding은 keras.layers.Embedding (단어사전크기, 특징백터 사이즈,...,input_legth=입력 시퀀스 길이)로 표현합니다. 

```python
from tensorflow import keras

model = keras.Sequential(name='GRU')

model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.GRU(8))
model.add(keras.layers.Dense(1, activation='sigmoid', name='output'))

model.summary()
```


이때 생성된 모델은 아래와 같습니다. 

```python
Model: "GRU"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_1 (Embedding)     (None, 100, 16)           8000      
                                                                 
 gru_1 (GRU)                 (None, 8)                 624       
                                                                 
 output (Dense)              (None, 1)                 9         
                                                                 
=================================================================
Total params: 8,633
Trainable params: 8,633
Non-trainable params: 0
_________________________________________________________________
```

이때 GRU 파라메터의 숫자는 (16 x 8) + (8 x 8) + 8 + 8 = 624입니다. 여기서 단어표현길이는 16이고, neuron의 숫가 8개입니다. 1개의 셀은 8개의 neuron과 순환하므로 8 x 8을 더해야 하며, bias(절편)로 neuron의 숫자인 8을 더합니다. 그런데 LSTM과 비교하여 bias를 2번을 더해야 합니다. 


3) 모델을 훈련합니다. 

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', 
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-gru-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
```                     

아래와 같이 epoch가 72일때 훈련이 종료됩니다. 

```python
Epoch 1/100
313/313 [==============================] - 11s 34ms/step - loss: 0.6927 - accuracy: 0.5191 - val_loss: 0.6922 - val_accuracy: 0.5362
Epoch 2/100
313/313 [==============================] - 10s 32ms/step - loss: 0.6910 - accuracy: 0.5620 - val_loss: 0.6904 - val_accuracy: 0.5528
Epoch 3/100
313/313 [==============================] - 10s 32ms/step - loss: 0.6883 - accuracy: 0.5794 - val_loss: 0.6873 - val_accuracy: 0.5780
...
Epoch 70/100
313/313 [==============================] - 10s 31ms/step - loss: 0.4088 - accuracy: 0.8170 - val_loss: 0.4348 - val_accuracy: 0.7986
Epoch 71/100
313/313 [==============================] - 10s 31ms/step - loss: 0.4084 - accuracy: 0.8169 - val_loss: 0.4348 - val_accuracy: 0.7994
Epoch 72/100
313/313 [==============================] - 10s 31ms/step - loss: 0.4086 - accuracy: 0.8163 - val_loss: 0.4339 - val_accuracy: 0.8006
```


아래와 같이 결과를 확인합니다. 

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/188257045-b3091143-31b2-4c18-927c-467ccc5ff681.png)

최적일때의 결과를 로드해서 확인하면 아래와 같습니다.

```python
test_seq = pad_sequences(test_input, maxlen=100)

rnn_model = keras.models.load_model('best-gru-model.h5')

rnn_model.evaluate(test_seq, test_target)
```

이때의 결과는 아래와 같습니다. 

```python
782/782 [==============================] - 5s 6ms/step - loss: 0.4384 - accuracy: 0.7966

[0.43842393159866333, 0.7965999841690063]
```
[0.43842393159866333, 0.7965999841690063]
