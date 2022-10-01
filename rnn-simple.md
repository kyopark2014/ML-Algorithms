# 순환신경망(RNN) 실습 

아래에서는 [혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)의 텍스트 분석관련 셈플을 중심으로 설명합니다. 

## Simple RNN 

[상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/rnn-simple.ipynb)에 대해 설명합니다.

1) [IMDB](https://www.imdb.com/) 데이터를 로드합니다.

읽어올때 토큰수(어휘수)는 num_words로 500만큼을 읽어오도록 설정합니다. 이때 데이터에서 0은 padding, 1은 문장시작을 의미하고, 어휘사전에 없는 수라면, 2로 변환합니다. 

```python
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)
```    



train_input은 [리뷰1, 리뷰2, 리뷰3, ...] 이런식으로 들어오는데, 아래와 같이 리뷰1의 크기는 218이고, 리뷰2의 크기는 189로 서로 다릅니다. 리뷰1을 보면 1은 문장의 시작을 어휘사전에 없는 단어는 2로 표시됩니다. train_target의 데이터를 보면, 1은 긍정적이고 0은 부정적을 의미합니다. 

```python
print(len(train_input[0]))
print(len(train_input[1]))
218
189

print(train_input[0])
[1, 14, 22, 16, 43, 2, 2, 2, 2, 65, 458, 2, 66, 2, 4, 173, 36, 256, 5, 25, 100, 43, 2, 112, 50, 2, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 2, 2, 17, 2, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2, 19, 14, 22, 4, 2, 2, 469, 4, 22, 71, 87, 12, 16, 43, 2, 38, 76, 15, 13, 2, 4, 22, 17, 2, 17, 12, 16, 2, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2, 2, 16, 480, 66, 2, 33, 4, 130, 12, 16, 38, 2, 5, 25, 124, 51, 36, 135, 48, 25, 2, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 2, 15, 256, 4, 2, 7, 2, 5, 2, 36, 71, 43, 2, 476, 26, 400, 317, 46, 7, 4, 2, 2, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2, 56, 26, 141, 6, 194, 2, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 2, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 2, 88, 12, 16, 283, 5, 16, 2, 113, 103, 32, 15, 16, 2, 19, 178, 32]

print(train_target[:20])
[1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]
```

2) train/validation dataset을 분리합니다.

```python
from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
```    

이때, 시퀀스의 길이는 병렬작업등을 고려하여 적절한 길이를 선정하여야 합니다. 이를 위해 평균(mean)과 중간값(median)값을 확인하면 아래와 같습니다. 여기서 중간값은 전체 분포의 중간을 나타냅니다. 

```python
import numpy as np

lengths = np.array([len(x) for x in train_input])

print(np.mean(lengths), np.median(lengths))

239.00925 178.0
```

또, train_input의 길이를 보면 아래 그래프와 같이 200단아 이내의 리뷰가 많음을 알 수 있습니다. 


```python
import matplotlib.pyplot as plt

plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
```

이때의 그래프입니다. 

![image](https://user-images.githubusercontent.com/52392004/188037315-0c49fd37-4727-4ee0-b0ee-f4d114c0a9fe.png)


3) padding 설정 

상기의 평균(mean), 중간값(median)값과 histogram을 참조하여, 입력데이터의 길이를 100으로 지정합니다. 즉, 길이가 100보다 짧으면 0을 패딩하고, 100보다 길면 앞부분을 잘라냅니다. padding과 truncationg을 이용하여 아래와 같이 pre/post를 설정할 수 있습니다. 

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)

# train_seq = pad_sequences(train_input, truncating='pre', padding = 'pre', maxlen=100)
```

아래와 같이 train_seq에서 1개의 sample을 보면, padding에 의해 앞의 0으로 채워졌음을 알 수 있습니다. 
```python
print(train_seq[5])

[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   1   2 128  74  12   2 163  15   4   2   2   2   2  32  85
 156  45  40 148 139 121   2   2  10  10   2 173   4   2   2  16   2   8
   4 226  65  12  43 127  24   2  10  10]
```

4) 순환신경망 만들기

SimpleRNN을 이용해 아래와 같이 8개의 neuron, 100개의 시퀀스길이, 500의 단어표현길이 (one hot encoding)로 순환신경망을 정의할 수 있습니다. 또한 이것은 이진분류이므로 output layer의 activation 함수는 "sigmoid"를 이용합니다. 

```python
from tensorflow import keras

model = keras.Sequential(name='imdb')

# input_shaepe = (시퀀스 길이 * 단어표현길이 (one hot encoding))
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))   # neuron:8 
model.add(keras.layers.Dense(1, activation='sigmoid', name='output'))

model.summary()
```

이때의 결과는 아래와 같습니다. 

```python
Model: "imdb"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn_11 (SimpleRNN)   (None, 8)                 4072      
                                                                 
 output (Dense)              (None, 1)                 9         
                                                                 
=================================================================
Total params: 4,081
Trainable params: 4,081
Non-trainable params: 0
```

순환층 파라메터의 수는 one hot encoding에 의한 단어표현길이(500) x neuron의 수(8) + 셀 내부의 순환 (8x8) + 편향(bias, 8) 이므로 전체가 4072개입니다. 추가적으로 output layer의 neuron의 수(8) + Bias(1)을 합치면 전체의 파라미터수는 4081개 입니다. 


Train과 Validation을 위한 dataset들을 one hot encoding을 합니다.


5) One hot encoding

```python
train_oh = keras.utils.to_categorical(train_seq)
val_oh = keras.utils.to_categorical(val_seq)
```

train_oh의 크기를 보면, one hot encoding에 의해 아래처럼 되었음을 알 수 있습니다. 

```python
print(train_oh.shape)

(25000, 100, 500)
```

6) 순환신경망 훈련하기 

- [Optimizer](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md#optimizer)는 RMSprop을 사용하는데, 이때 learning rate는 1e-4을 넣습니다. (기본값은 1e-3) learning rate를 낮은 값을 사용하면, 학습을 천천히하게 가장 적은 loss를 가지도록 훈련을 시킬 수 있습니다. 대신 속도가 느려집니다.  

- 이진분류이므로 [loss function](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md#%EC%86%90%EC%8B%A4%ED%95%A8%EC%88%98)은 binary_crossentropy를 사용합니다. 

- accuracy가 높아지는 방향으로 훈련이 되도록 matric으로 "accuracy"를 설정합니다. 

- callback으로 성능이 좋은것을 저장하고, 3번동안 개선이 안되면 학습을 종료하도록 합니다. 

- batch_size는 기본이 32인데 아래처럼 64로 설정합니다. 앞에서 learning rate을 줄이면서 늦어지는 속도를 높이기 위해 조정하였습니다. 보통은 batch size를 줄이는것이 작게 유지하는것이 성능이 향상된다고 합니다. 

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', 
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

이것의 결과는 아래와 같습니다. 아래와 같이 epoch가 100까지 가기전에 종료됩니다. 

```python
Epoch 1/100
313/313 [==============================] - 9s 22ms/step - loss: 0.6989 - accuracy: 0.5041 - val_loss: 0.6978 - val_accuracy: 0.5038
Epoch 2/100
313/313 [==============================] - 6s 19ms/step - loss: 0.6946 - accuracy: 0.5102 - val_loss: 0.6949 - val_accuracy: 0.5102
Epoch 3/100
313/313 [==============================] - 6s 19ms/step - loss: 0.6920 - accuracy: 0.5200 - val_loss: 0.6930 - val_accuracy: 0.5146
......
Epoch 59/100
313/313 [==============================] - 6s 18ms/step - loss: 0.4126 - accuracy: 0.8212 - val_loss: 0.4654 - val_accuracy: 0.7830
Epoch 60/100
313/313 [==============================] - 6s 18ms/step - loss: 0.4117 - accuracy: 0.8229 - val_loss: 0.4624 - val_accuracy: 0.7844
```

이것의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/188150748-89721f2a-e7d4-47d9-94ef-baf52aa1b88f.png)

One hot encoding으로 인해 메모리가 너무 커지는 문제가 있을 수 있습니다. 

```python
print(train_seq.nbytes, train_oh.nbytes)

10000000 5000000000
 ```

따라서, 아래와 같이 one hot encoding 대신에 embedding을 이용하기도 합니다. 



## Embedding을 이용하기 

Embedding을 이용하여 500개의 단어, 16개의 실수로 이루어진 벡터로 embdding을 하는데 sequence의 길이(input_length)는 100입니다. 

아래는 [상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/rnn-embeding.ipynb)에 대해 설명하고 있습니다. 

```python
from tensorflow import keras

model2 = keras.Sequential(name='imdb')

model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))   # neuron:8
model2.add(keras.layers.Dense(1, activation='sigmoid', name='output'))

model2.summary()
```

model summary는 아래와 같습니다. 

```python
Model: "imdb"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_1 (Embedding)     (None, 100, 16)           8000      
                                                                 
 simple_rnn_1 (SimpleRNN)    (None, 8)                 200       
                                                                 
 output (Dense)              (None, 1)                 9         
                                                                 
=================================================================
Total params: 8,209
Trainable params: 8,209
Non-trainable params: 0
_________________________________________________________________
```

이때에 Embedding층 파라메터 갯수 = 500 x 16 = 8000개이고, 순환층 파라메터는 갯수 = 16*8 + 8*8 + 8 = 200개입니다. 

아래와 같이 학습을 진행합니다. 

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', 
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
```                     

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/188250998-fb21fc8a-df94-48f4-b359-85f528cb5670.png)




## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
