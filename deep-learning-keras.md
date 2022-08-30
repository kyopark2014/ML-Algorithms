# Deep Learning (Keras) 실습

아래에서는 Fashion MNIST를 이용하여 사진에 대한 Classification을 Keras로 구현합니다. [상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/deep-learning-practice.ipynb)를 아래에서 상세하게 설명하고 있습니다. 

## Fashion MNIST 데이터를 준비

패션 MNIST 셈플을 사용합니다. 이것은 10가지 class에 대하여 28x28 픽셀의 이미지 70,000개를 제공합니다. 

<img width="337" alt="image" src="https://user-images.githubusercontent.com/52392004/187072325-912a6ee3-57f3-4184-a473-8a34013283a4.png">

이중 60000개는 train.input으로 10000개는 test_input으로 아래처럼 저장합니다. 

```python
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

from tensorflow import keras

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)
```

Sample 다운로드 위치는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/187067937-cb6108a7-b8e2-491a-a864-fce143d8a854.png)

10개의 Class는 아래와 같습니다. 

```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()
```

10개의 Class는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/187068125-5f4464da-f2aa-4512-b8d2-2033e2c56dd0.png)


## 입력층/출력층 만을 가지는 Artifical Neural Network으로 패션 아이템 분리하기 

아래와 같이 입력증과 출력층으로만 된 Neural Network로 모델을 훈련합니다.

![image](https://user-images.githubusercontent.com/52392004/187073354-3bc01ec0-ba49-470f-a44e-634317e0f06b.png)

다중선형분류이므로 출력층의 activation function으로는 softmax를 쓰고, Loss 함수로는 categorical crossentropy를 사용합니다. 

```python
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))   # output layer
model = keras.Sequential(dense)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5, batch_size=32, verbose=1) 
```

모델 훈련은 아래와 같이 5번의 epoch로 진행됩니다. 여기서, loss는 crossentropy값이고 accuracy는 정확도를 의미합니다. 

```python
Epoch 1/5
1500/1500 [==============================] - 1s 772us/step - loss: 0.6071 - accuracy: 0.7967
Epoch 2/5
1500/1500 [==============================] - 1s 785us/step - loss: 0.4785 - accuracy: 0.8394
Epoch 3/5
1500/1500 [==============================] - 1s 765us/step - loss: 0.4561 - accuracy: 0.8484
Epoch 4/5
1500/1500 [==============================] - 1s 779us/step - loss: 0.4447 - accuracy: 0.8536
Epoch 5/5
1500/1500 [==============================] - 1s 811us/step - loss: 0.4372 - accuracy: 0.8548
```

이때의 결과는 아래와 같습니다. 

```python
model.evaluate(val_scaled, val_target)

375/375 [==============================] - 0s 611us/step - loss: 0.4422 - accuracy: 0.8506
[0.44223520159721375, 0.8505833148956299]
```


## 다중 Network를 이용

입력층에 784개, 은닉층(Hidden layer)에 100개, 출력층으로 10개 Node를 사용합니다. 

<img width="629" alt="image" src="https://user-images.githubusercontent.com/52392004/187562060-041834a9-1c47-447d-8599-052ebde5e3d0.png">



### Layer 추가 방법

#### 1) Dense를 추가하는 방법 

```python
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden')
dense2 = keras.layers.Dense(10, activation='softmax', name='output')

model = keras.Sequential([dense1, dense2], name='fashion')

model.summary()
```

#### 2) keras.Sequential내부에서 정의하는 방법 

```python
model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')
], name='fashion')

model.summary()
```

#### 3) model에 add 하는 방법

```python
model = keras.Sequential(name='fashion')
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'))
model.add(keras.layers.Dense(10, activation='softmax', name='output'))

model.summary()
```

## Activation function을 바꾸는 방법

아래처럼 layer를 dense로 추가할때, "activation"을 추가합니다. 

```python
model.add(keras.layers.Dense(100, activation='relu', name='hidden'))
```

## Optimizer

아래는 Optimzer를 [Stochastic Gradient Descent](https://github.com/kyopark2014/ML-Algorithms/blob/main/classification.md#stochastic-gradient-descent)로 지정하는 예제입니다. 

```python
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')
```

또는 아래처럼 쓸수도 있습니다. 

```python
sgd = keras.optimizers.SGD()  # Stocastic Gradient Descent
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')
```

기본값은 RMSprop 입니다. 

## 검증손실

model 훈련시 결과를 history로 받아서 아래처럼 결과를 볼 수 있습니다.

```python
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

history = model.fit(train_scaled, train_target, epochs=20, verbose=0, 
                    validation_data=(val_scaled, val_target))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```                    

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/187082737-e0bfa3fc-6acb-4da3-ba37-e320c391bfe0.png)


## Dropout

아래처럼 Droupt을 지정할 수 있습니다. 

```python
model = model_fn(keras.layers.Dropout(0.3))

model.summary()
```

![image](https://user-images.githubusercontent.com/52392004/187082823-33f7da15-1c2a-4c69-934d-c9f71faa2f30.png)


## 모델 저장 

아래처럼 model의 weight만 저장하거나 전체를 저장할 수 있습니다. 

```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

history = model.fit(train_scaled, train_target, epochs=10, verbose=0, 
                    validation_data=(val_scaled, val_target))

model.save_weights('model-weights.h5')
model.save('model-whole.h5')
```

## 모델 복원 

아래처럼 weight를 복원 할 수 있습니다. 

```python
model = model_fn(keras.layers.Dropout(0.3))

model.load_weights('model-weights.h5')

import numpy as np

val_labels = np.argmax(model.predict(val_scaled), axis=-1)
print(np.mean(val_labels == val_target))

0.87875
```

아래처럼 전체를 로드후에 evaluate로 분석할 수 있습니다. 

```python
model = keras.models.load_model('model-whole.h5')

model.evaluate(val_scaled, val_target)
```

이때의 결과는 아래와 같습니다. 

```python
375/375 [==============================] - 0s 828us/step - loss: 0.3315 - accuracy: 0.8788
[0.33149588108062744, 0.8787500262260437]
```

## 콜백

save_best_only로 저장후 다시 로드해서 사용합니다. 

```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', 
                                                save_best_only=True)

model.fit(train_scaled, train_target, epochs=20, verbose=0, 
          validation_data=(val_scaled, val_target),
          callbacks=[checkpoint_cb])
```

가장 좋은 성능을 가진 모델을 로딩하고 evaluate 합니다. 

```python
model = keras.models.load_model('best-model.h5')

model.evaluate(val_scaled, val_target)
```

이후 결과는 아래처럼 확인합니다.

```python
375/375 [==============================] - 1s 2ms/step - loss: 0.3172 - accuracy: 0.8863
[0.31721678376197815, 0.8862500190734863]
```

## Reference 

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
