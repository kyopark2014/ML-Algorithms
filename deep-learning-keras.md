# Deep Learning (Keras) 실습

아래에서는 Fashion MNIST를 이용하여 사진에 대한 Classification을 Keras로 구현합니다. 

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


## 
 


## Reference 

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
