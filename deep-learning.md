# Deep Learning

## Neural Network

[Neural Network](https://github.com/kyopark2014/ML-Algorithms/blob/main/neural-network.md)에서는 Neural Network의 특징 및 Error Back Propagation을 이용하여 최적의 Weight를 찾기 위한 방법에 대해 설명합니다. 

## Deep Learning 방법

- Keras: TensorFlow2.0에서 내장 라이브러리로 사용, 사용 용이
- TensorFlow: 산업계에서 선호, 구글이 만든 ML framework (2015), GPU 사용 용이하나 사용법이 복잡함 (함수 python 문법에서 제한이 있어 디버깅 불편)
- CNTK: 마이크로소프트
- theano: 학계
- PyTorch: 함수형 Python 사용이 가능하여 디버깅 용이, 상용화 어려움 



## Keras Model

이진분류의 경우에는 binary_crossentropy를, 다중분류의 경우에는 categorical_crossentropy을 사용합니다. 또한, 만약 티셔츠를 (1,0,0,0,0,0,0,0,0,0)와 같이
[one hot encoding](https://github.com/kyopark2014/ML-Algorithms/blob/main/neural-network-design.md#multi-class-classification)으로 표현된다면 아래와 같이 티셔츠는 -log(a1)으로 표현됩니다. 여기서는 Keras이 손실함수로 "sparse_categorical_crossentropy"을 씁니다. "sparse"를 붙이면 one hot encoding을 처리하여 줍니다. 

![image](https://user-images.githubusercontent.com/52392004/187072798-c115d22c-18d5-4c89-81a9-d51ee5849269.png)

아래처럼 Artifical Neural Network를 정의할 수 있습니다. 

```python
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))   # Output
model = keras.Sequential(dense)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5, verbose=1) 
```


## Neural Network 실습

1) Fashion MNIST 데이터를 준비합니다. 

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


2) 입력층/출력층만을 가지는 Artifical Neural Network으로 패션 아이템 분리하기 

아래와 같이 입력증과 출력층으로만 된 Neural Network로 모델을 훈련합니다.

![image](https://user-images.githubusercontent.com/52392004/187073354-3bc01ec0-ba49-470f-a44e-634317e0f06b.png)


```python
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))   # output layer
model = keras.Sequential(dense)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5, verbose=1) 
```

이때의 결과는 아래와 같습니다. 

```python
model.evaluate(val_scaled, val_target)

375/375 [==============================] - 0s 611us/step - loss: 0.4422 - accuracy: 0.8506
[0.44223520159721375, 0.8505833148956299]
```





