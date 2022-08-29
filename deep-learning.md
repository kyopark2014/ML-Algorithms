# Deep Learning

## Neural Network

[Neural Network](https://github.com/kyopark2014/ML-Algorithms/blob/main/neural-network.md)에서는 Neural Network의 특징 및 Error Back Propagation을 이용하여 최적의 Weight를 찾기 위한 방법에 대해 설명합니다. 

## Deep Learning 방법

- Keras: TensorFlow2.0에서 내장 라이브러리로 사용, 사용 용이
- TensorFlow: 산업계에서 선호, 구글이 만든 ML framework (2015), GPU 사용 용이하나 사용법이 복잡함 (함수 python 문법에서 제한이 있어 디버깅 불편)
- CNTK: 마이크로소프트
- theano: 학계
- PyTorch: 함수형 Python 사용이 가능하여 디버깅 용이, 연구용으로 많이 사용



## Keras Model

### Classification 예제

[상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/deep_learnig.ipynb)는 Fashion MNIST를 가지고 classification을 하는 deep learning 예제를 보여줍니다. 

```python
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

from tensorflow import keras

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
test_scaled = test_input / 255.0

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
    
def model_fn(a_layer=None):
    model = keras.Sequential(name='fashion')
    model.add(keras.layers.Flatten(input_shape=(28, 28), name='flatten'))   ## Batch Normalization
    model.add(keras.layers.Dense(100, activation='relu', name='hidden'))    ## Activation Function
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax', name='output'))
    return model

model = model_fn(keras.layers.Dropout(0.3))    ## Dropout

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')   # Optimizer, Loss Function

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only=True)   # Callback to save the best

history = model.fit(train_scaled, train_target, 
                    epochs=20, 
                    batch_size=32, 
                    verbose=1, 
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb])      # epoch, batch_size

model = keras.models.load_model('best-model.h5')

model.evaluate(val_scaled, val_target)

375/375 [==============================] - 0s 873us/step - loss: 0.3125 - accuracy: 0.8909
[0.3125297725200653, 0.890916645526886]
```

Test dataset에 대한 결과는 아래와 같습니다. 

```python
model.evaluate(test_scaled, test_target)
313/313 [==============================] - 0s 1ms/step - loss: 0.3394 - accuracy: 0.8841
[0.33944830298423767, 0.8841000199317932]
```


이때의 history graph는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/187084827-f0cf2722-13bf-46c7-ba2d-fd0778e57d8c.png)




### 손실함수 

이진분류의 경우에는 binary_crossentropy를, 다중분류의 경우에는 categorical_crossentropy을 사용합니다. 또한, 만약 티셔츠를 (1,0,0,0,0,0,0,0,0,0)와 같이
[one hot encoding](https://github.com/kyopark2014/ML-Algorithms/blob/main/neural-network-design.md#multi-class-classification)으로 표현된다면 아래와 같이 티셔츠는 -log(a1)으로 표현됩니다. 여기서는 Keras이 손실함수로 "sparse_categorical_crossentropy"을 씁니다. "sparse"를 붙이면 one hot encoding을 처리하여 줍니다. 

![image](https://user-images.githubusercontent.com/52392004/187072798-c115d22c-18d5-4c89-81a9-d51ee5849269.png)

아래처럼 Artifical Neural Network를 정의할 수 있습니다. 모델 훈련은 fit()으로 진행하는데, batch_size의 기본값은 32입니다. 만약 train_scaled가 4800개라면, epoch가 5개이고, batch_size가 32일때 한번의 epoch가 1500개씩 사용하게 됩니다. batch_size가 너무 크면 GPU에 올라갈수 없는 경우도 있으므로 중요한 옵션입니다. 

```python
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))   # Output
model = keras.Sequential(dense)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5, batch_size=32, verbose=1) 
```

### Activation Function

아래와 같이 [activation function](https://github.com/kyopark2014/ML-Algorithms/blob/main/perceptron.md#%ED%99%9C%EC%84%B1%ED%95%A8%EC%88%98-activation-function)을 dense에서 지정할때 지정할 수 있습니다. 예) signoid, softmax, relu

```python
model = keras.Sequential(name='fashion')
model.add(keras.layers.Flatten(input_shape=(28, 28), name='flatten'))
model.add(keras.layers.Dense(100, activation='relu', name='hidden'))
model.add(keras.layers.Dense(10, activation='softmax', name='output'))

model.summary()
```

### Optimizer (개선된 gradient descent method)

[Gradient Descent](https://github.com/kyopark2014/ML-Algorithms/blob/main/stochastic-gradient-descent.md#gradient-descent)는 모든 데이터를 가지고 에러값을 찾은 후에 기울기를 구해서 Weight를 업데이트 합니다. [Stochastic Gradient Descent](https://github.com/kyopark2014/ML-Algorithms/blob/main/stochastic-gradient-descent.md)는 확율을 이용해서 속도를 개선합니다. Adam은 Momentum과 Step size를 모두 고려하여 가장 많이 사용되고 있습니다.

![image](https://user-images.githubusercontent.com/52392004/187076472-21b31bbd-3bbb-4f89-8e0a-b457bf11cc49.png)

기본값인 RMSprop을 아래처럼 adom으로 변경 할 수 있습니다. 

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(train_scaled, train_target, epochs=5, batch_size=32, verbose=1) 
```

optimizer를 최적화할때는 accuracy가 최적인 상태로 수렴하는 시간과 최대 accuracy을 기준으로 합니다. 



### Dropout

Neural Network에서 불필요한 일부 Node를 제외함으로써 과적합(Overfitting) 문제를 해결할 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/187076792-5a5db38e-8f53-48f4-b4d4-e3886b6edf3f.png)

- Dropout을 통해 보통 성능이 좋아지지만, 데이터 적으면 오히려 나빠질수도 있습니다. 
- 학습속도는 2-3배 느려집니다.
- 테이터가 매우 클 경우에는 이미 과적합이 없으므로 성능개선이 미미합니다. 
- 40~60% 정도 dropout이 좋음

![image](https://user-images.githubusercontent.com/52392004/187076917-472b0c3b-83d9-4293-9862-4454eb54192a.png)

```python
model = model_fn(keras.layers.Dropout(0.3))

model.summary()
```

### Batch Normalization

Internal Covariate Shift (내부공변량 변화)는 [z-fold cross validation](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md#k-fold-cross-validation%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D)처럼 입력 데이터(mini batch)의 분포가 바뀌는 현상을 말합니다. 즉, 노드의 가중치가 바뀌면 입력의 분포가 바뀌게 되는 현상인데, 바뀌게 되는 분포가 레이어를 타고 전파되는 문제를 가지고 있어서, 학습시 분포가 계속 바뀌어서 정상적인 학습이 되지 않습니다. 

아래는 내부 공변량의 변화를 보여줍니다. 

![image](https://user-images.githubusercontent.com/52392004/187076987-35763aa5-494f-4e74-9e22-f958a317352b.png)

이 경우에 아래와 같이 입력을 정규분포로 변환하는 Layer를 추가(z-transform)하는 방법으로 해결 할 수 있습니다. 에러 그래프가 수렴하지 않고 계속 튀는 경우에 Batch normalization을 고려할 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/187077375-02d10cd7-e932-4931-963d-47055e302339.png)


## Flatten 

아래처럼 Flatten()을 이용하여 구현합니다. 

```python
model = keras.Sequential(name='fashion')
model.add(keras.layers.Flatten(input_shape=(28, 28), name='flatten'))
model.add(keras.layers.Dense(100, activation='relu', name='hidden'))
model.add(keras.layers.Dense(10, activation='softmax', name='output'))

model.summary()
```


### GPU 상태 확인 

아래처럼 console에서 nvidia-smi라는 명령어로 GPU 상태를 확인할 수 있습니다. 

```c
sh-4.2$ nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

## Deep Learning 실습

[Deep Learning 실습](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning-keras.md)에서는 Fashion MNIST 데이터를 이용하여 Keras 실습에 대해 설명합니다. 


## Reference 

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

