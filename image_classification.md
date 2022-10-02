# Deep Learning 이용한 이미지 분류 (Image Classification)


[Keras Model](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md#keras-model)을 이용하여 이미지를 분류하는 방법에 대해 설명합니다.

## Convolutional Neural Network

합성곱 신경망(Convolutional Neural Network)은 이미지 또는 비디오상의 객체를 식별하는 이미지 분류에 이용할 수 있습니다. 

### Convolution

Convolution은 Kernel을 이용하여 Feature map을 추출합니다. 이때 Convolution에 대한 수식은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/187317373-5033a609-d8e4-454d-9e4e-1464fd065fad.png)

Neural network에서 입력값에 Kernel로 convolution하려면 180 회전하여 사용하여야 합니다. 따라서, 연산의 효율성을 고려하여, 180도 회전없는 [Cross-Correlation](https://glassboxmedicine.com/2019/07/26/convolution-vs-cross-correlation/)을 이용합니다. Cross-Correlation에 대한 수식은 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/187317929-08b95e8c-5cd6-4558-9066-42fe127e8347.png)


### Padding

경계 처리 방법으로 valid와 same을 선택할 수 있습니다. 

- valid: 유효한 영역만 출력되므로 출력 이미지 크기는 입력 이미지 크기보다 작습니다. kernel을 이용한 convolution을 (0,0)에서 시작하여 입력값이 줄어듭니다.
- same: 출력 이미지 크기가 입력 이미지 크기와 동일합니다. kernel에 매칭되지 않는 부분을 0으로 padding하여 원래 입력값을 유지(full cross correlation)합니다.

아래는 Keras에서 Convolution을 적용할때 사용하는 코드를 보여줍니다. 

```python
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
```

상기와 같이 28x28인 이미지를 3x3크기의 Weight kernel 32개를 Convolution으로 적용하면, 32개의 Feature map이 생성됩니다. 이때, Feature map의 크기는 padding이 "same"(full cross correlation)이므로 원본과 같은 32x32 입니다. 

![image](https://user-images.githubusercontent.com/52392004/187319120-7bb8e3d3-e5bf-4b27-8e6c-641763f885ae.png)

이때의 Weight Parameter들의 수는 (input channel) x (kernel width) x (kernel height) + biases로서 여기서는 1 x 3 x 3 x 32 + 32로서 전체가 320개 입니다.


## Image Classification

### 구현방법 

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)을 아래와 같이 Convolution Neural Network을 구성하여 Image Classification을 구현합니다.

![image](https://user-images.githubusercontent.com/52392004/187320259-a941410d-22ea-47d2-b14c-aa0ddbc09ff2.png)

이것을 구현하는 순서는 아래와 같습니다. 

1) 입력되는 이미지의 크기는 28x28 입니다.
2) 3x3 Kernel 32개를 이용하여 Convolution을 진행하여 32개의 Feature map을 생성합니다.
3) Pooling Layter에서 이미지를 1/2로 down-scale을 하여, Feature map 크기를 14x14로 줄입니다.
4) 3x3 Kernel 64개를 이용하여 Convolution을 다시 진행하여, 64개의 Feature map을 생성합니다.
5) Polling layer에서 이미지를 1/2로 down-scale을 하여, Feature map 크기를 7x7로 줄입니다.
6) Neural Network에서 사용하기 편리하도록 Flatten을 이용하여 1차원 행열로 변경합니다.
7) 100개의 Node를 가지는 Hiden Layer를 적용합니다.
8) Output layer에서는 activation function으로 softmax를 적용합니다.

### 구현된 코드 

이를 [Code로 구현](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/image_classification.ipynb)하면 아래와 같습니다. 

1) Fashion MNIST를 읽어와서 전처리를 하고 Train/Test dataset을 아래처럼 생성합니다. 
```python
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```    

2) CNN을 구성합니다. 

```python
model = keras.Sequential(name='fashion')
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten(input_shape=(28, 28), name='flatten'))   ## Flatten
model.add(keras.layers.Dense(100, activation='relu', name='hidden'))    ## Activation Function
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax', name='output'))    
```

생성된 모델은 아래와 같습니다. 

```python
model.summary()

Model: "fashion"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_2 (Conv2D)           (None, 28, 28, 32)        320       
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 14, 14, 32)       0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 14, 14, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 3136)              0         
                                                                 
 hidden (Dense)              (None, 100)               313700    
                                                                 
 dropout_1 (Dropout)         (None, 100)               0         
                                                                 
 output (Dense)              (None, 10)                1010      
                                                                 
=================================================================
Total params: 333,526
Trainable params: 333,526
Non-trainable params: 0
_________________________________________________________________
```

3) 모델은 훈련시킵니다. 

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20,
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

4) 결과를 확인합니다. 

아래와 같이 손실함수를 그려보면, 6번째 epoch일때 최상의 결과를 얻고 있습니다. 

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

이때의 그림입니다. 

![image](https://user-images.githubusercontent.com/52392004/187322325-aca8ce28-f5c2-4137-981b-520e11ade26e.png)

validation dataset에 대한 결과는 아래와 같습니다. 

```python
model.evaluate(val_scaled, val_target)
375/375 [==============================] - 1s 2ms/step - loss: 0.2141 - accuracy: 0.9204
[0.21413874626159668, 0.9204166531562805]
```

test dataset에 대한 결과는 아래와 같습니다. 

```python
model.evaluate(test_scaled, test_target)
313/313 [==============================] - 1s 2ms/step - loss: 0.2360 - accuracy: 0.9161
[0.23603354394435883, 0.916100025177002]
```

validation dataset의 0번째가 '가방'일 경우에, 아래처럼 class들에 대한 리스트를 만들어서 predict()를 하면 classification 결과를 활용 수 있습니다. 

```python
classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
           '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']
import numpy as np
preds = model.predict(val_scaled[0:1])
print(classes[np.argmax(preds)])

가방
```

## Deep CNN Layer 소개 

아래와 같이 VGG16은 [ImageNet](https://image-net.org/)을 활용하여 우수한 Classfication을 수행하였습니다. 

![image](https://user-images.githubusercontent.com/52392004/187321098-8516ce26-870d-48fd-b57c-72a8ca528311.png)

아래는 VGG16의 구조를 보여줍니다. 여기서 conv3은 3x3필터를, conv1은 1x1필터를 의미합니다. 

![image](https://user-images.githubusercontent.com/52392004/187321146-7f78e489-1fcb-4d19-825d-374ff2674b0a.png)

Neural Network의 마지막 3 Fully-Connected Layer는 각각 4096, 4096, 1000 개의 Unit으로 구성돼 있으며, 1000 Unit을 가지는 출력층은 classification을 위한 Softmax 함수를 사용한다.

유사한 데이터에 대한 CNN 모델을 Transfer learning을 이용하면 적은 노력으로 좋은 결과를 얻을 수 있습니다. 


## Reference 

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)


[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)


[VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/)

[VGG16 논문 리뷰 — Very Deep Convolutional Networks for Large-Scale Image Recognition](https://medium.com/@msmapark2/vgg16-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-very-deep-convolutional-networks-for-large-scale-image-recognition-6f748235242a)

[딥러닝 텐서플로 교과서 - 서지영, 길벗](https://github.com/gilbutITbook/080263)
