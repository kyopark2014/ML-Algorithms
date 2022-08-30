# Deep Learning 이용한 이미지 분류 (Image Classification)


[Keras Model](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md#keras-model)을 이용하여 이미지를 분류하는 방법에 대해 설명합니다.

## Convolutional Neural Networks

#### Convolution

Convolution은 Kernel을 이용하여 Feature map을 추출합니다. 이때 Convolution에 대한 수식은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/187317373-5033a609-d8e4-454d-9e4e-1464fd065fad.png)

Neural network에서 입력값에 Kernel로 convolution하려면 180 회전하여 사용하여야 합니다. 따라서, 학습의 목적과 복잡도를 생각할때, 180도 회전없는 [Cross-Correlation]https://glassboxmedicine.com/2019/07/26/convolution-vs-cross-correlation/)을 이용합니다. Cross-Correlation에 대한 수식은 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/187317929-08b95e8c-5cd6-4558-9066-42fe127e8347.png)


#### Padding

아래는 Keras에서 Convolution을 적용할때 사용하는 코드를 보여줍니다. 여기서 Padding에는 kernel을 이용한 convolution을 (0,0)에서 시작하여 입력값이 줄어드는 "valid"와 kernel에 매칭되지 않는 부분을 0으로 padding하여 원래 입력값을 유지하는 "same"의 옵션이 있습니다.

```python
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
```

상기와 같이 28x28인 이미지를 3x3크기의 Weight kernel 32개를 Convolution으로 적용하면, 32개의 Feature map이 생성됩니다. 

![image](https://user-images.githubusercontent.com/52392004/187319120-7bb8e3d3-e5bf-4b27-8e6c-641763f885ae.png)

이때의 Weight Parameter들의 수는 (input channel) x (kernel width) x (kernel height) + biases로서 여기서는 1 x 3 x 3 x 32 + 32로서 전체가 320개 입니다.


## Image Classification

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)을 아래와 같이 Convolution Neural Network을 구성하여 Image Classification을 구현합니다.

1) 입력되는 이미지의 크기는 28x28 입니다.
2) 3x3 Kernel 32개를 이용하여 Convolution을 진행하여 32개의 Feature map을 생성합니다.
3) Pooling Layter에서 이미지를 1/2로 down-scale을 하여, Feature map 크기를 14x14로 줄입니다.
4) 3x3 Kernel 64개를 이용하여 Convolution을 다시 진행하여, 64개의 Feature map을 생성합니다.
5) Polling layer에서 이미지를 1/2로 down-scale을 하여, Feature map 크기를 7x7로 줄입니다.
6) Neural Network에서 사용하기 편리하도록 Flatten을 이용하여 1차원 행열로 변경합니다.
7) 100개의 Node를 가지는 Hiden Layer를 적용합니다.
8) Output layer에서는 activation function으로 softmax를 적용합니다.

![image](https://user-images.githubusercontent.com/52392004/187320259-a941410d-22ea-47d2-b14c-aa0ddbc09ff2.png)

## Deep CNN Layer 소개 

아래와 같이 VGG16은 [ImageNet](https://image-net.org/)을 활용하여 우수한 Classfication을 수행하였습니다. 

![image](https://user-images.githubusercontent.com/52392004/187321098-8516ce26-870d-48fd-b57c-72a8ca528311.png)

아래는 VGG16의 구조를 보여줍니다. 여기서 conv3은 3x3필터를, conv1은 1x1필터를 의미합니다. 

![image](https://user-images.githubusercontent.com/52392004/187321146-7f78e489-1fcb-4d19-825d-374ff2674b0a.png)

Neural Network의 마지막 3 Fully-Connected Layer는 각각 4096, 4096, 1000 개의 Unit으로 구성돼 있으며, 1000 Unit을 가지는 출력층은 classification을 위한 Softmax 함수를 사용한다.



## Reference 

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)


[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)


[VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/)

[VGG16 논문 리뷰 — Very Deep Convolutional Networks for Large-Scale Image Recognition](https://medium.com/@msmapark2/vgg16-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-very-deep-convolutional-networks-for-large-scale-image-recognition-6f748235242a)
