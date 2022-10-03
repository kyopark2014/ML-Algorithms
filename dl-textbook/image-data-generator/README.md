# 이미지 데이터 확장

[Chapter 5.3.1 특성추출기법 - 이미지 데이터 확장](https://github.com/gilbutITbook/080263/blob/master/chap5/python_5%EC%9E%A5.ipynb)을 참조하여, [image-data-generator.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/dl-textbook/image-data-generator/image-data-generator.ipynb)를 아래와 같이 설명합니다. 

## Data Loading

데이터를 아래처럼 읽어옵니다. 

```python
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from matplotlib import pyplot as plt

img=load_img('./data/bird.jpg')
data=img_to_array(img) 
```

## width_shift_range 이용한 이미지 증가

```python
img_data=expand_dims(data, 0) 
data_gen=ImageDataGenerator(width_shift_range=[-200,200]) 
data_iter=data_gen.flow(img_data, batch_size=1) 
fig=plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch=data_iter.next()
    image=batch[0].astype('uint16')
    plt.imshow(image)
plt.show()
```

![image](https://user-images.githubusercontent.com/52392004/193434434-06433ac2-a5de-475f-b1d3-494f6615f924.png)

## height_shift_range 이용한 이미지 증가

```python
img_data=expand_dims(data, 0) 
data_gen=ImageDataGenerator(height_shift_range=0.5) 
data_iter=data_gen.flow(img_data, batch_size=1) 
fig=plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch=data_iter.next()
    image=batch[0].astype('uint16')
    plt.imshow(image)
plt.show()
```

![image](https://user-images.githubusercontent.com/52392004/193434452-0281053e-45f6-497c-af16-ae24431e4c4a.png)


## flip 이용한 이미지 증가

```python
img_data=expand_dims(data, 0) 
data_gen=ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
data_iter=data_gen.flow(img_data, batch_size=1) 
fig=plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch=data_iter.next()
    image=batch[0].astype('uint16')
    plt.imshow(image)
plt.show()
```

![image](https://user-images.githubusercontent.com/52392004/193434469-8ca17db6-f4b7-4176-a834-16e64a19bbb5.png)



## rotation_range 이용한 이미지 증가

```python
img_data=expand_dims(data, 0) 
data_gen=ImageDataGenerator(rotation_range=90) 
data_iter=data_gen.flow(img_data, batch_size=1) 
fig=plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch=data_iter.next()
    image=batch[0].astype('uint16')
    plt.imshow(image)
plt.show()
```

![image](https://user-images.githubusercontent.com/52392004/193434484-355841c1-1ee8-40dc-bf21-07a9eafe78fe.png)

## brightness 이용한 이미지 증가

```python
img_data=expand_dims(data, 0) 
data_gen=ImageDataGenerator(brightness_range=[0.3,1.2]) 
data_iter=data_gen.flow(img_data, batch_size=1) 
fig=plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch=data_iter.next()
    image=batch[0].astype('uint16')
    plt.imshow(image)
plt.show()
```

![image](https://user-images.githubusercontent.com/52392004/193434493-e9d9cb1d-9b46-4287-b971-403589656104.png)


## zoom 이용한 이미지 증가

```python
img_data=expand_dims(data, 0) 
data_gen=ImageDataGenerator(zoom_range=[0.4, 1.5]) 
data_iter=data_gen.flow(img_data, batch_size=1) 
fig=plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch=data_iter.next()
    image=batch[0].astype('uint16')
    plt.imshow(image)
plt.show()
```

![image](https://user-images.githubusercontent.com/52392004/193434510-d6bac07d-4e0b-4a5b-8be9-5c4e44df9421.png)


## Reference 

[딥러닝 텐서플로 교과서 - 서지영, 길벗](https://github.com/gilbutITbook/080263)
