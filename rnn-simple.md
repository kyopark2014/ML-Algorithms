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


 

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
