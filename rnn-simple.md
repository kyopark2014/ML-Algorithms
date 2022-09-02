# 순환신경망(RNN) 실습 

아래에서는 [혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)의 텍스트 분석관련 셈플을 중심으로 설명합니다. 

## Simple RNN 

[상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/rnn-simple.ipynb)에 대해 설명합니다.

1) [IMDB](https://www.imdb.com/) 데이터를 로드합니다.

읽어올때 토큰수(어휘수)는 num_words로 500만큼을 읽어오도록 설정합니다. 이때 데이터에서 0은 padding, 1은 문장시작을 의미하고, 어휘사전에 없는 수라면, 2로 변환하고 

```python
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)
```    



## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
