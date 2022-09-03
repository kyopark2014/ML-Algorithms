# 순환신경망 - GRU

## GRU 

GRU(Gated Recurrent Unit)에 대해 설명합니다. 아래에서는 GRU의 셀의 구조를 보여주고 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/188256227-e21516f9-456b-424d-b23c-02c4794e12b1.png)

이것은 [LSTM](https://github.com/kyopark2014/ML-Algorithms/blob/main/rnn-lstm.md)의 Cell state가 없으며, update와 reset gate를 가지고 있습니다. 


- Update Gate는 LSTM의 Forget Gate와 Input Gate 처럼 동작합니다. 
- Reset Gate는 얼마나 많은 과거 정보를 삭제할지를 결정할 수 있습니다. 

Activation function으로 빨간색은 sigmoid이고 파란색은 tanh 입니다. 

LSTM과 흡사하지만 경량화된 모델입니다. 

## GRU Sample

1) 데이터를 준비합니다. 

```python
import tensorflow as tf

from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)
    
from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
    
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)
```

