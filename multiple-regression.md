# 다중회귀(Multiple Regression)

## 다중회귀를 이용한 농어 예측

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)에는 여러개의 특성(길이, 두께, 높이)을 사용하여 예측할 때 쓸수 있는데 다중회귀 예제를 아래처럼 제공하고 있습니다. 

[Multiple Regression 상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/feature_enginnering.ipynb)에서는 농어의 무게를 예측하는 예제입니다. 


데이터를 아래와 같이 pandas로 읽어올 수 있습니다. [CSV 파일](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/perch_full.csv)에는 length, height, width로 도미(perch) 데이터가 정리되어 있습니다.

```python
import pandas as pd

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)
```

이때 읽어진 데이터의 형태는 아래와 같습니다. 

```java
[[ 8.4   2.11  1.41]
 [13.7   3.53  2.  ]
 [15.    3.82  2.43]
 [16.2   4.59  2.63]
 [17.4   4.59  2.94]
 [18.    5.22  3.32]
 [18.7   5.2   3.12]
 [19.    5.64  3.05]
```

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
