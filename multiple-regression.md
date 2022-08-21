# 다중회귀(Multiple Regression)

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)을 참조하여, 여러개의 특성(길이, 두께, 높이)을 사용하여 예측을 수행할 수 있습니다.  

## 농어 예측

1) 데이터를 아래와 같이  pandas로 읽어옵니다. 

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
