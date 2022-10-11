# Data Wrangling


### Data Loading

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```

### 누락된 값 확인 

[df_bikes](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-bike.ipynb)에 있는 누락된 값을 출력합니다.

```python
df_bikes[df_bikes.isna().any(axis=1)]
```

누락된 값의 갯수 확인을 합니다.

```python
df_bikes.isna().sum().sum()
```

누락된 data를 삭제하는 방법입니다.

```python
df.dropna(inplace=True)
```

### 범주형 데이터를 수치형으로 변환

Panda의 [get_dummies()](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-census.ipynb)를 사용해 숫자가 아닌 열을 바꿉니다. get_dummies() 사용시 데이터프레임 용량이 증가하므로 메모리 사용량을 확인해야 합니다. 반면에 희소행열(sparse matrix)라면 희소한 값을 1로 저장하므로 메모리를 줄일 수도 있습니다. 

```python
df_census = pd.get_dummies(df_census)
```

### 특정 열 삭제하기 

아래와 같이 padas의 drop()을 이용하여 특정 열을 삭제할 수 있습니다. 여기서 axis=0로 행, axis=1로 열을 지정할 수 있습니다. 

```python
data = data.drop('casual', 'registered'], axis=1)
```

### CSV 파일로 저장하기 

향후에 사용할 수 있도록 dataset을 csv 파일로 저장할 수 있습니다. 여기서 'index=False'로 지정하면 데이터프레임의 인덱스가 하나의 열로 저장되는것을 막아줍니다. 

```python
data.to_csv('data_cleaned.csv', index=False)
```

### Feature and Target

아래와 같이 데이터의 마지막이 타겟(target)일 경우에, data의 마지막 열을 제외한 데이터를 특징(Feature)인 X로, 마지막 열을 타겟(Target)인 y로 분리할 수 있습니다. 

```python
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
```


### 특정 열의 값 확인하기 

아래와 같이 pandas의 unique()를 이용하여 'class' 열의 값을 확인 할 수 있습니다. 아래와 같이 [decision_tree.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/decision_tree.ipynb)에서 사용할 수 있습니다. 

```python
print(pd.unique(wine['class']))
```

이때의 한 예는 아래와 같이 0, 1로 구성되어 있으므로, classification의 target으로 활용할 수 있습니다. 

```java
[0. 1.]
```

### Shuffle

dataset을 섞습니다. 

```python
from sklearn.utils import shuffle
df = shuffle(df, random_state=2)

df.head()
```

## Wrangling Examples

### Bike Sharing

수정전의 [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/194686814-5bb6e301-313a-42dd-9bad-eb4ed012e99b.png)

학습할 수 있도록 [wrangling-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-bike.ipynb)을 이용하여 수정한 dataset은 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/194686869-ea8d9fbb-c094-4924-a0c7-6831e75a3256.png)




### Census Income

수정전의 [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)은 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/194686432-8df77926-381b-4ade-8899-4b261bde5944.png)

학습할 수 있도록 [wrangling-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-census.ipynb)
을 이용하여 수정한 dataset은 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/194688720-b70da40d-3bec-41e7-b49b-b2ff09bc52f4.png)


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
