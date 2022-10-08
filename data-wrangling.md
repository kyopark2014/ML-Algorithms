# Data Wrangling

## Example

#### Bike Sharing

[wrangling-bike.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-bike.ipynb)

## Data Loading

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
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

### 특성과 target 준비하기 

아래와 같이 데이터의 마지막이 target일 경우에, data의 마지막 열을 제외한 데이터를 X로, 마지막 열을 y로 분리할 수 있습니다. 

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
