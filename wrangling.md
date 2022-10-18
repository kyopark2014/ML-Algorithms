# Data Wrangling


### Data Loading

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```

[Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/student+performance)의 경우에 [student-performance-data-set](https://github.com/kyopark2014/ML-Algorithms/blob/main/wrangling.md#student-performance-data-set)와 같이 하나의 column에 데이터가 모여 있습니다. 이런 경우에 아래와 같이 분리할 수 있습니다. 

```python
df = pd.read_csv('https://raw.githubusercontent.com/rickiepark/handson-gb/main/Chapter10/student-por.csv', sep=';')
df.head()
```

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/195976014-622a61a0-ea26-4bc2-962a-bcfe23579508.png)


### Load한 데이터의 특성 파악 

[xgboost-diabetes.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-diabetes.ipynb)와 같이 scikit-learn에서 로드한 diabetes 데이터에서 'X'에 대한 속성 확인을 아래와 같이 info() method를 수행할 수 있습니다. 마찬가지로 describe(), head(), value_counts() 등을 사용할 수 있습니다. 

```python
from sklearn import datasets
X, y = datasets.load_diabetes(return_X_y=True)

import pandas as pd
pd.DataFrame(X).info()
```

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/195358289-50927a7c-ad6b-4f80-b231-abd9919d24b3.png)


### Target의 value count 하기

[xgboost-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-census.ipynb)와 같이, Panda로 읽은 데이터에서 Target의 현황을 value_counts()로 확인할 수 있습니다. 

```python
import pandas as pd
df_census = pd.read_csv('census_income_cleaned.csv')
df_census.head()

df_census['income_ >50K'].value_counts()
```

이때의 결과는 아래와 같습니다. 

```python
0    24719
1     7841
Name: income_ >50K, dtype: int64
```

### Feature의 value를 replace 하기

[xgboost-higgs-boson.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-higgs-boson.ipynb)와 같이 feature의 값을 다른 값으로 바꿀 수 있습니다. 먼저 label열의 값을 확인하면 아래와 같이 's', 'b'를 가지고 있습니다.

![image](https://user-images.githubusercontent.com/52392004/195362960-2d54bc7c-56bc-4f35-ac18-0d2996facdac.png)

이것을 아래와 같이 replace 합니다. 

```python
df['Label'].replace(('s', 'b'), (1, 0), inplace=True)
```

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/195363305-a436de67-9855-496d-9fec-4176b35591c6.png)


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

### Timestamp

[wrangling-cab-rides.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-cab-rides.ipynb)와 같이 Timestamp에서 date 추출은 아래와 같이 수행합니다. 더불어 datatime을 이용하여 아래와 같이 'month', 'hour', 'dayofweek'를 추출할 수 있습니다. 

```python
df['date'] = pd.to_datetime(df['time_stamp']*(10**6))

import datetime as dt
df['month'] = df['date'].dt.month
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
```

이것으로 아래와 같이 weekend와 rush_hour를 계산할 수 있습니다. 

```python
def weekend(row):
    if row['dayofweek'] in [5,6]:
        return 1
    else:
        return 0

df['weekend'] = df.apply(weekend, axis=1)

def rush_hour(row):
    if (row['hour'] in [6,7,8,9,15,16,17,18]) & (row['weekend'] == 0):
        return 1
    else:
        return 0

df['rush_hour'] = df.apply(rush_hour, axis=1)
```

이때의 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/195973003-a5bac706-e074-4ea2-b481-ed282550575f.png)


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

### 범위를 분류하기 

[xgboost-titanic.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/ml-stragegy/src/xgboost-titanic.ipynb)과 같이 요금의 범위를 분류할 수 있습니다. 

#### 정해진 범위로 분류 

아래와 같이 요금(Fare)를 가지고 Level을 구분할 수 있습니다. 

```python
train.loc[train['Fare'] <= 20, 'Level'] = 'L1'
train.loc[(train['Fare'] > 20) & (train['Fare'] <= 40), 'Level'] = 'L2'
train.loc[(train['Fare'] > 40) & (train['Fare'] <= 60), 'Level'] = 'L3'
train.loc[(train['Fare'] > 60) & (train['Fare'] <= 80), 'Level'] = 'L4'
train.loc[train['Fare'] > 80, 'Fare'] = 'L4'

train = train.drop(['Fare'], axis=1)

train['Level'].value_counts()
```
#### 데이터의 분포로 분류 

데이터의 분포를 4개로 나누어서, level을 붙일 수 있습니다. 

```
def get_categorise(df):
    return pd.qcut(df, q=4, labels = ['low','medium','high','very_high'])
    
FareLevel = get_categorise(train['Fare'])
FareLevel.value_counts()

train['Fare'] = FareLevel
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


### 우버와 리프트의 텍시 데이터 

[Uber & Lyft Cab prices](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices)의 데이터셋은 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/195972136-19e63d75-91cb-4166-a746-7b56fde41743.png)


[wrangling-cab-rides.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-cab-rides.ipynb)을 이용하여 수정한 dataset은 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/195972276-cf4d1719-b617-48a7-9486-33ec287d4e29.png)


### Student Performance Data Set

[Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/student+performance)의 데이터는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/195975998-59d29719-67a3-4994-9ec9-ab92f6925d32.png)

이것을 [wrangling-student-Performance](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/wrangling-student-por.ipynb)을 이용하여 아래와 같이 수정할 수 있습니다.

![image](https://user-images.githubusercontent.com/52392004/195976251-c6ada97b-a416-434c-ac89-f1b649c2698c.png)


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
