# 데이터 전처리 (Preprocessong)

## Data Loading

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```

#### 특정 열 삭제하기 

아래와 같이 padas의 drop()을 이용하여 특정 열을 삭제할 수 있습니다. 여기서 axis=0로 행, axis=1로 열을 지정할 수 있습니다. 

```python
data = data.drop('casual', 'registered'], axis=1)
```

#### CSV 파일로 저장하기 

향후에 사용할 수 있도록 dataset을 csv 파일로 저장할 수 있습니다. 여기서 'index=False'로 지정하면 데이터프레임의 인덱스가 하나의 열로 저장되는것을 막아줍니다. 

```python
data.to_csv('data_cleaned.csv', index=False)
```

#### 특성과 target 준비하기 

아래와 같이 데이터의 마지막이 target일 경우에, data의 마지막 열을 제외한 데이터를 X로, 마지막 열을 y로 분리할 수 있습니다. 

```python
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
```




## Scaling

### 표준점수 (Standard score)를 이용한 정규화

특성값을 일정한 기준으로 맞추는 작업이 필요합니다. 이때 Z점수(표준점수, standard score)를 사용하여 각 데이터가 원점에서 몇 표준편차만큼 떨어져 있는지 나타내므로, 특성값의 크기와 상관없이 동일한 조건으로 비교가 가능합니다. 

![image](https://user-images.githubusercontent.com/52392004/185774334-00e687e7-226e-410b-b6dd-85989f5147e1.png)

scikit-learn에서는 StandardScaler, MinMaxScaler, RobustScaler Normalizer를 scaler로 제공하는데, 여기서는 StandardScaler를 이용하여, 표준점수로 변환할 수 있습니다. 

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)    

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

## Dataset 

### Train과 Test Dataset 

Train과 Test의 Dataset이 골고루 섞이지 않으면 Sampling Bias가 발생할 수 있으므로, 준비된 데이터 중에 일부를 떼어 train set과 test set으로 활용합니다. 아래에서는 scikit-learn의 train_test_split을 사용하는 방법을 보여줍니다. 

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)
```

- stratify=fish_target: fish_target을 기준으로 섞을때 사용합니다. 
- 기본값은 전체 데이터에서 25%를 test set으로 분리합니다. 

### Splitting dataset

Validation dataset은 모델을 학습한 후 성능 측정을 하는데 사용됩니다. 또한 이것은 Hyperparameter Optimization(HPO)에도 사용되어 집니다. 

![image](https://user-images.githubusercontent.com/52392004/186666166-e9e40b07-adb4-4b4e-8b89-108d101abf61.png)

아래와 같이 scikit-learn의 train_test_split()을 2번 사용하면, test_input(Test Set)으로 20%를 분리하고, 나머지 80%를 다시 80%의 sub_input(Train Set)과 val_input(Test Set)으로 지정할 수 있습니다. Training dataset(sub_input)으로 모델을 학습(Train)을 한후에, Validation dataset(val_input)으로 성능을 측정하거나 HPO를 수행하고, Test dataset으로 최종모델의 성능을 평가합니다. 

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

print(data.shape, sub_input.shape, val_input.shape, test_input.shape)

(6497, 3) (4157, 3) (1040, 3) (1300, 3)
```

### Epoch

전체 dataset에서 일정한 묶음으로 나누어 처리할 수 있는 배치와 훈련의 횟수인 epoch의 선택이 중요합니다. 이때 훈련과정에서 값의 변화를 시각적으로 표현하여 파라메터에 대한 최적의 값을 찾을 수 있어야 합니다. 

Dataset 1000개에 대한 배치 크기가 20이라면, sample 단위 20개마다 모델 가중치를 한번씩 업데이트 시킨다는 의미입니다. 즉, 50번 (=1000/20)의 가중치가 업데이트 됩니다. 이때 epoch가 10이고 모델크기가 20이라면, 가중치를 50번 업데이트하는것을 총 10번 반복한다는 의미입니다.  각 데이터 sample이 총 10번씩 사용되는것이므로, 결과적으로 총 500번 업데이트 됩니다. 

![image](https://user-images.githubusercontent.com/52392004/193483155-3294fbaf-fa2f-4b20-9ead-aa80d13748f0.png)

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[교차 검증(cross validation)](https://m.blog.naver.com/ckdgus1433/221599517834)

[딥러닝 텐서플로 교과서 - 서지영, 길벗](https://github.com/gilbutITbook/080263)

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
