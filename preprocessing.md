# 데이터 전처리 (Preprocessong)

### 표준점수 (Standard score)를 이용한 정규화

특성값을 일정한 기준으로 맞추는 작업이 필요합니다. 이때 Z점수(표준점수, standard score)를 사용하여 각 데이터가 원점에서 몇 표준편차만큼 떨어져 있는지 나타내므로, 특성값의 크기와 상관없이 동일한 조건으로 비교가 가능합니다. 

![image](https://user-images.githubusercontent.com/52392004/185774334-00e687e7-226e-410b-b6dd-85989f5147e1.png)

아래와 같이 scikit-learn을 이용하여 표준점수로 변환할 수 있습니다. 

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)    

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

### Train과 Test Set 

Train과 Test의 Set이 골고루 섞이지 않으면 Sampling Bias가 발생할 수 있으므로, 준비된 데이터 중에 일부를 떼어 train set과 test set으로 활용합니다. 아래에서는 scikit-learn의 train_test_split을 사용하는 방법을 보여줍니다. 

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


### k-fold cross validation를 이용한 교차검증

전체 데이터셋을 k개의 fold로 분할한 후 각 iteration마다 Validation set을 겹치지 않게 할당하고 Accuracy의 평균을 모델의 성능으로 계산합니다. 

![image](https://user-images.githubusercontent.com/52392004/186666830-cae6a8f1-43d8-4d07-8066-8979927df07f.png)

1) 데이터를 준비합니다. 

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
print(train_input.shape, test_input.shape)

(5197, 3) (1300, 3)
```

2) Decision Tree를 이용하여 cross validation을 수행합니다.

```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)

from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)

{"fit_time": array([0.00589418, 0.00558877, 0.00579906, 0.00572824, 0.00548029]),
"score_time": array([0.00050735, 0.00045037, 0.0004468 , 0.0004673 , 0.00045609]),
"test_score": array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}
```

여기서, score 분석을 위해 아래의 값을 참고할 수 있습니다. 

- fit_tme: 모델을 훈련하는 시간
- socre_time: 모델을 검증하는 시간
- test_score: 검증폴더의 점수

최종 결정계수는 아래와 같이 구할 수 있습니다. 여기서 사용한 fold의 수는 5개(기본값)입니다.

```python
import numpy as np

print(np.mean(scores['test_score']))

0.855300214703487
```

fold의 숫자를 조정할 때에는, 회귀모델일 경우에는 KFold 분할기를 사용하고, 분류모델일 때에는 StratifiedKFold을 사용합니다. 아래는 10개의 fold로 나누는 splitter를 이용하였습니다. 

```python
from sklearn.model_selection import StratifiedKFold

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))

0.8574181117533719
```

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)





