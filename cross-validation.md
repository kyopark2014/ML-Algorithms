# Cross Validation

데이터를 train과 test dataset으로 분할하는 방법에 따라 결과(예 RMSE)가 달라질 수 있다면, k-fold cross validation을 사용할 수 있습니다. 데이터를 trainr과 test dataset으로 여러번 나누어서 각 결과를 평균하는 것입니다. 분할횟수인 k를 fold라고 합니다. fold의 횟수의 기본값은 5입니다. k가 클수록 평균 점수의 이상치에 덜 민감합니다. 

### k-fold cross validation

k-fold cross validation를 이용한 교차검증을 할수 있는데, 전체 데이터셋을 k개의 fold로 분할한 후 각 iteration마다 Validation set을 겹치지 않게 할당하고 Accuracy의 평균을 모델의 성능으로 계산합니다. 

<img src="https://user-images.githubusercontent.com/52392004/186666830-cae6a8f1-43d8-4d07-8066-8979927df07f.png" width="600">


[cross_validation.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/cross_validation.ipynb)의 코드를 아래와 같이 설명합니다. 

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
- score_time: 모델을 검증하는 시간
- test_score: 검증폴더의 점수

최종 [결정계수](https://github.com/kyopark2014/ML-Algorithms/blob/main/evaluation.md#coefficient-of-determination)는 아래와 같이 평균값을 이용해 구할 수 있습니다. 여기서 사용한 fold의 수는 5개(기본값)입니다.

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

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
