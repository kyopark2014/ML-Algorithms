# Notebook Commends

## Data Loading

```python
import pandas as pd
pd.options.display.max_rows=20
pd.options.display.max_columns=10

data = pd.read_csv('../dataset/train.csv')

data.head()
data.describe()
data.info()
```

파일을 github에서 바로 로드하는 방법은 아래와 같습니다. 

- 경로조정전: https://github.com/Suji04/ML_from_Scratch/blob/master/Breast_cancer_data.csv
- 경로조정후: https://raw.githubusercontent.com/Suji04/ML_from_Scratch/master/Breast_cancer_data.csv

```python
datapath = "https://raw.githubusercontent.com/Suji04/ML_from_Scratch/master/Breast_cancer_data.csv"
import pandas as pd
data = pd.read_csv(datapath)
```

### Elapsed Time

시간 측정하는 방법은 아래와 같습니다. 

```python
import time
start = time.time()

# Operation

print('Elased time: %0.2fs' % (time.time()-start))
```

### train_test_split

Train/Test dataset 분리를 합니다. 

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
```

### accuracy_score

[xgboost-heart-desease.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-heart-desease.ipynb)와 같이 분류에서 score(accuracy)를 확인합니다. 

```python
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)

print('Accuracy: %0.2f' % (score))
```

이때의 결과는 아래와 같습니다. 

```python
Accuracy: 0.84
```

### cross_val_score

#### Classification 

[xgboost-heart-desease.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-heart-desease.ipynb)와 같이 교차검증(cross validation)으로 분류에서 Accuacy를 구합니다. 

```python
from sklearn.model_selection import cross_val_score
import numpy as np
​
scores = cross_val_score(model, X, y, cv=5)
​
print('Accuracy:', np.round(scores, 2))
print('Avg. Accuracy: %0.2f' % (scores.mean()))
```

#### Regression

[xgboost-heart-desease.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-heart-desease.ipynb)와 같이 교차검증으로 회귀에서 RMSE를 구할 수 있습니다. 

```python
from sklearn.model_selection import cross_val_score

model = LinearRegression()

scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=50)

rmse = np.sqrt(-scores)

print('RMSE:', np.round(rmse, 2))
print('Avg. RMSE: %0.2f' % (rmse.mean()))
```


### classification_report

[xgboost-heart-desease.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-heart-desease.ipynb)와 같이 분류에서 accuracy, precision, recall, f1-score을 아래와 같이 확인합니다. 

```python
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred = y_pred))
```

이때의 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/195368069-24441412-b3d4-43cb-a9be-2ebd2f888d06.png)

### Feature Importance

[xgboost-heart-desease.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-heart-desease.ipynb)와 같이 Feature Importance의 값을 아래와 같이 확인합니다. 

```python
print(model.feature_importances_)
```

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/195369225-fc8ea777-a16e-436e-ba7f-2249809c8937.png)

중요도를 그림으로 표시합니다. 

```python
import xgboost as xgb

feature_data = xgb.DMatrix(X_test)
model.get_booster().feature_names = feature_data.feature_names
model.get_booster().feature_types = feature_data.feature_types

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15, 8))
xgb.plot_importance(model, ax=ax, importance_type='gain')
```
이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/195369423-47f2340a-bcb2-4aa2-a656-d847f3825595.png)

### Model의 parameter 확인하기

[random-forest-census.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/random-forest-census.ipynb)와 같이 모델의 parameter들을 확인합니다. 

```python
best_model.get_params(deep=True)
```
이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/195372145-6321b2fd-04d2-46b8-8e87-37fd8f8fe0af.png)

