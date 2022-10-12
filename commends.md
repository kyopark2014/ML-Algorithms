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

[xgboost-higgs-boson.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-higgs-boson.ipynb)와 같이 분류에서 score(accuracy)를 확인합니다. 

```python
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)

print('Accuracy:', np.round(score, 2))
```

이때의 결과는 아래와 같습니다. 

```python
Accuracy: 0.84
```

### cross_val_score

[xgboost-higgs-boson.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-higgs-boson.ipynb)와 같이 교차검증(cross validation)으로 분류에서 Accuacy를 구합니다. 

```python
from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(best_model, X, y, cv=5)

print('Accuracy:', np.round(scores, 2))
print('Avg. Accuracy: %0.2f' % (scores.mean()))
```

### classification_report

분류에서 accuracy, precision, recall, f1-score을 아래와 같이 확인합니다. 

```python
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred = y_pred))
```

이때의 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/195368069-24441412-b3d4-43cb-a9be-2ebd2f888d06.png)

