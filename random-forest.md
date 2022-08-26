# Random Forest

다수의 decision tree의 분류 결과를 취합해서 최종 예측값을 결정하는 앙상블 학습입니다. [상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/random_forest.ipynb)에서 아래의 코드를 보여줍니다.

- 앙상블 학습(Ensemble learning): 여러 개의 분류기를 생성하고, 그 예측을 결합함으로써 보다 정확한 예측을 도출하는 기법을 의미합니다. 

## 구현방법

1) Bootstrap dataset 생성합니다.

2) Feature중 Random하게 n개 선택 후, 선택한 feature로 결정트리(Decision Tree) 생성후 반복합니다.

   - scikit-learn는 100개의 결정트리를 기본값으로 생성하여 사용 

3) Inference: 모든 Tree를 사용하고 분류한 후 최종 Voting합니다.

4) Validation: Bootstrap을 통해 랜덤 중복추출을 했을 때, Original dataset의 샘플 중에서 Bootstrap 과정에서 선택되지 않은 샘플들을 OOB(Out-of-Bag) 샘플이라 하고, 이 샘플들을 이용하여 Validation 수행합니다.

## 코드분석

1) 데이터를 준비합니다.


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
```

여기서 wine은 아래의 feature를 가지고 있습니다.

```python
wine.head()
```

![image](https://user-images.githubusercontent.com/52392004/186914946-170ca7d9-930e-4994-8135-0114537fc98f.png)


2) Random Forest 방식으로 Training을 수행합니다. 

Train/Test dataset의 결과를 보면 아래와 같이 과대적합이지만, [결정트리로 수행한 결과](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)보다는 좋은 결과를 얻고 있습니다. 

여기서, [k-fold cross validation를 이용한 교차검증](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md#k-fold-cross-validation%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D)을 cross_validate()을 이용해 수행하고 있습니다. n_splits를 지정하고 있지 않으므로 기본값인 5번 수행하고 있습니다. 

```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

0.9973541965122431 0.8905151032797809
```

alcohol, sugar, pH의 중요도는 
ㅇㅏ래와 


```python
rf.fit(train_input, train_target)
print(rf.feature_importances_)

[0.23167441 0.50039841 0.26792718]
```


```python
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

rf.fit(train_input, train_target)
print(rf.oob_score_)

0.8934000384837406
```

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
