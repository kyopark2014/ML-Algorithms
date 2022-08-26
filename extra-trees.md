# Extra Trees

## 특징 
- 전체 훈련 세트를 사용해 훈련 합니다.
- Split 할 때 무작위로 feature를 선정 합니다.
- Extra Tree는 random하게 노드를 분할하므로, Random Forest보다 일반적으로 더 많은 결정트리를 필요로 하지만, 더 빠릅니다. 

![image](https://user-images.githubusercontent.com/52392004/186904414-39fe90f7-d39e-465a-b40a-9a592b2cc5f9.png)

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

2) Extra tree로 Training을 수행합니다.

[Random Forest 결과](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)와 유사한 결과를 아래와 같이 얻었습니다. 

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_validate

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

0.9974503966084433 0.8887848893166506
```

alcohol, sugar, pH의 중요도는 아래와 같이 sugar의 중요도가 더 높습니다.

```python
et.fit(train_input, train_target)
print(et.feature_importances_)

[0.20183568 0.52242907 0.27573525]
```

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
