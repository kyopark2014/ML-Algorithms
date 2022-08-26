# Random Forest

다수의 decision tree의 분류 결과를 취합해서 최종 예측값을 결정하는 앙상블 학습입니다. [상세코드](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/random_forest.ipynb)에서 아래의 코드를 보여줍니다.

- 앙상블 학습(Ensemble learning): 여러 개의 분류기를 생성하고, 그 예측을 결합함으로써 보다 정확한 예측을 도출하는 기법을 의미합니다. 

## 구현방법

1) Bootstrap dataset 생성
2) Feature중 Random하게 n개 선택 후, 선택한 feature로 결정트리(Decision Tree) 생성후 반복
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

여기서 wine은 아래의 feature를 가지고 





## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)
