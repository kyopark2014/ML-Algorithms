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


### k-fold cross validation

전체 데이터셋을 k개의 fold로 분할한 후 각 iteration마다 Validation set을 겹치지 않게 할당하고 Accuracy의 평균을 모델의 성능으로 계산합니다. 

![image](https://user-images.githubusercontent.com/52392004/186666830-cae6a8f1-43d8-4d07-8066-8979927df07f.png)

## Hyperparameter Optimization (HPO)

[Hyperparameter Optimization](https://github.com/kyopark2014/ML-Algorithms/blob/main/hyperparameter-optimization.md)에서는 머신러닝 학습 알고리즘별 최적의 Hyperparameter 조합을 찾아가는 과정을 의미 합니다. 

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)





