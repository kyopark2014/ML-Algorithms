# Hyperparameter Optimization (HPO)

Hyperparameter optimization은 머신러닝 학습 알고리즘별 최적의 Hyperparameter 조합을 찾아가는 과정을 의미 합니다.

x축은 중요한 파라메터이고 y축은 중요하지 않은 파라메터라면, Grid search와 Random Search를 이용하여 최적의 Hyperparameter를 구할 수 있습니다. Grid Search는 9번 시도 했지만, 3개의 시도를 한 효과를 가지므로, Rnadom search가 일반적으로 Grid search보다 더 좋은 결과를 얻습니다. 그리고 Grid search는 Random search보다 느립니다. 

이것은 GridSearchCV 클래스와 RandomizedSearchCV 클래스를 이용해 구할 수 있습니다.

![image](https://user-images.githubusercontent.com/52392004/186670429-43eae8fc-7bc5-4a46-8ae8-91f827474604.png)


## 일반적인 가이드라인

- 알고리즘 별 hyperparameter를 이해합니다.
- 경험적으로 중요한 hyperparameter를 먼저 탐색하고 값을 고정합니다.
- 덜 중요한 hyperparameter를 나중에 탐색합니다.
- 먼저 넓은 범위에 대해 hyperparameter를 탐색하고 좋은 결과가 나온 범위에서 다시 탐색합니다.
- Random Search가 Grid Search에 더 적은 trial로 더 높은 최적화를 기대할 수 있습니다.
- HPO에 test dataset을 사용하지 않고, validation dataset을 사용합니다. 



## 구현 예

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

2) Hyperparameter tuning 없이 Training한 경우는 아래와 같고, 과대적합(Overfiting)되고 있습니다.

```
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

0.996921300750433
0.8584615384615385
```

3) min_impurity_decrease(분할로 얻어질 최소한의 불순도 감소량)을 Hyperparameter tuning 한 경우입니다.

GridSearchCV로 min_impurity_decrease을 튜닝시에 이전보다 개선이 있습니다. 

```python
from sklearn.model_selection import GridSearchCV

params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

0.892053107562055
0.8615384615384616
```

최고일때의 min_impurity_decrease의 값은 0.001입니다.

```python
print(gs.best_params_)

{'min_impurity_decrease': 0.0001}
```

4) 여러개 Hyperparameter tuning 한 경우

거의 동일한 결과를 얻습니다. 

```python
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

0.892053107562055
0.8615384615384616
```

이때의 최적값입니다. 

```python
print(gs.best_params_)

{'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
```

