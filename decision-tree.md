# 결정트리 (Decision Tree)

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)을 참조하여 [Decision Tree](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/decision_tree.ipynb)에서는 결정트리에 대해 예를 보여주고 있습니다. 


## 장단점

#### 장점 

- 결과를 해석하고 이해하기 쉽습니다.
- 분류/회귀에 모두 사용 가능합니다.
- Data preprocessing(Scailing) 이 거의 필요하지 않습니다.
- Outliner에 민감하지 않습니다.
- 연속형 변수(Numinical feature), 범주형 변수(Category)에 모두 적용 가능합니다.
- 대규모의 데이터 셋에서도 잘 동작합니다. 

#### 단점

- 데이터의 특성이 특정 변수에 수직/수평적으로 구분되지 못할 경우 분류률이 떨어지고 트리가 복잡해집니다. 즉, Tree의 depth가 깊어질수록 느려집니다.

![image](https://user-images.githubusercontent.com/52392004/186661527-5362a5ae-894a-4777-8666-07eb6347c0f0.png)

- Overfitting (High variance) 위험이 있으므로 규제(Regularization) 필요합니다. 대표적인 가지치기 방법은 "Pruning(가지치기)"으로 max_depth를 이용합니다. 

## Decision Tree 예제

[Decision Tree](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/decision_tree.ipynb)는 아래의 전체 코드를 가지고 있습니다.

1) 데이터를 준비합니다. 

데이터를 읽어옵니다. 여기에는 "alcohol", "sugar", "pH"라는 3개의 column이 있습니다.

```python 
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

wine.head()
```


![image](https://user-images.githubusercontent.com/52392004/186591846-a6ee86b4-6c7a-4036-8a14-b896ce1a71e0.png)

Train / Test Set을 만들고 정규화를 합니다.

```python
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

2) Decision Tree를 아래와 같이 구합니다.

scikit-learn의 DecisionTreeClassifier로 결정계수를 아래처럼 구할 수 있습니다. 

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

0.996921300750433
0.8592307692307692
```

이 값은 [Logistric Regression 결과](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/logistic-regression-low-accuracy.ipynb)보다는 좋지만 과대적합(Overfit)인 결과를 얻습니다. 

이때의 트리구조를 sciket-learn의 plot_tree로 그리면, 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/186592557-6e7b5a12-e38a-4d6d-add1-c3f8e4fcbd3c.png)

3) 가지치기 (Pruning)

가지치기를 위해 max_depth를 3으로 설정했을때 아래와 같습니다. 

```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

0.8454877814123533
0.8415384615384616
```

이것을 plot_tree로 그리면 아래와 같습니다. 

```python
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```

이때, 아래 그림과 같이 5197개의 sample중에 1141개 sample이 레드 와인으로 분류됩니다.

![image](https://user-images.githubusercontent.com/52392004/186660116-df754bd8-0946-439e-9ae0-f290672b91d5.png)


아래는 정규화를 하지 않은 경우인데, 거의 동일한 결과를 얻고 있습니다.

```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

0.8454877814123533
0.8415384615384616
```



## Criterion 매개변수 

결정트리에서 노드를 분할하는 기준에는 Gini Impurity와 Entropy Imputiry가 있습니다. scikit-learn에서는 기본값으로 Gini impurity을 사용합니다. 

아래는 Max Depth가 1인 결정트리를 아래처럼 그릴수 있습니다.

```python
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```

아래와 같이 Max Depth가 1인 결정트리를 그릴수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/186655980-8b0674b7-2b0e-4c69-af3b-fed6271447dd.png)



### Gini Impurity

Gini Impurity은 정답이 아닌 값이 나올 확율을 의미 합니다. 

![image](https://user-images.githubusercontent.com/52392004/186560214-844f1030-a80b-4190-a9cb-1b6151f01cde.png)

### Entropy Imputiry

Entropy Imputiry는 정보의 불확실성 또는 무질서도를 의미합니다. 

![image](https://user-images.githubusercontent.com/52392004/186560305-1651f4e1-880b-49e5-bea4-bf9d00bb6dd6.png)

### Information Gain 

정보이득(Information Gain)은 부모노드가 가진 정보량에서 자식노드들의 정보량을 뺀 차이입니다.부모노드와 자식노드의 정보량의 차이가 없을때, 트리는 분기 split을 멈추게 됩니다. 

![image](https://user-images.githubusercontent.com/52392004/186560390-350d25b2-2f8d-4d06-ac66-99943b6e3e35.png)

## feature_importances

결정 트리 모델의 특성중요도(feature importances) 속성으로 특성 중요도를 알수 있습니다. 아래와 같이 "alcohol", "sugar", "pH"의 중요도는 0.15210271, 0.70481604, 0.14308125이고, 이것의 합은 1입니다. 즉, "sugar"가 가장 중요한 특성으로 불순도에 줄이는데 가장 큰 역할하고 있으므로, 직관적으로 문제를 이해하는데 도움이 됩니다. 

```python
print(dt.feature_importances_)

[0.15210271 0.70481604 0.14308125]
```



## Decision Tree의 Hyperparameter

‘min’값을 높이거나 ’max’값을 줄이면 모델에 규제가 커져 이용하여 과대적합(overfit)을 막을 수 있습니다. 

- min_samples_split: 분할되기 전에 노드가 가져야 하는 최소 샘플 수
- min_samples_leaf: 리프노드가 가지고 있어야 할 샘플 수
- min_weight_fraction_leaf: min_samples_leaf과 같지만 전체 샘플에서 클래스 별 샘플 수 비율을 고려
- max_leaf_nodes: 리프노드의 최대 수
- max_features: 각 노드에서 분할에 사용할 특성의 최대 수
- min_impurity_decrease: 분할로 얻어질 최소한의 불순도 감소량
- max_depth: 트리의 최대 깊이 (루트 노드 깊이=0)



## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[sklearn.tree.DecisionTreeClassifie](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

