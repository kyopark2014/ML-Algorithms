# ML-Algorithms

## AI / ML

- 인공지능(AI, Artificial Intelligence): 사람처럼 학습하고 추론할 수 있는 지능을 가진 시스템을 만드는 기술

- 머신러닝(ML, Machine Learning): 규칙을 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘 연구, 예: Sciket-learn

- 딥려닝(DL, Deep Learning): 인공 신경망, 예: TensorFlow, PyTorch



## Binary Classification (HelloWorld)

[KNN을 이용한 binary classification](https://github.com/kyopark2014/ML-Algorithms/blob/main/helloworld.md)에서는 기본 이진분류를 노트북으로 구현합니다. 


- score() 호출시 결과는 결정계수로소 1에 가까울수록 예측이 잘된것임을 알 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/185774224-2209e555-c3ed-4d79-b5e7-d20bef381bc1.png)


## 데이터전처리 

### 표준점수 

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

## Regression

Regression은 예측하고 싶은 종속변수가 숫자일때 사용하는 머신러닝 방법입니다. [Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/regression.md)에서는 Regression에 대한 기본 설명 및 구현하는 코드를 예제로 설명합니다. 



#### 모델 평가

일반적으로 train set의 score가 test set보다 조금 높음

- 과대적합(Overfitting): 모델의 train set 성능이 test set보다 훨씬 높은 경우 
- 과소적합(Underfitting): train set와 test set 성능이 모두 낮거나, test set 성능이 오히려 더 높은 경우
- 특성공학(Feature Engineering): 주어진 특성을 조합하여 새로운 특성을 만드는 과정

 

#### 규제 (Regularization)

모델이 과적합 되게 학습하지 않고 일반성을 가질 수 있도록 파라미터값에 제약을 주는것을 말합니다. L1 규제(Lasso), L2 규제(Ridge), alpha 값으로 규제량을 조정합니다. 

<img width="423" alt="image" src="https://user-images.githubusercontent.com/52392004/185773329-8b542165-3c41-42d9-ba0f-e437a2f9f811.png">


- Ridge: 계수를 제곱한 값을 기준으로 규제를 적용

```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
```

alpha를 매개변수로 규제의 강도를 조절할 수 있습니다. 이때, alpha값이 크면 규제 강도가 세짐으로 계수값을 더 줄이고 좀 더 과대적합 해소가 가능합니다. 

```python
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델을 만듭니다
    ridge = Ridge(alpha=alpha)
    # 릿지 모델을 훈련합니다
    ridge.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
```

아래는 릿지 회귀의 한 예입니다. 

<img width="284" alt="image" src="https://user-images.githubusercontent.com/52392004/185773607-69cefcfb-e931-47c6-b9ff-6f2045015674.png">


- Lasso: 계수의 절대값을 기준으로 규제를 적용 

```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```

alpha를 매개변수로 규제의 강도를 조절할 수 있습니다. 이때, alpha값이 크면 규제 강도가 세짐으로 계수값을 더 줄이고 좀 더 과대적합 해소가 가능합니다. 

```python
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 라쏘 모델을 만듭니다
    lasso = Lasso(alpha=alpha, max_iter=10000)
    # 라쏘 모델을 훈련합니다
    lasso.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
```

## 각종 유용한 라이브러리

- [Numpy](https://github.com/kyopark2014/ML-Algorithms/blob/main/numpy.md)로 데이터를 준비합니다. 


## [Amazon SageMaker Built-in Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)

### Predict if an item belongs to a category: an email spam filter

. Supervised Learning

. Problem type: Binary/multi-class classification

. 데이터는 table 형태로 제공  


#### Classification Algorithms 

. Factorization Machines Algorithm, 

. [kNN (K - Nearest Neighbor)](https://github.com/kyopark2014/ML-Algorithms/blob/main/KNN.md)은 Euclidean distance를 이용하여 가장 가까운 k개의 sample을 선택하여, 해당 sample의 결과값으로 예측할 수 있습니다. 

. Linear Learner Algorithm

. XGBoost Algorithm


. [Machine Learning의 Classification 방법에 따른 특징](https://en.wikipedia.org/wiki/MNIST_database)은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/162556347-9d57ea09-1741-4645-a785-82b27466e8a2.png)




