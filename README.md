# ML-Algorithms

## AI / ML

- 인공지능(AI, Artificial Intelligence): 사람처럼 학습하고 추론할 수 있는 지능을 가진 시스템을 만드는 기술

- 머신러닝(ML, Machine Learning): 규칙을 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘 연구, 예: Sciket-learn

- 딥려닝(DL, Deep Learning): 인공 신경망, 예: TensorFlow, PyTorch



## Binary Classification (HelloWorld)

[kNN(k-Nearest Neighbors)을 이용한 binary classification](https://github.com/kyopark2014/ML-Algorithms/blob/main/helloworld.md)에서는 기본 이진분류를 노트북으로 구현합니다. 


### 결정계수 (Coefficient of determination)

score() 호출시 결과는 분류(Classification)에서는 정확도(정답을 맞춘 개숫의 비율)을 의미하고, 회귀(Regression)에서는 결정계수(1에 가까울수록 예측이 잘된것)를 의미합니다.

![image](https://user-images.githubusercontent.com/52392004/185774224-2209e555-c3ed-4d79-b5e7-d20bef381bc1.png)

아래는 K Neighbors Classifier을 이용한 예제입니다. 여기서 score는 결정계수를 의미합니다. 

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))
```

## 데이터전처리 

[Preprocessing](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md)에서는 표준점수(z)를 이용한 데이터 정규화 및 Train/Test Dataset을 준비하는 과정을 설명합니다. 

## Regression

Regression은 예측하고 싶은 종속변수가 숫자일때 사용하는 머신러닝 방법입니다. [Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/regression.md)에서는 Regression에 대한 기본 설명 및 구현하는 코드를 예제로 설명합니다. 

## Classification

[분류 알고리즘 (Classification)](https://github.com/kyopark2014/ML-Algorithms/blob/main/classification.md)을 통해 Sample을 몇개의 Class중에 하나로 분류할 수 있습니다.

## 모델 평가

일반적으로 train set의 score가 test set보다 조금 높음습니다.

- 과대적합(Overfitting): 모델의 train set 성능이 test set보다 훨씬 높은 경우 
- 과소적합(Underfitting): train set와 test set 성능이 모두 낮거나, test set 성능이 오히려 더 높은 경우
- 특성공학(Feature Engineering): 주어진 특성을 조합하여 새로운 특성을 만드는 과정

[규제 (Regularization)](https://github.com/kyopark2014/ML-Algorithms/blob/main/regularization.md)을 이용하여 과대적합을 방지할 수 있습니다. 

Regularization과 Epoch를 비교하면 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/186548434-d12e684a-d139-414a-8fe6-e449b4348354.png)





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




