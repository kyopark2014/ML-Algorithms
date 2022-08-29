# ML-Algorithms

## AI / ML

- 인공지능(AI, Artificial Intelligence)은 사람처럼 학습하고 추론할 수 있는 지능을 가진 시스템을 만드는 기술을 의미합니다.

- 머신러닝(ML, Machine Learning)은 규칙을 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습합니다.

- [딥러닝(DL, Deep Learning)](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md)은 인공 신경망에 기반한 머신러닝으로서 TensorFlow, PyTorch가 해당됩니다. 

<img width="381" alt="image" src="https://user-images.githubusercontent.com/52392004/187052186-e3c810ed-1487-425e-8e91-93307dccfbc9.png">



## Binary Classification (HelloWorld)

[kNN(k-Nearest Neighbors)을 이용한 binary classification](https://github.com/kyopark2014/ML-Algorithms/blob/main/helloworld.md)에서는 기본 이진분류를 노트북으로 구현합니다. 




## 데이터전처리 

[Preprocessing](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md)에서는 [표준점수(z)를 이용한 데이터 정규화](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md#%ED%91%9C%EC%A4%80%EC%A0%90%EC%88%98-standard-score%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%A0%95%EA%B7%9C%ED%99%94) 및 [Train/Test Dataset](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md#train%EA%B3%BC-test-dataset)을 준비하는 과정을 설명합니다. 

## Machine Learning

### 1) Supervised Learning

#### Regression

Regression은 예측하고 싶은 종속변수가 숫자일때 사용하는 머신러닝 방법입니다. [Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/regression.md)에서는 Regression에 대한 기본 설명 및 구현하는 코드를 예제로 설명합니다. 

#### Classification

[Classification](https://github.com/kyopark2014/ML-Algorithms/blob/main/classification.md)을 통해 Sample을 몇개의 Class중에 하나로 분류할 수 있습니다.

### 2) Unsupervised Learning

#### Clustering: k-Means

[k-Means](https://github.com/kyopark2014/ML-Algorithms/blob/main/k-means.md)는 비지도학습(Unsupervised Learning)으로 정답 label이 없는 데이터에서 유사도를 기준으로 k개의 군집으로 분류할 수 있습니다. 

#### Dimensionally Reduction: PCA (Principal Component Analysis)

[PCA](https://github.com/kyopark2014/ML-Algorithms/blob/main/pca.md)를 이용해 데이터의 분산(variance)을 최대한 보존하면서 축소된 데이터를 학습데이터로 사용할수 있습니다. 


## Deep Learning

[Deep Learning](https://github.com/kyopark2014/ML-Algorithms/blob/main/deep-learning.md)에 대해 예제 중심으로 설명합니다. 

#### Image Classification

Deep Learning을 이용한 [Image 분류](https://github.com/kyopark2014/ML-Algorithms/blob/main/image_classification.md)방법에 대해 설명합니다.


## 모델 평가

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

### 모델 적합 

모델의 복잡도가 증가 할수록 보통 Training Error는 감소합니다.

![image](https://user-images.githubusercontent.com/52392004/187076574-013e9c72-36af-4e6f-a2ab-54872eb19622.png)


일반적으로 train set의 score가 test set보다 조금 높음습니다.

- 과대적합(Overfitting): 모델의 train set 성능이 test set보다 훨씬 높은 경우입니다.
- 과소적합(Underfitting): train set와 test set 성능이 모두 낮거나, test set 성능이 오히려 더 높은 경우 입니다.

아래와 같이 Linear regression에서 과적합(Overfitting)이 발생할 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/187076564-174edbf0-26ba-434f-937c-6a8e8e9e46e7.png)



- 특성공학(Feature Engineering): 주어진 특성을 조합하여 새로운 특성을 만드는 과정입니다. 

[규제 (Regularization)](https://github.com/kyopark2014/ML-Algorithms/blob/main/regularization.md)을 이용하여 과대적합을 방지할 수 있습니다. 

Regularization과 Epoch를 비교하면 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/186548434-d12e684a-d139-414a-8fe6-e449b4348354.png)


### Hyperparameter Optimization (HPO)

[Hyperparameter Optimization](https://github.com/kyopark2014/ML-Algorithms/blob/main/hyperparameter-optimization.md)에서는 머신러닝 학습 알고리즘별 최적의 Hyperparameter 조합을 찾아가는 과정을 의미 합니다. 

## ML application

![image](https://user-images.githubusercontent.com/52392004/187052254-d12b4cb5-8835-457c-80d9-0c279de9f9dc.png)


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




