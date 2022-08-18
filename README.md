# ML-Algorithms

## AI / ML

- 인공지능(AI, Artificial Intelligence): 사람처럼 학습하고 추론할 수 있는 지능을 가진 시스템을 만드는 기술

- 머신러닝(ML, Machine Learning): 규칙을 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘 연구, 예: Sciket-learn

- 딥려닝(DL, Deep Learning): 인공 신경망, 예: TensorFlow, PyTorch

## 기본 Binary Classification (HelloWorld)

[KNN을 이용한 binary classification](https://github.com/kyopark2014/ML-Algorithms/blob/main/helloworld.md)에서는 기본 이진분류를 노트북으로 구현합니다. 

#### 모델 평가

일반적으로 train set의 score가 test set보다 조금 높음

과대적합: 모델의 train set 성능이 test set보다 훨씬 높은 경우 

과소적합: train set와 test set 성능이 모두 낮거나, test set 성능이 오히려 더 높은 경우


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




