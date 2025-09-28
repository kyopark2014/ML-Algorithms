# Machine Learning Algorithms

<img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">

## 용어 Index

[Index](https://github.com/kyopark2014/ML-Algorithms/blob/main/index.md)에서는 주요 용어에 대한 링크를 제공합니다. 

## AI / ML

- 인공지능(AI, Artificial Intelligence)은 사람처럼 학습하고 추론할 수 있는 지능을 가진 시스템을 만드는 기술을 의미합니다.

- 머신러닝(ML, Machine Learning)은 규칙을 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습합니다. 범용적인 목적에 적합하지만, 주어진 데이터에 대한 특성 추출(Feature Extraction)을 인간이 처리(전처리)해야 합니다.

- [딥러닝(DL, Deep Learning)](https://github.com/kyopark2014/deep-learning-algorithms/blob/main/deep-learning.md)은 인공 신경망에 기반한 머신러닝으로서 TensorFlow, PyTorch가 해당됩니다. 대량의 데이터를 신경망에 적용하면 컴퓨터가 스스로 분석하여 특성 추출(Feature Extraction)을 수행합니다. 

<!-- <img width="381" alt="image" src="https://user-images.githubusercontent.com/52392004/187052186-e3c810ed-1487-425e-8e91-93307dccfbc9.png"> -->


## 데이터전처리 

[Preprocessing](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md)에서는 [표준점수(z)를 이용한 데이터 정규화](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md#%ED%91%9C%EC%A4%80%EC%A0%90%EC%88%98-standard-score%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%A0%95%EA%B7%9C%ED%99%94) 및 [Train/Test Dataset](https://github.com/kyopark2014/ML-Algorithms/blob/main/preprocessing.md#train%EA%B3%BC-test-dataset)을 준비하는 과정을 설명합니다. 

머신러닝에서 특성(Feature)는 원하는 값을 예측하기 위해 활용하는 데이터를 의미하고, 타깃(Target)은 예측해야 할 값입니다. 

## 특성공학 

특성공학(Feature Engineering)은 주어진 특성을 조합하여 새로운 특성을 만드는 과정입니다. 

- Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work (e.g., separating time from a date/time field, combining fields — height/weight). Feature engineering can improve model accuracy and speed up training.

- Feature transfomation: Putting data in a format optimized for machine learning and generalization

- Feature selection (variable selection, attribute selection) is the process of selecting a subset of relevant features (independent variables, predictors) for use in model construction. Feature selection can improve model accuracy, simplify models, shorten model training times, and reduce overfitting.

- Correlation analysis is a method of statistical evaluation used to study the strength of a relationship between two, numerically measured, continuous variables (e.g., height and weight). This particular type of analysis is useful when a researcher wants to establish if there are possible connections between variables.

- Label encoding: Converting categorical text data into model-understandable numerical data

## Machine Learning

### Supervised Learning

회귀(Regression)은 예측하고 싶은 종속변수가 숫자일때 사용하는 머신러닝 방법입니다. [Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/regression.md)에서는 Regression에 대한 기본 설명 및 구현하는 코드를 예제로 설명합니다. 

분류(Classification)은 Sample을 몇개의 Class중에 하나로 분류할 수 있습니다. [Classification](https://github.com/kyopark2014/ML-Algorithms/blob/main/classification.md)을 통해 예제 중심으로 설명합니다. 

### Unsupervised Learning

Clustering의 대표적인 예로 k-Means가 있습니다. [k-Means](https://github.com/kyopark2014/ML-Algorithms/blob/main/k-means.md)는 비지도학습(Unsupervised Learning)으로 정답 label이 없는 데이터에서 유사도를 기준으로 k개의 군집으로 분류할 수 있습니다. 

-  K-means is an unsupervised learning algorithm. It attempts to find discrete groupings within data, where members of a group are as similar as possible to one another and as different as possible from members of other groups. You define the attributes you want the algorithm to use to determine similarity.

Dimensionally Reduction의 예로서는 PCA (Principal Component Analysis)가 있습니다. [PCA](https://github.com/kyopark2014/ML-Algorithms/blob/main/pca.md)를 이용해 데이터의 분산(variance)을 최대한 보존하면서 축소된 데이터를 학습데이터로 사용할수 있습니다. 


## Deep Learning

[Deep Learning Algorithms](https://github.com/kyopark2014/deep-learning-algorithms)에서는 Deep Learning에 대한 설명 및 예제에 대해 다루고 있습니다. 

## 모델 과적합 방지

[regularization](https://github.com/kyopark2014/ML-Algorithms/blob/main/regularization.md)에서는 모델 과적합을 방지하는 방법에 대해 설명합니다. 


## 모델 평가

[평가 (Evaluation)](https://github.com/kyopark2014/ML-Algorithms/blob/main/evaluation.md)은 알고리즘에  모델 평가 지표에 대해 설명합니다. 



## Hyperparameter Optimization (HPO)

[Hyperparameter Optimization](https://github.com/kyopark2014/ML-Algorithms/blob/main/hyperparameter-optimization.md)에서는 머신러닝 학습 알고리즘별 최적의 Hyperparameter 조합을 찾아가는 과정을 의미 합니다. 



## ML application

[ML로 수행할수 있는 응용 영역](https://github.com/kyopark2014/ML-Algorithms/blob/main/applications.md)에 대해 설명합니다. 



## Machine Learning Examples

[XGBoost Algorithms](https://github.com/kyopark2014/ML-xgboost)에서는 XGBoost를 사용한 다양한 사례에 대해 설명합니다. 

## 각종 유용한 라이브러리

- [Numpy](https://github.com/kyopark2014/ML-Algorithms/blob/main/numpy.md)로 데이터를 준비합니다. 

## AWS 제공 알고리즘

[Built-in Algorithms](https://github.com/kyopark2014/ML-Algorithms/blob/main/built-in.md)에서는 AWS에서 제공하는 ML 알고리즘에 대해 설명합니다. 

## ML 알고리즘을 Python 코드로 변환

노트북으로 작성한 ML 알고리즘을 [Python 코드로 변환](https://github.com/kyopark2014/ML-Algorithms/blob/main/python-translation.md)합니다. 


## LLM

[Transformer 모델의 이론적 이해](./transformer.md)

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[머신러닝·딥러닝 문제해결 전략 - 신백균, 골든래빗](https://github.com/BaekKyunShin/musthave_mldl_problem_solving_strategy)

[Machine Learning at Work - 한빛미디어]

[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)


