# 평가 (Evaluation)

## 분류(Classification)의 평가 지표

- [정확도 (accuracy)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#accuracy): 단순히 결과가 얼마나 바르게 나뉘었는지를 보여줍니다. 데이터 분포가 한쪽으로 치우치면 의미가 없습니다. 

- [정밀도 (Precision)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#precision): 데이터 분포를 고려하여 제대로 예측된 비중을 보여줍니다. 예) 스펨을 검출할때 스펨을 정확히 찾는것 (스펨을 놓칠수도 있으나 정상은 메일이 스펨으로 분류되지 않음)

- [재현율 (Recall)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#recall): 데이터 분포를 고려하여 특정 class가 얼마나 정확하게 예측되었는지를 보여줍니다. 예) 암을 검출할때 놓치지 않는것 (정상이 암으로 판정될 수 있음)

- [F1 Score](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#f1-score): 정밀도와 재현율의 조화 평균입니다. F1 Score가 높으면 재현율과 정밀도가 고르게 높다는 의미입니다. 

평가지표에는 [Confusion Matrix (오차행렬)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md), 마이크로 평균 / 매크로 평균이 있습니다. 

scikit-learn의 classification_report를 이용하여 [아래처럼 acuracy, precision, recall, f1 score를 확인](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#classification-report)할 수 있습니다. 

```python
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred = predictions))
```

이때의 결과의 예는 아래와 같습니다. 
```python
              precision    recall  f1-score   support

           0       0.99      0.72      0.83       967
           1       0.09      0.79      0.16        33

    accuracy                           0.72      1000
   macro avg       0.54      0.75      0.50      1000
weighted avg       0.96      0.72      0.81      1000
```

### 분류 모델의 비교 

데이터의 분포차이가 있는 경우에 F score를 사용하여 튜닝을 합니다. 만약 주어진 문제가 잘못 판정하면 안되는 문제여서 정밀도(precision)이 0.9 이상이여야 한다면, 이 기준을 만족하면서 F score가 높아지도록 파라미터를 튜닝합니다. 
 

### ROC 

ROC 곡선 (receiver operating characteristics curve)

### AUC 

AUC (Area under the curve)는 ROC 곡선으로 부터 계산합니다.



## 회귀(Regression)의 평가 지표

회귀는 연속적인 값을 예측하는데 평가지표로는 평균제곱근(RMSE)와 결정계수(Coefficient of determination)이 있습니다. 

## RMSE

평균제곱근(RMSE: Root Mean Squared Error)는 회귀 모델을 평가하는 주요 지표입니다. 이것은 예측값과 실제값을 두개의 배열로 만든다음에 각 요소의 차이를 제곱하여 합한 후에, 다시 배열의 요소수로 나눈값의 제곱근을 의미합니다. 아래와 같이 RMSE를 정의할 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/193978087-5f884bad-c6f9-428d-a241-2e6ee1bfea9f.png)



scikit-learn으로 아래처럼 구할 수 있습니다. 여기서 mean_squared_error()는 평균제공오차(MSE)이므로 sqrt()를 사용하여야 합니다. 

```python
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_true, y_pred, squared=False)
```


## Coefficient of determination

결정계수(Coefficient of determination)는 R^2로 표현되는데, RMSE(편균제곱근오차)의 분모를 '평균과 예측값의 차를 제공하여 합한값'으로 나눈 다음, 그 값을 1에서 뺀값에 해당합니다. 즉, 항상 평균을 출력하는 예측모델보다 성능이 얼마나 더 좋은가를 나타낸다고 할 수 있습니다. 

score() 호출시, 분류(Classification)에서는 정확도(정답을 맞춘 개숫의 비율)을 의미하고, 회귀(Regression)에서는 결정계수(Coefficient of determination)을 나타내는데 1에 가까울수록 예측이 잘된것를 의미합니다.

![image](https://user-images.githubusercontent.com/52392004/185774224-2209e555-c3ed-4d79-b5e7-d20bef381bc1.png)

아래는 K Neighbors Classifier을 이용한 예제입니다. 여기서 score는 결정계수를 의미합니다. 

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))
```

## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)

[Machine Learning at Work - 한빛미디어]

[sklearn.metrics.classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

[위키백과 - 평균 제곱근 편차](https://ko.wikipedia.org/wiki/%ED%8F%89%EA%B7%A0_%EC%A0%9C%EA%B3%B1%EA%B7%BC_%ED%8E%B8%EC%B0%A8)

[sklearn.metrics.mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
