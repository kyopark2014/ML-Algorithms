# 평가 (Evaluation)

## 분류(Classification)의 평가 지표

- [정확도 (accuracy)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#accuracy): 단순히 결과가 얼마나 바르게 나뉘었는지를 보여줍니다. 데이터 분포가 한쪽으로 치우치면 의미가 없습니다.  

- [정밀도 (Precision)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#precision): 데이터 분포를 고려하여 제대로 예측된 비중을 보여줍니다. 예로서는 스펨을 검출할때 스펨을 정확히 찾는것을 말하는데, 스펨을 놓칠수도 있으나 정상은 메일이 스펨으로 분류되지 않도록 하는것을 의미합니다. 올바르게 양성으로 예측된 양성 셈플의 수인 진짜 양성(true positive)와 올바르지 않게 양성으로 예측된 음성 샘플의 수인 거짓 양성(false positive)로 정의됩니다. 정밀도의 정의는 TP / (TP + FP) 입니다. 여기서 TP는 진짜 양성이고 FP는 가짜 양성입니다. 

- [재현율 (Recall)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#recall): 데이터 분포를 고려하여 특정 class가 얼마나 정확하게 예측되었는지를 보여주는데 예측하지 못한 양성 클랫의 비율을 의미합니다. 예로서는 암을 검출할때 놓치지 않는것을 의미하는데, recall이 높으면 암을 잘 찾지만, 정상이 암으로 판정될 수 있습니다. recall의 정의는 TP / (TP + FN)으로서 TP는 진자 양성이고 FN은 거짓 음성입니다. 거짓음성은 올바르지 않게 음성으로 예측된 양성 셈풀의 개수입니다. 

- [F1 Score](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#f1-score): 정밀도와 재현율의 분포가 다르기 때문에 조화 평균을 사용입니다. F1 Score가 높으면 재현율과 정밀도가 고르게 높다는 의미입니다. 

평가지표에는 [Confusion Matrix (오차행렬)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md), 마이크로 평균 / 매크로 평균이 있습니다. 


[xgboost-breast-cancer.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost/src/xgboost-breast-cancer.ipynb)와 같이 confusion matrix를 구하면 아래와 같습니다. 

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```

아때의 결과는 아래와 같습니다.
```python
array([[53,  3],
       [ 6, 81]])
```

scikit-learn의 classification_report를 이용하여 [아래처럼 acuracy, precision, recall, f1 score를 확인](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#classification-report)할 수 있습니다. 

```python
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred = y_pred))
```

이때의 결과의 예는 아래와 같습니다. 
```python
              precision    recall  f1-score   support

           0       0.90      0.95      0.92        56
           1       0.96      0.93      0.95        87

    accuracy                           0.94       143
   macro avg       0.93      0.94      0.93       143
weighted avg       0.94      0.94      0.94       143
```

세부 값은 아래와 같이 구할 수 있습니다. 

```python
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy Score: %0.2f' % (accuracy_score(y_test,y_pred)))
print('Precision Score: %0.2f' % (precision_score(y_test,y_pred)))
print('Recall Score: %0.2f' % (recall_score(y_test,y_pred)))
print('F1 Score: %0.2f' % (f1_score(y_test,y_pred)))
```

이때의 결과는 아래와 같습니다. 

```python
Accuracy Score: 0.94
Precision Score: 0.96
Recall Score: 0.93
F1 Score: 0.95
```


### 분류 모델의 비교 

데이터의 분포차이가 있는 경우에 F score를 사용하여 튜닝을 합니다. 만약 주어진 문제가 잘못 판정하면 안되는 문제여서 정밀도(precision)이 0.9 이상이여야 한다면, 이 기준을 만족하면서 F score가 높아지도록 파라미터를 튜닝합니다. 
 

### ROC 

ROC 곡선 (receiver operating characteristics curve)은 거짓 양성 비율(FPR) 대비 양성 비율(TPR) 그래프입니다. FPR은 FP / (FP+TN)이며 TPR은 재현율의 다른이름입니다. 아래의 면적이 roc_auc_score()입니다. 

### AUC 

AUC (Area under the curve)는 ROC 곡선 아래 면적을 말합니다. 예측값이 확률인 분류 문제에 사용합니다. 

AUC/ROC: The Area Under the ROC (Receiver Operating Characteristic) Curve (AUC). AUC measures the ability of the model to predict a higher score for positive examples as compared to negative examples. Because it is independent of the score cut-off, you can get a sense of the prediction accuracy of your model from the AUC metric without picking a threshold.

## 회귀(Regression)의 평가 지표

회귀는 연속적인 값을 예측하는데 평가지표로는 평균제곱근(RMSE)와 결정계수(Coefficient of determination)이 있습니다. 

## RMSE

평균제곱근(RMSE: Root Mean Squared Error)는 회귀 모델을 평가하는 주요 지표입니다. 이것은 예측값과 실제값을 두개의 배열로 만든다음에 각 요소의 차이를 제곱하여 합한 후에, 다시 배열의 요소수로 나눈값의 제곱근을 의미합니다. 아래와 같이 RMSE를 정의할 수 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/193978087-5f884bad-c6f9-428d-a241-2e6ee1bfea9f.png)



scikit-learn으로 아래처럼 구할 수 있습니다. 여기서 mean_squared_error()는 평균제공오차(MSE)이므로 sqrt()를 사용하여야 합니다. 

```python
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_true, y_pred, squared=False)
```

또는 아래처럼 numpy를 쓸수도 있습니다. 

```python
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred, squared=False)
```

## Coefficient of determination

결정계수(Coefficient of determination)는 R<sup>2</sup>로 표현되는데, RMSE(편균제곱근오차)의 분모를 '평균과 예측값의 차를 제공하여 합한값'으로 나눈 다음, 그 값을 1에서 뺀값에 해당합니다. 즉, 항상 평균을 출력하는 예측모델보다 성능이 얼마나 더 좋은가를 나타낸다고 할 수 있는데, 1에 가까울수록 예측이 잘된것를 의미합니다.

![image](https://user-images.githubusercontent.com/52392004/185774224-2209e555-c3ed-4d79-b5e7-d20bef381bc1.png)

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)

print(f'Coefficient of determination: {r2}')
```

아래는 K Neighbors Classifier을 이용한 예제입니다. 여기서 score는 결정계수를 의미합니다. score() 호출시, 분류(Classification)에서는 [정확도(Accuracy)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#accuracy)을 의미하고, 회귀(Regression)에서는 [결정계수(Coefficient of determination)](https://github.com/kyopark2014/ML-Algorithms/blob/main/evaluation.md#coefficient-of-determination)을 나타냅니다.

 
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

[Classification Report](https://www.scikit-yb.org/en/latest/api/classifier/classification_report.html)

[Metrics and scoring: quantifying the quality of predictions](https://www.scikit-yb.org/en/latest/api/classifier/classification_report.html)

