# Confusion Matrix (오차행렬)

Logistric regression, XGBoost등에서 성능을 평가할때 Confusion matrix를 이용할 수 있습니다. 

Confusion matrix는 실제 데이터와 예측 데이터에 대한 분포를 표현한 Matrix입니다. 아래와 같이 TP(True Positive), TN(True Negative)가 중요합니다. 

![image](https://user-images.githubusercontent.com/52392004/190932784-2061b5ea-149a-4c34-90b9-a8f92d158938.png)

## Accuracy

Accuracy(정확도)는 아래와 같이 정의할 수 있습니다. 불균형(Imbalance) 데이터에서는 accuracy가 항상 좋은 성능 평가지표가 되지 않을 수도 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/190932881-ae8e0ad3-7c80-4d71-aa60-a086e6e1c8ef.png)

스팸 메일과 정상 메일을 분류할 경우에, 수신된 메일 100개중 스펨이 60개이고 정상이 40라면, 모든 메일을 스펨으로 분류한다면 정확도(accuracy)는 0.6이 됩니다. 분류 대상 class의 분포가 한쪽으로 치우친 경우에 정확도로 평가할 수 없습니다. 

## Recall

데이터 분포를 고려하여, Recall(재현율)은 특정 class가 얼마나 정확하게 예측되었는지를 보여줍니다. 

Sensitivity 또는 true positive rate (TPR)로도 생각할 수 있습니다.

![image](https://user-images.githubusercontent.com/52392004/190935655-ece4b472-ae13-4df7-8524-2e23f3109f37.png)

- 실제로 참인것을 거짓으로 예측하면 안되는 경우에 Recall이 중요합니다. 즉, 거짓인것을 중점적으로 찾을때 중요한 지표입니다. 예) 암진단 

- 스펨을 예로 들면, **실제 스팸중에 진짜 스펨의 비율**에 해당합니다. 전체 메일 중에 실제 스펨이 60개이고, 스펨으로 판정한 메일 개수가 55개라면, 재현율은 55/60 = 0.92 입니다. 

- 잘못 걸러지는 비율이 높더라도 놓치는 것이 없도록 하는 경우에 재현율이 중요합니다. 재현율 중시는 나중에 전체 데이터를 놓고 볼때 놓친 개수가 얼마나 되느냐를 중요하게 생각합니다. 발생 빈도가 낮은 질명을 분류한다면, 거짓 양성이 나오더라도 재검사를 통해 재확인 하면 된다고 생각할 수 있습니다. 

- 스팸을 분류할때 재현율(Recall)을 중시하면, 정상 메일이 스펨으로 분류될수 있어서, 바람직하지 않은 결과일 수 있습니다. 



## Precision

Precision(정밀도)는 데이터 분포를 고려하여 positive 예측된 관측치중 제대로 예측된 비중을 보여주는데, 도출된 결과가 얼마나 정확한지 보여줍니다. 

![image](https://user-images.githubusercontent.com/52392004/190937018-175c0987-8dfe-49a3-9589-03d5f237292a.png)

- 실제로 거짓인것을 참으로 예측하면 안되는 경우에 Precision이 중요합니다. 즉, 참인것을 중점적으로 찾을때 중요한 지표입니다. 예) 스팸메일 

- 스펨을 예로 들면, **스팸으로 판정한 메일 중에서 진짜 스펨의 비율**에 해당합니다. 전체 메일 중에 스펨으로 판정한 메일이 80개이고 진짜 스펨이 55개였다면, 정밀도는 55/80 = 0.68 입니다. 

- 미처 잡아내지 못하는 갯수가 많더라도 더 정확한 예측이 필요하다면 정밀도를 중시합니다. 스팸 분류로 예를 들면 가끔 스팸이 보아도 좋으니 꼭 수신해야 하는 메일이 스팸으로 걸러지지 않도록 하는것이 해당됩니다. 




## F1 Score

F1 Score는 Precision과 Recall의 harmonic mean(조화 평균)입니다. F1 score는 데이터 label이 불균형(Imbalance) 구조일 때, 모델의 성능을 정확하게 평가할 수 있으며, 성능을 하나의 숫자로 표현할 수 있습니다. 


![image](https://user-images.githubusercontent.com/52392004/190932892-85c8214f-d2ca-434e-94d4-155085f4785e.png)



## Classification report

[Fraud Detection](https://github.com/kyopark2014/ML-xgboost/tree/main/jupyter-local)에서는 XGBoost을 이용해 Predict를 수행합니다. 이때의 결과는 아래와 같이 확인할 수 있습니다. 

```python
from sklearn.metrics import classification_report

print(classification_report(y_true=y_test, y_pred = predictions))
```

이때의 결과는 아래와 같습니다. 

```python
              precision    recall  f1-score   support

           0       0.99      0.72      0.83       967
           1       0.09      0.79      0.16        33

    accuracy                           0.72      1000
   macro avg       0.54      0.75      0.50      1000
weighted avg       0.96      0.72      0.81      1000
```


## Reference

[Confusion Matrix - wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)

[[ML] 모델의 성능, 어떻게 평가해야 하나? - Confusion matrix](https://sooyounhan.blogspot.com/2020/09/ml-confusion-matrix.html)

[분류성능평가지표 - Precision(정밀도), Recall(재현율) and Accuracy(정확도)](https://sumniya.tistory.com/26)

[Machine Learning at Work - 한빛미디어]
