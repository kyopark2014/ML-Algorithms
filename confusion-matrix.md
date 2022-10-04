# Confusion Matrix (오차행렬)

Logistric regression, XGBoost등에서 성능을 평가할때 Confusion matrix를 이용할 수 있습니다. 

Confusion matrix는 실제 데이터와 예측 데이터에 대한 분포를 표현한 Matrix입니다. 아래와 같이 TP(True Positive), TN(True Negative)가 중요합니다. 

![image](https://user-images.githubusercontent.com/52392004/190932784-2061b5ea-149a-4c34-90b9-a8f92d158938.png)

## Accuracy

Accuracy(정확도)는 아래와 같이 정의할 수 있습니다. 불균형(Imbalance) 데이터에서는 accuracy가 항상 좋은 성능 평가지표가 되지 않을 수도 있습니다. 

![image](https://user-images.githubusercontent.com/52392004/190932881-ae8e0ad3-7c80-4d71-aa60-a086e6e1c8ef.png)


## Recall

Recall(재현율)은 특정 class가 얼마나 정확하게 예측되었는지를 보여줍니다. 

Sensitivity 또는 true positive rate (TPR)로도 생각할 수 있습니다.

![image](https://user-images.githubusercontent.com/52392004/190935655-ece4b472-ae13-4df7-8524-2e23f3109f37.png)

실제로 참인것을 거짓으로 예측하면 안되는 경우에 Recall이 중요합니다. 즉, 거짓인것을 중점적으로 찾을때 중요한 지표입니다. 예) 암진단 


## Precision

Precision(정밀도)는 positive로 예측된 관측치중 제대로 예측된 비중을 보여주는데, 도출된 결과가 얼마나 정확한지 보여줍니다. 

![image](https://user-images.githubusercontent.com/52392004/190937018-175c0987-8dfe-49a3-9589-03d5f237292a.png)

실제로 거짓인것을 참으로 예측하면 안되는 경우에 Precision이 중요합니다. 즉, 참인것을 중점적으로 찾을때 중요한 지표입니다. 예) 스팸메일 


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
