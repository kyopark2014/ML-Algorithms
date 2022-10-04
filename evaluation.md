# 평가 (Evaluation)

## 분류(Classification)의 평가 지표

- [정확도 (accuracy)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#accuracy): 단순히 결과가 얼마나 바르게 나뉘었는지를 보여줍니다. 데이터 분포가 한쪽으로 치우치면 의미가 없습니다. 
- [재현율 (Recall)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#recall): 데이터 분포를 고려하여 특정 class가 얼마나 정확하게 예측되었는지를 보여줍니다. 예) 스펨을 검출할때 놓치지 않는것
- [정밀도 (Precision)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#precision): 데이터 분포를 고려하여 제대로 예측된 비중을 보여줍니다. 예) 스펨을 검출할때 스펨을 정확히 찾는것  
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


## 회귀(Regression)의 평가 지표

## Coefficient of determination

결정계수(Coefficient of determination)는 수식에서 R^2를 의미합니다. 

score() 호출시, 분류(Classification)에서는 정확도(정답을 맞춘 개숫의 비율)을 의미하고, 회귀(Regression)에서는 결정계수(Coefficient of determination)을 나타내는데 1에 가까울수록 예측이 잘된것)를 의미합니다.

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
