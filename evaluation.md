# 평가 (Evaluation)

## 분류(Classification)의 평가 지표

- [정확도 (accuracy)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#accuracy)
- [정밀도 (Precision)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#precision)
- [재현율 (Recall)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#recall)
- [F1 Score](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md#f1-score)

평가지표에는 [Confusion Matrix (오차행렬)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md), 마이크로 평균 / 매크로 평균이 있습니다. 


## 회귀(Regression)의 평규 지표

### Coefficient of determination

score() 호출시, 분류(Classification)에서는 정확도(정답을 맞춘 개숫의 비율)을 의미하고, 회귀(Regression)에서는 결정계수(Coefficient of determination)을 나타내는데 1에 가까울수록 예측이 잘된것)를 의미합니다.

![image](https://user-images.githubusercontent.com/52392004/185774224-2209e555-c3ed-4d79-b5e7-d20bef381bc1.png)

아래는 K Neighbors Classifier을 이용한 예제입니다. 여기서 score는 결정계수를 의미합니다. 

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))
```
