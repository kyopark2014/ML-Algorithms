# 결정트리 (Decision Tree)

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)을 참조하여 [Decision Tree](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/decision_tree.ipynb)에서는 결정트리에 대해 예를 보여주고 있습니다. 


## 장단점

#### 장점 

- 결과를 해석하고 이해하기 쉽습니다.
- 분류/회귀 모두 사용 가능합니다.
- Data preprocessing이 거의 필요하지 않습니다.
- Outliner에 민감하지 않습니다.
- 연속형 변수, 범주형 변수 모두 적용 가능합니다.
- 대규모의 데이터 셋에서도 잘 동작합니다. 

#### 단점

- 데이터의 특성이 특정 변수에 수직/수평적으로 구분되지 못할 경우 분류률이 떨어지고 트리가 복잡해집니다.
- Overfitting (High variance) 위험이 있으므로 규제(Regularization) 필요합니다. 대표적인 방법은 "Pruning(가지치기)"입니다. 




아래는 Max Depth가 1인 결정트리의 예입니다.

![image](https://user-images.githubusercontent.com/52392004/186559659-7522f4ba-62e2-42ec-856a-a47fb9a55061.png)


## Criterion 매개변수 

Gini Impurity와 Entropy Imputiry가 있습니다. scikit-learn에서는 기본값으로 Gini impurity을 사용합니다. 


### Gini Impurity

Gini Impurity은 정답이 아닌 값이 나올 확율을 의미 합니다. 

![image](https://user-images.githubusercontent.com/52392004/186560214-844f1030-a80b-4190-a9cb-1b6151f01cde.png)

### Entropy Imputiry

Entropy Imputiry는 정보의 불확실성 또는 무질서도를 의미합니다. 

![image](https://user-images.githubusercontent.com/52392004/186560305-1651f4e1-880b-49e5-bea4-bf9d00bb6dd6.png)

### Information Gain 

![image](https://user-images.githubusercontent.com/52392004/186560390-350d25b2-2f8d-4d06-ac66-99943b6e3e35.png)


## Reference

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)


