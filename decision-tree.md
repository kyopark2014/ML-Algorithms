# 결정트리 (Decision Tree)

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)을 참조하여 [Decision Tree](https://github.com/kyopark2014/ML-Algorithms/blob/main/src/decision_tree.ipynb)에서는 결정트리에 대해 예를 보여주고 있습니다. 

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


