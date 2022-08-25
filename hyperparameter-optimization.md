# Hyperparameter Optimization (HPO)

Hyperparameter optimization은 머신러닝 학습 알고리즘별 최적의 Hyperparameter 조합을 찾아가는 과정을 의미 합니다.

x축은 중요한 파라메터이고 y축은 중요하지 않은 파라메터라면, Grid search와 Random Search를 이용하여 최적의 Hyperparameter를 구할 수 있습니다. Grid Search는 9번 시도 했지만, 3개의 시도를 한 효과를 가지므로, Rnadom search가 일반적으로 Grid search보다 더 좋은 결과를 얻습니다. 그리고 Grid search는 Random search보다 느립니다. 

이것은 GridSearchCV 클래스와 RandomizedSearchCV 클래스를 이용해 구할 수 있습니다.

![image](https://user-images.githubusercontent.com/52392004/186670429-43eae8fc-7bc5-4a46-8ae8-91f827474604.png)


## 일반적인 가이드라인

- 알고리즘 별 hyperparameter를 이해
- 경험적으로 중요한 hyperparameter를 먼저 탐색하고 값을 고정
- 덜 중요한 hyperparameter 를 나중에 탐색
- 먼저 넓은 범위에 대해 hyperparameter를 탐색하고 좋은 결과가 나온 범위에서 다시 탐색
- Random Search가 Grid Search에 더 적은 trial로 더 높은 최적화를 기대할 수 있음
- HPO에 test dataset을 사용하지 않음




