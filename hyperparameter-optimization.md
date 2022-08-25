# Hyperparameter Optimization (HPO)

머신러닝 학습 알고리즘별 최적의 Hyperparameter 조합을 찾아가는 과정을 의미 합니다.

## 일반적인 가이드라인

- 알고리즘 별 hyperparameter를 이해
- 경험적으로 중요한 hyperparameter를 먼저 탐색하고 값을 고정
- 덜 중요한 hyperparameter 를 나중에 탐색
- 먼저 넓은 범위에 대해 hyperparameter를 탐색하고 좋은 결과가 나온 범위에서 다시 탐색
- Random Search가 Grid Search에 더 적은 trial로 더 높은 최적화를 기대할 수 있음
- HPO에 test dataset을 사용하지 않음


GridSearchCV 클래스와 RandomizedSearchCV 클래스를 보여주고 있습니다. 


![image](https://user-images.githubusercontent.com/52392004/186670429-43eae8fc-7bc5-4a46-8ae8-91f827474604.png)
