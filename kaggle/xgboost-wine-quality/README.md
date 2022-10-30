# XGBoost를 이용한 Wine Quality 

[Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)를 이용하여 Regression 문제에 XGBoost를 사용합니다. 이때 dataset의 형태는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/198870165-4992a598-8aa4-4682-a93f-3c2de1285449.png)

[xgboost-wine-quality-EDA.ipynb
](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/xgboost-wine-quality-EDA.ipynb)에서 data를 불러와서 Wrangling을 수행합니다. 

이때의 수정된 특성(Feature) Dataset은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/198870199-ee856ffb-d3a7-4a39-ac33-1c05c0dedd64.png)


[wine_concat.csv](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/data/wine_concat.csv)와 같이 수정된 데이터를 저장한 후에, [xgboost-wine-quality.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/xgboost-wine-quality.ipynb)에서 로드하여 Regression 작업을 수행합니다. 여기서 얻어진 RMSE는 0.686 입니다. 

여기서는 Bayesian Optimizer를 이용하여 HPO를 수행하였습니다.

```python
from bayes_opt import BayesianOptimization

def xgbc_cv(n_estimators, learning_rate, max_depth, gamma, min_child_weight, subsample, colsample_bytree, ):
    xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror',
                        n_estimators=int(n_estimators),
                        learning_rate=learning_rate,
                        max_depth=int(max_depth),
                        gamma=gamma,
                        min_child_weight=min_child_weight,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        random_state=2, verbosity=0, use_label_encoder=False, n_jobs=-1)

    xgb.fit(X_train, y_train)    

    y_pred = xgb.predict(X_test)

    reg_mse = mean_squared_error(y_test, y_pred)
    reg_rmse = np.sqrt(reg_mse)

    print('RMSE: %0.3f' % (reg_rmse))   

    return -reg_rmse

hyperparameter_space = {
    'n_estimators': (50, 800),
    'learning_rate': (0.01, 1.0),
    'max_depth': (1, 8),
    'gamma' : (0.01, 1),
    'min_child_weight': (1, 20),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.1, 1)
}

optimizer = BayesianOptimization(f=xgbc_cv, pbounds=hyperparameter_space, random_state=2, verbose=0)

#gp_params = {"alpha": 1e-10}
#optimizer.maximize(init_points=3,n_iter=10,acq='ucb', kappa= 3, **gp_params)    
#optimizer.maximize(init_points=2, n_iter=10)
optimizer.maximize(init_points=5, n_iter=10, acq='ei')

optimizer.max
```

이때의 결과는 아래와 같습니다.

```java
RMSE: 0.838
RMSE: 0.722
RMSE: 0.812
RMSE: 0.723
RMSE: 1.534
RMSE: 1.470
RMSE: 0.737
RMSE: 0.734
{'target': -0.722000420424151,
 'params': {'colsample_bytree': 0.6573438697155973,
  'gamma': 0.30665812693777794,
  'learning_rate': 0.274159002351838,
  'max_depth': 5.347936829385064,
  'min_child_weight': 11.053699791263742,
  'n_estimators': 150.93495900870016,
  'subsample': 0.7567890606328732}}
```

## Feature Importance

특성중요도(Feature Importance)는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/198870039-c3917787-8711-40dd-8dfb-5b1ed569fde0.png)

이때의 트리구조는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/198870049-3baa08af-6e8b-4cde-a61e-c9b5fbac0daa.png)




## Reference

[UCI - Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)


[White Wine Quality](https://www.kaggle.com/datasets/piyushagni5/white-wine-quality)

[Wine Quality EDA kor](https://www.kaggle.com/code/rakgyunim/wine-quality-eda-kor/notebook)
