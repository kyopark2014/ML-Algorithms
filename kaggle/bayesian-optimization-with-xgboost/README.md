# Bayesian Optimization with XGBoost

[Bayesian Optimization with XGBoost](https://www.kaggle.com/code/lucamassaron/tutorial-bayesian-optimization-with-xgboost)에 대해 설명합니다. 

[Bayesian optimization for Hyperparameter Tuning of XGboost classifier](https://ayguno.github.io/curious/portfolio/bayesian_optimization.html)에서는 아래와 같이 사용합니다. 

```python
seed = 112 # Random seed

def xgbc_cv(min_child_weight,colsample_bytree,gamma):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
    estimator_function = xgb.XGBClassifier(max_depth=int(5.0525),
                                           colsample_bytree= colsample_bytree,
                                           gamma=gamma,
                                           min_child_weight= int(min_child_weight),
                                           learning_rate= 0.2612,
                                           n_estimators= int(75.5942),
                                           reg_alpha = 0.9925,
                                           nthread = -1,
                                           objective='binary:logistic',
                                           seed = seed)
    # Fit the estimator
    estimator_function.fit(X_train_trans_pl1,y_train)
    
    # calculate out-of-the-box roc_score using validation set 1
    probs = estimator_function.predict_proba(X_val1_trans_pl1)
    probs = probs[:,1]
    val1_roc = roc_auc_score(y_val1,probs)
    
    # calculate out-of-the-box roc_score using validation set 2
    probs = estimator_function.predict_proba(X_val2_trans_pl1)
    probs = probs[:,1]
    val2_roc = roc_auc_score(y_val2,probs)
    
    # return the mean validation score to be maximized 
    return np.array([val1_roc,val2_roc]).mean()

from bayes_opt import BayesianOptimization

gp_params = {"alpha": 1e-10}

seed = 112 # Random seed

hyperparameter_space = {
    'min_child_weight': (1, 20),
    'colsample_bytree': (0.1, 1),
    'gamma' : (0,10)
}

xgbcBO = BayesianOptimization(f = xgbc_cv, 
                             pbounds =  hyperparameter_space,
                             random_state = seed,
                             verbose = 10)

xgbcBO.maximize(init_points=3,n_iter=10,acq='ucb', kappa= 3, **gp_params)
```


## Reference 

[Bayesian Optimization with XGBoost](https://www.kaggle.com/code/lucamassaron/tutorial-bayesian-optimization-with-xgboost)

[Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)

[Bayesian optimization for Hyperparameter Tuning of XGboost classifier](https://ayguno.github.io/curious/portfolio/bayesian_optimization.html)


[30-Days-of-ML-Kaggle](https://github.com/rojaAchary/30-Days-of-ML-Kaggle)

[Bayes Optimization 기초부터 XGB까지](https://www.kaggle.com/code/toastls93/bayes-optimization-xgb/notebook)

[2019_2nd_ml_month_with_kakr](https://github.com/noveline4530/2019_2nd_ml_month_with_kakr)
