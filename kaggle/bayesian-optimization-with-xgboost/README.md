# Bayesian Optimization with XGBoost

### bayesian-optimization-with-xgboost.ipynb

[bayesian-optimization-with-xgboost.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/bayesian-optimization-with-xgboost/bayesian-optimization-with-xgboost.ipynb)은 아래와 같이 사용하고 있습니다. 

```python
reg = XGBRegressor(random_state=0, booster='gbtree', objective='reg:squarederror', tree_method='gpu_hist')

scoring = make_scorer(partial(mean_squared_error, squared=False), greater_is_better=False)

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=0)
cv_strategy = list(skf.split(X_train, y_stratified))

opt = BayesSearchCV(estimator=reg,                                    
                    search_spaces=search_spaces,                      
                    scoring=scoring,                                  
                    cv=cv_strategy,                                           
                    n_iter=120,                                       # max number of trials
                    n_points=1,                                       # number of hyperparameter sets evaluated at the same time
                    n_jobs=1,                                         # number of jobs
                    iid=False,                                        # if not iid it optimizes on the cv score
                    return_train_score=False,                         
                    refit=False,                                      
                    optimizer_kwargs={'base_estimator': 'GP'},        # optmizer parameters: we use Gaussian Process (GP)
                    random_state=0)                                   # random state for replicability

# Running the optimizer
overdone_control = DeltaYStopper(delta=0.0001)                    # We stop if the gain of the optimization becomes too small
time_limit_control = DeadlineStopper(total_time=60*60*4)          # We impose a time limit (7 hours)

best_params = report_perf(opt, X_train, y_train,'XGBoost_regression', 
                          callbacks=[overdone_control, time_limit_control])
```

### 2019-2nd-ML-month-with-KaKR.ipynb

[2019-2nd-ML-month-with-KaKR.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/bayesian-optimization-with-xgboost/2019-2nd-ML-month-with-KaKR/2019-2nd-ML-month-with-KaKR.ipynb)에서는 아래와 같이 사용합니다. 

```python
def XGB_cv(max_depth,learning_rate, n_estimators, gamma
           ,min_child_weight, max_delta_step, subsample
           ,colsample_bytree, silent=True, nthread=-1):
    model = xgb.XGBClassifier(max_depth=int(max_depth),
                              learning_rate=learning_rate,
                              n_estimators=int(n_estimators),
                              silent=silent,
                              nthread=nthread,
                              gamma=gamma,
                              min_child_weight=min_child_weight,
                              max_delta_step=max_delta_step,
                              subsample=subsample,
                              colsample_bytree=colsample_bytree)
    RMSE = cross_val_score(model, X, y, scoring='accuracy', cv=5).mean()
    return -RMSE

# 주어진 범위 사이에서 적절한 값을 찾는다.
pbounds = {'max_depth': (5, 10),
          'learning_rate': (0.01, 0.3),
          'n_estimators': (50, 1000),
          'gamma': (1., 0.01),
          'min_child_weight': (2, 10),
          'max_delta_step': (0, 0.1),
          'subsample': (0.7, 0.8),
          'colsample_bytree' :(0.5, 0.99)
          }

xgboostBO = BayesianOptimization(f = XGB_cv,pbounds = pbounds, verbose = 2, random_state = 1 )

# 메소드를 이용해 최대화!
xgboostBO.maximize(init_points=2, n_iter = 10)

xgboostBO.max # 찾은 파라미터 값 확인
```

### Bayesian optimization for Hyperparameter Tuning of XGboost classifier

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

[Hyperparameter Optimization using bayesian optimization](https://medium.com/spikelab/hyperparameter-optimization-using-bayesian-optimization-f1f393dcd36d)
