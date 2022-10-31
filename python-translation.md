# ML 알고리즘을 Python 코드로 변환 

Jupyter 노트북으로 생성한 ML 알고리즘을 Python 코드로 변환하는 과정에 대해 설명합니다. 

1) 불필요한 코드 정리

jupyter notebook으로 작성한 코드의 일부는 데이터의 구조를 이해하고, 도표로 이해하는 코드로서 본격적인 학습에서는 사용되지 않습니다. 

[xgboost-wine-quality.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/xgboost-wine-quality.ipynb)는 [step1-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step1-xgboost-wine-quality.py)와 같이 변환됩니다.

## Jupyter Notebook 코드를 함수로 리팩터링

Jupyter 코드를 함수로 변경합니다. 함수로 변경하면 refactoring이 쉬워지고 유지 관리가 수월해집니다. 

## 관련 작업을 위한 Python 스크립트 만들기



단위 테스트 만들기

## Reference 

[ML 실험을 프로덕션 Python 코드로 변환](https://learn.microsoft.com/ko-kr/azure/machine-learning/v1/how-to-convert-ml-experiment-to-production)

[SageMaker basic - XGBoost](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/sagemaker/xgboost)

[xgboost_starter_script.py](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-basic/src/xgboost_starter_script.py)
