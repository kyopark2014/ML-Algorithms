# ML 알고리즘을 Python 코드로 변환 

Jupyter 노트북으로 생성한 ML 알고리즘을 Python 코드로 변환하는 과정에 대해 설명합니다. 


1) 확장자가 ipyb인 jupyter notebook 파일을 아래 명령어를 이용하여 python 파일로 변환 합니다. 


https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step0-xgboost-wine-quality.py 

jupyter nbconvert jupyter nbconvert xgboost-wine-quality.ipynb --to script --output step0-xgboost-wine-quality

[xgboost-wine-quality.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/xgboost-wine-quality.ipynb)을 상기 명령어로 변환하면, [step0-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step0-xgboost-wine-quality.py)와 같이 python 파일로 변환할 수 있습니다. 

2) 불필요한 코드 정리

jupyter notebook에서 데이터의 구조를 이해하고, 도표를 작성할때 사용했던 코드들은 본격적인 학습에서는 사용되지 않습니다. 따라서, 

[step0-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step0-xgboost-wine-quality.py)을 [step1-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step1-xgboost-wine-quality.py)와 같이 수정합니다. 


3) Python 함수로 리팩터링

함수로 변경하면 refactoring이 쉬워지고 유지 관리가 수월하여 지므로 [step1-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step1-xgboost-wine-quality.py)을 [step2-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step2-xgboost-wine-quality.py)와 같이 함수로 변환합니다.
이때, main은 진입점(entry point)이므로 실행중인지 여부를 확인하여 아래처럼 사용합니다. 

```python
if __name__ == '__main__':
    main()
```



## 관련 작업을 위한 Python 스크립트 만들기



단위 테스트 만들기

## Reference 

[ML 실험을 프로덕션 Python 코드로 변환](https://learn.microsoft.com/ko-kr/azure/machine-learning/v1/how-to-convert-ml-experiment-to-production)

[SageMaker basic - XGBoost](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/sagemaker/xgboost)

[xgboost_starter_script.py](https://github.com/kyopark2014/aws-sagemaker/blob/main/training-basic/src/xgboost_starter_script.py)
