# Notebook Commends

## Data Loading

```python
import pandas as pd
pd.options.display.max_rows=20
pd.options.display.max_columns=10

data = pd.read_csv('../dataset/train.csv')

data.head()
data.describe()
data.info()
```

파일을 github에서 바로 로드하는 방법은 아래와 같습니다. 

- 경로조정전: https://github.com/Suji04/ML_from_Scratch/blob/master/Breast_cancer_data.csv
- 경로조정후: https://raw.githubusercontent.com/Suji04/ML_from_Scratch/master/Breast_cancer_data.csv

```python
datapath = "https://raw.githubusercontent.com/Suji04/ML_from_Scratch/master/Breast_cancer_data.csv"
import pandas as pd
data = pd.read_csv(datapath)
```

