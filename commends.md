# Notebook Commends

## Data 읽어오기

```python
import pandas as pd
pd.options.display.max_rows=20
pd.options.display.max_columns=10

data = pd.read_csv('../dataset/train.csv')

data.head()

data.describe()

data.info()
```

