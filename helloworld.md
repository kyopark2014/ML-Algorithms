# KNN을 이용한 Binary Classification

[혼자 공부하는 머신러닝+딥러닝](https://github.com/rickiepark/hg-mldl)을 따라서 KNN을 이용해 이진분류 문제를 시험합니다. 


## Hello-Workd

scikit-learn을 이용하여 간단하게 knn으로 binary classification을 수행합니다. 

1) 소스 다운로드 받기 

```c
git clone https://github.com/rickiepark/hg-mldl
```

2) [[Amazon SageMaker] - [Notebook instances]](https://ap-northeast-2.console.aws.amazon.com/sagemaker/home?region=ap-northeast-2#/notebook-instances)로 진입하여 [Create notebook instance]에서 먼저 instance를 생성합니다. 

3) 생성한 instance를 실행후에 [Open jupyter]을 선택합니다.

<img width="1099" alt="image" src="https://user-images.githubusercontent.com/52392004/185267530-cb6a8ec5-87a5-48c1-81fd-d19b6cc3d67e.png">

4) Jupyter nodebook에서 아래와 같이 [New] - [conda_python3]을 선택합니다. 

![noname](https://user-images.githubusercontent.com/52392004/185268106-bb5264c2-c5f9-48ac-ae9b-802fb4e26299.png)

5) 아래 라이브러리는 기설치되어 있으나 설치되어 있지 않은 경우에 설치합니다. 

```c
pip install matplotlib
pip install scikit-learn
pip install numpy
```

6) 도미(bream) 데이터 준비하기 

```python
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
```

아래와 같이 데이터 확인을 합니다. 

```python
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

확인된 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/185269341-9cc3010e-4362-496e-9226-3dfa501beba8.png)

7) 빙어 데이터 준비하기

```python
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```

아래와 같이 데이터를 확인합니다. 

```python
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

확인된 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/185269567-c58cd59d-4370-4135-a340-ef86fdd4745f.png)


8) 데이터 편집 

지금 하는 학습방법은 지도학습이며 훈련을 위하여 데이터(Input)과 정답(Target)이 필요합니다. 아래와 scikit-learn에서 사용하기 위해 array로 저장하여 입력으로 사용합니다. 여기서 feature는 길이와 무게입니다. 

```python
length = bream_length+smelt_length
weight = bream_weight+smelt_weight
fish_data = [[l, w] for l, w in zip(length, weight)]
```

아래와 정답(Target)도 생성합니다. 

```python
fish_target = [1]*35 + [0]*14
print(fish_target)
```


8) scikit-learn으로 training

```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
kn.score(fish_data, fish_target)
```

test시 training과 같은 데이터를 사용했으므로 결과는 1이 나옯니다.

<img width="1097" alt="image" src="https://user-images.githubusercontent.com/52392004/185276285-df9e250f-fb23-41f0-9b2d-2f88c18d54ef.png">

9) 시험 

아래와 같이 (30, 600)을 가지는 sample은 도미입니다. 

```python
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

그래프에서 보면 아래와 같이 세모는 도미입니다. 

![image](https://user-images.githubusercontent.com/52392004/185276481-4405e6c5-e850-40c0-964c-a1f3467814db.png)

이제 knn으로 값을 예측해보면 아래와 같이 1이 나오므로 도미로 예측되었음을 알 수 있습니다. 

```python
kn.predict([[30, 600]])
```

결과는 아래와 같습니다. 

```python
array([1])
```

KNN classifier의 경우에 default로 5개의 주위의 값을 가지고 예측을 하는데, 이 값은 아래와 같이 "n_neighbors"를 이용해 변경할 수 있습니다. 

```python
kn49 = KNeighborsClassifier(n_neighbors=49)
```

## Numpy를 활용하여 Train과 Test 셈플을 만들어서 KNN 적용 

1) fish_data, fish_target을 numpy array로 변경합니다.  

```python
import numpy as np
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
```
target_arr = np.array(fish_target)

2) random index를 생성합니다. 

```python
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
print(index)
```

index가 아래와 같이 random하게 생성되었습니다.
```python
[13 45 47 44 17 27 26 25 31 19 12  4 34  8  3  6 40 41 46 15  9 16 24 33
 30  0 43 32  5 29 11 36  1 21  2 37 35 23 39 10 22 18 48 20  7 42 14 28
 38]
```

3) 앞에서 35개는 train으로 나머지는 test용으로 만듧니다. 

```python
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
```

이를 아래와 같이 그림으로 확인합니다. 

```python
import matplotlib.pyplot as plt

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/185286750-1dcb2f83-5f36-4edb-9a7d-efbdff62bbb1.png)

4) KNN으로 분석합니다. 

```python
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```

이때 결과는 마찬가지로 1.0 입니다. 

#### sciket-learn으로도 같은 동작을 수행할 수 있습니다.

```python
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, random_state=42)
```    

이때 fish_target기준으로 비율을 맞출수도 있습니다.    

```python
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)    
```    

## f
