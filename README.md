# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: NATHIN R
RegisterNumber: 212222230090

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:
![SVM For Spam Mail Detection](sam.png)

![243150574-6f7e92e1-fa6d-49db-97d0-d49137816e7a](https://github.com/NathinR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679646/f0ec48f8-3f80-4a92-988b-4a44353b64ca)

![243150589-e9e3bea5-7dbd-4180-9958-f70be63dd6b6](https://github.com/NathinR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679646/3a77f6d7-18ee-4f56-bfda-59afde817ca1)

![243150599-c0a3825c-5a8a-4135-a3b6-d8ad263e7faa](https://github.com/NathinR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679646/96263047-571c-47dc-9e74-48836d24befa)

![243150645-8d892621-0ed2-4697-b3de-ecb5b9a94ce4](https://github.com/NathinR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679646/1a3007f2-c9ca-46b6-8240-0e7f6f8bcb65)

![243151284-2855afe7-bdb1-4e12-a2d0-e0a957e3eb3a](https://github.com/NathinR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679646/4dde1f87-b6f2-4499-af76-42fa6a617dc3)

![243151294-5695d1a4-84a5-4f4a-94fd-c20dc3d80805](https://github.com/NathinR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679646/df20d2e4-c15a-4bb2-a84a-4cea1a7c45f8)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
