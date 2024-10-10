# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 : Start

STEP 2 : Import and Load Data: Load the student placement dataset using pandas.

STEP 3 : Preprocess Data: Copy the dataset, then drop irrelevant columns like "sl_no" and "salary" to prepare for training.

STEP 4 : Check Data Integrity: Check for missing values and duplicated rows in the cleaned dataset.

STEP 5 : Define Features and Labels: Separate the independent variables (features) and the dependent variable (target) 'status'.

STEP 6 : Split the Data: Split the dataset into training and testing sets using an 80/20 ratio.

STEP 7 : Train the Model: Initialize and train a Logistic Regression model on the training data.

STEP 8 : Evaluate the Model: Predict using the test data, calculate accuracy, generate the classification report, and test with new input.

STEP 9 : End

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SIBIRAJ P
RegisterNumber: 212222220046

import pandas as pd
data=pd.read_csv("C:/Users/SEC/Downloads/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull()

data1.duplicated().sum()

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#Library for Linear classfication
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
Accuracy

![image](https://github.com/user-attachments/assets/d430928d-0a69-4990-b78f-773f6134af2b)

Classification Report

![image](https://github.com/user-attachments/assets/c8bb2652-de6b-44ef-9a60-ff1bccc8b3df)

predicted

![image](https://github.com/user-attachments/assets/812139b3-153a-48e3-a90b-4532904caa9f)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
