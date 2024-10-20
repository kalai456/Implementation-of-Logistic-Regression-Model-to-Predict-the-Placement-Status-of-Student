# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data by removing unnecessary columns and encoding categorical variables using LabelEncoder.  
2. Split the data into training and test sets using train_test_split.  
3. Train a Logistic Regression model on the training data and make predictions on the test data.  
4. Evaluate the model using accuracy score, confusion matrix, and classification report for performance analysis.  
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KALAISELVAN J
RegisterNumber: 212223080022
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
# Placement data:

![277169578-0d65163e-89bd-4559-a827-b6c984a8b69c](https://github.com/rajalakshmi8248/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860827/19b35163-7efc-4c7d-aa80-3b8a68fe16ce)

# Salary data:

![image](https://github.com/user-attachments/assets/acaab8f6-d8e7-4743-a906-4345a00415d0)



# Checking the null() function:

![277169635-27949900-92a0-468a-bb12-8dcb01a83ec0](https://github.com/rajalakshmi8248/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860827/d413eb3b-2c9a-489e-8c3c-844c3425c55d)

# Data duplicate:

![277169662-9eef8da7-cdf1-4ce6-86f0-9b6c587c6146](https://github.com/rajalakshmi8248/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860827/44d47176-c76e-4483-8622-a953864a04af)

# Print data:

![277169680-6ae4dffb-18b2-4b82-95dc-6c7f83b8c23d](https://github.com/rajalakshmi8248/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860827/01f1d051-407c-4bdb-b0f1-11195b76e2ee) 

# Data-status:

![277169700-1810a3d9-e2d4-4dc5-8c46-0215a94b6f8d](https://github.com/rajalakshmi8248/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860827/2dad67a2-3939-471b-9f7c-0e2f50bb0cd1)

# Y_prediction array:

![277169740-91beecf1-7253-42fb-a1af-22205d0c25a5](https://github.com/rajalakshmi8248/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860827/f471fbaf-a6bc-476d-b4ea-0903a23dba24)

# Accuracy value:

![277169758-86e41762-900b-463d-814c-90a1d0e84355](https://github.com/rajalakshmi8248/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860827/8831a1a5-adb3-4636-a129-16298730536e)

# Confusion array:

![277169800-5ec7c8f5-829c-4042-a9e1-a1c9b0595adb](https://github.com/rajalakshmi8248/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860827/900b225f-4b0b-4339-bff9-54d5cd69a7b6)

# Classification Report:

![277169830-2125bc90-ce93-4f7b-93b7-c6bf6786f51b](https://github.com/rajalakshmi8248/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860827/9f0672e7-b0e6-4d34-aaba-bfb8c62c787f)

# Prediction of LR:

![277169929-024f856e-41c9-44aa-874b-c778bf7f1c28](https://github.com/rajalakshmi8248/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860827/8aef5dd1-1d30-421e-bc82-bfa6ff084a14)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
