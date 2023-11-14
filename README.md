# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Udayakumar R
RegisterNumber:  212222230163
*/
```
```python
import pandas as pd
data = pd.read_csv('Employee.csv')
data.head()
data.isnull().sum()
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours",
          "time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
## Data Head :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708024/2511f197-1c54-424f-ac27-592b84c721d2)
## Null Values :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708024/7e35ffbf-7615-4674-bc63-ae26e1ca24d5)
## Assignment of X value :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708024/c815b43f-7465-4be3-afdb-e5e0f6297fe2)
## Assignment of Y value :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708024/1c13ff1b-071f-40b0-ae7b-beb22f23798e)
## Converting string literals to numerical values using label encoder :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708024/d6e369db-af68-4565-adb3-a8f5ee19fe9a)
## Accuracy :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708024/aff7b940-f7a9-4161-8690-babd7dd4a70b)
## Prediction :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708024/813140ca-4a3a-49bc-92a3-6970d9306272)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
