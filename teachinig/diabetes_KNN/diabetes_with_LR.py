import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

dataframe = pd.read_csv("diabetes.csv")

zero_not_accepted = ["Glucose","BloodPressure",
                     "SkinThickness","Insulin","BMI"]
for coloumns in zero_not_accepted:
    dataframe[coloumns] = dataframe[coloumns].replace(0, np.nan)
    mean = int(dataframe[coloumns].mean(skipna=True))
    dataframe[coloumns] = dataframe[coloumns].replace(np.nan,mean)

x = dataframe.iloc[: , :8]
y = dataframe.iloc[: , 8]

x_train,x_test,y_train,y_test = train_test_split(x , y , train_size=0.2,random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

clf = SGDClassifier(loss='log')
clf.fit(x_train , y_train)

y_pred = clf.predict(x_test)
acc = accuracy_score(y_test , y_pred)

print(f"the accuracy of this classifier methode is = {acc*100}")