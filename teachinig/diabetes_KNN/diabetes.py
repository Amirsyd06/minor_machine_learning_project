import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataframe = pd.read_csv("diabetes.csv")

zero_not_accepted = ["Glucose","BloodPressure",
                     "SkinThickness","Insulin","BMI"]
for coloumns in zero_not_accepted:
    dataframe[coloumns] = dataframe[coloumns].replace(0, np.nan)
    mean = int(dataframe[coloumns].mean(skipna=True))
    dataframe[coloumns] = dataframe[coloumns].replace(np.nan,mean)

x = dataframe.iloc[: , :8]
y = dataframe.iloc[: , 8]
print(x)
x_test,x_train,y_test,y_train = train_test_split(x , y , train_size=0.2)
clf = KNeighborsClassifier(n_neighbors=35)
clf.fit(x_train , y_train)

y_pred = clf.predict(x_test)
acc = accuracy_score(y_test , y_pred)

print(f"the accuracy of KNN is = {acc*100}")