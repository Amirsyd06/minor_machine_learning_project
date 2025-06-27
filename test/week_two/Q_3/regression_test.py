import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np

dataset = pd.read_csv("data.csv")

zero_not_accepted = ["population"]

for coloumn in zero_not_accepted:
    dataset[coloumn] = dataset[coloumn].replace(0, np.nan)
    mean = int(dataset[coloumn].mean(skipna=True))
    dataset[coloumn] = dataset[coloumn].replace(np.nan,mean)

X = dataset["population"]  
y = dataset[["benefit"]]       


x_train , x_test , y_train , y_test = train_test_split( y, 
                                                       X , test_size=0.2 , random_state=20)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

print(regressor.coef_)
print(regressor.intercept_)
y_pred = regressor.predict(x_test)

df = pd.DataFrame({"Actual":y_test, "Predict":y_pred})

print("MAE:" , mean_absolute_error(y_test , y_pred))
print("MSE:" , mean_squared_error(y_test , y_pred))
print("RMSE:" , np.sqrt(mean_absolute_error(y_test , y_pred)))