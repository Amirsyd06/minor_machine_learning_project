import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error

dataset = pd.read_csv("regression.csv")

label = dataset.iloc [ : , 0]
feature = dataset.iloc [: , 1]
print(label)

x_train , x_test , y_train , y_test = train_test_split(feature , label , test_size=0.2)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

print(regressor.coef_)
print(regressor.intercept_)