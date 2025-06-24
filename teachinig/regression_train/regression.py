import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np

dataframe = pd.read_csv("student_score.csv")

data = dataframe.iloc[: , :-1]
label = dataframe.iloc[: ,1]

x_test , x_train , y_test , y_train = train_test_split(data , label , test_size=0.2)

regression = LinearRegression()
regression.fit(data,label)

# plt.scatter(data,label)
# plt.title("Hourse VS Scores")
# plt.xlabel("Hourse")
# plt.ylabel("Scores")
# plt.show()

print(regression.coef_)
print(regression.intercept_)

y_pred = regression.predict(x_test)
df = pd.DataFrame({"Actual":y_test, "Predict":y_pred})

print("MAE:" , mean_absolute_error(y_test , y_pred))
print("MSE:" , mean_squared_error(y_test , y_pred))
print("RMSE:" , np.sqrt(mean_absolute_error(y_test , y_pred)))