from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

dataset = pd.read_csv("petrol_consumption.csv")
 
features = dataset[["Petrol_tax","Average_income",
                    "Paved_Highways","Population_Driver_licence(%)"]]
label = dataset["Petrol_Consumption"]

x_train,x_test,y_train,y_test = train_test_split(features , label , test_size=0.2,random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

regressor = LinearRegression()
regressor.fit(x_train , y_train)

sgd_regressor = SGDRegressor(max_iter=2000,random_state=20,eta0=0.01)
sgd_regressor.fit(x_train,y_train)

coef_df = pd.DataFrame(regressor.coef_ , features.columns , columns=["coefficient"])
sgd_coef_df = pd.DataFrame(sgd_regressor.coef_ , features.columns , columns=["coefficient"])

print(x_train.shape)
print(sgd_regressor.n_iter_)
print(sgd_regressor.t_)
print(coef_df)
print(sgd_coef_df)
