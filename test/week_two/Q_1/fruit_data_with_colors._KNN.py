import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("fruit_data_with_colors.csv")

label = dataset["fruit_label"]
features = dataset[["mass","width","height","color_score"]]

x_train , x_test , y_train , y_test = train_test_split(features , label , test_size=0.25,random_state=20)

sc =StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

clf = KNeighborsClassifier(n_neighbors=13)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test,y_pred)
print(f"Accuracy of this methode is = {acc*100}")