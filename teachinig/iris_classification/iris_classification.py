import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import zero_one_loss

def load_data():
    dataset = pd.read_csv("iris.data",header=None,
                      names=["sepal.length","sepal.width",
                             "petal.length","petal.width","label"])

#print(dataset.head()) 

    rows , cols = dataset.shape
#print(f"number of samples are = {rows} and number of features are = {cols}")

    data = dataset.iloc[: , :4]
    label = dataset.iloc[: , 4]

    x_train , x_test , y_train , y_test = train_test_split(data , label ,test_size=0.2)
    return x_train , x_test , y_train , y_test 

x_train , x_test , y_train , y_test = load_data()

def trainig():
    clf = KNeighborsClassifier(5)
    clf.fit(x_train,y_train)
    return clf 

clf = trainig()
y_pred = clf.predict(x_test)

loss = zero_one_loss(y_test , y_pred)
accuracy = accuracy_score(y_test , y_pred)
print("accuracy ={:.2f},loss = {:.2f}".format(accuracy*100,loss*100))
