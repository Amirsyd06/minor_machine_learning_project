import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_house_attributes (inputpath):
    cols = ["bedrooms","bathrooms","area","zipcode","price"]

    df = pd.read_csv(inputpath , sep = " " , header=None , names=cols)

    zipcodes = df["zipcode"].value_counts().keys().to_list()
    counts = df["zipcode"].value_counts().tolist()

    for (zipcode , count) in zip(zipcodes , counts) :
        if count < 25 :
            idxs = df[df["zipcode"] == zipcode].index 
            df.drop(idxs , inplace=True)

    return df 

df = load_house_attributes("HousesInfo.txt")

train , test = train_test_split(df , test_size=0.2 , random_state=42)

def preprocess_house_attribute (df , train ,test):
    continous = ["bedrooms","bathrooms","area"]
    sc = StandardScaler() 
    traincontinous = sc.fit_transform(train[continous])
    testcontinous = sc.transform(test[continous])

    encoder = LabelBinarizer()
    traincategorical = encoder.fit_transform(train["zipcode"])
    testcategorical = encoder.transform(test["zipcode"])

    trainX = np.hstack([traincontinous,traincategorical])
    testX = np.hstack([testcontinous , testcategorical])

    return trainX , testX

trainX , testX = preprocess_house_attribute (df , train ,test)

maxprice = train["price"].max()
train["price"] = train["price"] / maxprice
test["price"] = test["price"] / maxprice

print(train["price"])

