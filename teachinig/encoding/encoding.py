from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder

data = asarray[["red"],["green"],["blue"]]

print(data)

encoder = OrdinalEncoder()

result = encoder.fit_transform(data)

print(result)