from sklearn.preprocessing import StandardScaler

data = [[0,0],[0,0],[1,1],[1,1]]

scaler = StandardScaler()

scaler.fit(data)

new_data = scaler.transform(data)

print(new_data)