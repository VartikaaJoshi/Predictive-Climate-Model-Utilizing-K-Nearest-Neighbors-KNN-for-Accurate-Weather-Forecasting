import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.array([
    [30, 50, 10, 'Sunny'],
    [28, 55, 15, 'Sunny'],
    [10, 80, 20, 'Rainy'],
    [5, 85, 25, 'Snowy'],
    [0, 90, 30, 'Snowy'],
    [15, 65, 5, 'Rainy']    
])

#Splitting the data into features and labels
X = data[:, 0:3] # features - temp, humidity, wind speed
y= data[:, 3] # labels- sunny, rainy and snowy

#converting features to float
X = X.astype(float)

#creating training KNN model
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)

#new data for prediction
new_data = np.array([[23, 34, 4]])
prediction = knn.predict(new_data)

print("The predicted climate for the new data point is: ", prediction[0])
