from sklearn import datasets,neighbors
import numpy as np 
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load datasets
iris = datasets.load_iris()
# data(pedal length and width, sepal length and width)
iris_x = iris.data
# label
iris_y = iris.target

# shuffle by index
randIndex = np.arange(iris_x.shape[0])
np.random.shuffle(randIndex)

iris_x = iris_x[randIndex]
iris_y = iris_y[randIndex]

x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size = 50)

knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)

accuracy = accuracy_score(y_predict, y_test)

print(y_predict)
print(y_test)
print(accuracy)