from sklearn import datasets,neighbors
import numpy as np 
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 

# load datasets
digits = datasets.load_digits()
# data(pedal length and width, sepal length and width)
digit_x = digits.data
# label
digit_y = digits.target

# shuffle by index
randIndex = np.arange(digit_x.shape[0])
np.random.shuffle(randIndex)

digit_x = digit_x[randIndex]
digit_y = digit_y[randIndex]

# Divide into 2 groups: training and testing
x_train, x_test, y_train, y_test = train_test_split(digit_x, digit_y, test_size = 360)

# apply knn
knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
# train data
knn.fit(x_train, y_train)
# predict data
y_predict = knn.predict(x_test)
# calculate accuracy
accuracy = accuracy_score(y_predict, y_test)
print(accuracy)	

# Test 1 image
plt.gray()
print(x_test[0])
plt.imshow(x_test[0].reshape(8,8))
dig = knn.predict(x_test[0].reshape(1,-1))
print(dig)
plt.show()