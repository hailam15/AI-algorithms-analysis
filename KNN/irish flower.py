from sklearn import datasets
import numpy as np 
import math
import operator

# function to calculate distances btw 2 points
def cal_distance(p1, p2):
	distance = 0
	for i in range(len(p1)):
		distance += (p1[i] - p2[i])*(p1[i] - p2[i])
	return math.sqrt(distance)

# function to get k nearest neighbors
def get_k_neighbors(training_x, training_y, point, k):
	# array of distances from point to other training points
	distances = []
	# array of k nearest distances
	neighbors = []

	for i in range(len(training_x)):
		distance = cal_distance(point, training_x[i])
		distances.append((distance, training_y[i]))

	# sort distances array based on distance 
	distances.sort(key = operator.itemgetter(0))

	# adding labels sorted to array list
	for i in range(k):
		neighbors.append(distances[i][1])

	return neighbors

def highest_votes(labels):
	final_labels = [0,0,0]
	for label in labels:
		final_labels[label] += 1
	
	max_count = max(final_labels)
	return final_labels.index(max_count)

def predict(training_x, training_y, point, k):
	neighbors_labels = get_k_neighbors(training_x, training_y, point, k)
	return highest_votes(neighbors_labels)

def accuracy(predict, groundTruth):
	total = len(predict)
	count = 0
	for i in range(total):
		if predict[i] == groundTruth[i]:
			count += 1

	return count/total

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

x_train = iris_x[:100]
x_test = iris_x[100:]
y_train = iris_y[:100]
y_test = iris_y[100:]

k = 5
y_predict = []
for p in x_test:
	label = predict(x_train, y_train, p, k)
	y_predict.append(label)


print(y_predict)
print(y_test)
print(accuracy(y_predict, y_test))


