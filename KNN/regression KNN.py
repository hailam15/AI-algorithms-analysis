import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import neighbors

# data
x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
y = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]

# plot x,y
plt.plot(x, y, 'ro')

# reshape x,y
x = np.array(x).reshape(-1,1)
y = np.array(y)

# create x0
x0 = np.linspace(3,25,10000).reshape(-1,1)
y0 = []

# aplly knn
knn = neighbors.KNeighborsRegressor(n_neighbors = 3)
# train data
knn = knn.fit(x,y)
# predict data
y0 = knn.predict(x0)

# plot and show data
plt.plot(x0,y0)
plt.show()