import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model as lm 

# Random data
A = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]
b = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

plt.plot(A,b,'ro')

# Change row vector to column vector
A = np.array([A]).T
b = np.array([b]).T

# aply formula
lr = lm.LinearRegression()

# fit(training model)
# y = ax + b, a:coefficient, b:interceptn the model
lr.fit(A,b)

x0 = np.array([[1,46]]).T
y0 = x0*lr.coef_ + lr.intercept_
print(x0)

# plot and show the figure
plt.plot(x0,y0)
plt.show()

