import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy

img = plt.imread("a.jpg")

width = img.shape[0]
height = img.shape[1]

img = img.reshape(width*height, 3)

kmeans = KMeans(n_clusters=32).fit(img)

labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

img2 = numpy.zeros((width, height, 3), dtype = numpy.uint8)

index = 0
for i in range(width):
	for j in range(height):
		img2[i][j] = clusters[labels[index]]
		index += 1

plt.imshow(img2)
plt.show()