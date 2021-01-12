import pygame
from random import randint
import math
from sklearn.cluster import KMeans 

def distance(p1, p2):
	return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])) 

pygame.init()

screen = pygame.display.set_mode((1200, 700))
pygame.display.set_caption("kmeans visualization")

running = True
clock = pygame.time.Clock()

BACKGROUND = (124, 124, 124)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BACKGROUND_PANEL = (249, 255, 230)
RED= (200, 0, 0 )
BLUE = (0, 0, 128)
GREEN = (0, 200, 0 )
PURPLE = (102, 0, 102)
ORANGE = ((255,100,10))
YELLOW = ((255,255,0))
BROWN = ((100,40,0))
FOREST_GREEN = ((0,50,0))

colors = [RED, BLUE, GREEN, PURPLE, ORANGE, YELLOW, BROWN, FOREST_GREEN ]

font = pygame.font.SysFont('sans', 40)
font_small = pygame.font.SysFont('sans', 20)
text_plus = font.render('+', True, WHITE)
text_minus = font.render('-', True, WHITE)
text_run = font.render('Run', True, WHITE)
text_random = font.render('Random', True, WHITE)
text_algorithm = font.render('Algorithm', True, WHITE)
text_reset = font.render('Reset', True, WHITE)

k = 0
error = 0
points = []
clusters = []
labels = []

while running:
	clock.tick(60)
	screen.fill(BACKGROUND)
	mouse_x, mouse_y = pygame.mouse.get_pos()
	# Draw panel
	# Draw interface
	
	pygame.draw.rect(screen, BLACK, (50, 50, 700, 500))
	pygame.draw.rect(screen, BACKGROUND_PANEL, (55, 55, 690, 490))

	# K button +
	pygame.draw.rect(screen, BLACK, (850, 50, 50, 50))
	screen.blit(text_plus, (860, 50))

	# K button -
	pygame.draw.rect(screen, BLACK, (950, 50, 50, 50))
	screen.blit(text_minus, (960, 50))

	# K value
	text_k = font.render("K = " + str(k), True, BLACK)
	screen.blit(text_k, (1050, 50))

	# run button
	pygame.draw.rect(screen, BLACK, (850, 150, 150, 50))
	screen.blit(text_run, (900, 150))

	# random button
	pygame.draw.rect(screen, BLACK, (850, 250, 150, 50))
	screen.blit(text_random, (850, 250))

	# algorithm button
	pygame.draw.rect(screen, BLACK, (850, 450, 150, 50))
	screen.blit(text_algorithm, (850, 450))

	# reset button
	pygame.draw.rect(screen, BLACK, (850, 550, 150, 50))
	screen.blit(text_reset, (850, 550))

	#  error text
	text_error = font.render("Error = " + str(int(error)), True, BLACK)
	screen.blit(text_error, (850, 350))

	# Draw mouse position when mouse is in panel
	if 50 < mouse_x < 750 and 50 < mouse_y < 550:
		text_mouse = font_small.render("(" + str(mouse_x - 50) + "," + str(mouse_y - 50) + ")", True, BLACK)
		screen.blit(text_mouse, (mouse_x + 10, mouse_y)) 

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.MOUSEBUTTONDOWN:
			#  create point on panel
			if 50 < mouse_x < 750 and 50 < mouse_y < 550:
				labels = []
				point = [mouse_x - 50, mouse_y - 50]
				points.append(point)
			# change k + 
			if 850 < mouse_x < 900 and 50 < mouse_y < 100:
				if k < 8:
					k += 1
					print("Press K+")
			# change k + 
			if 950 < mouse_x < 1000 and 50 < mouse_y < 100:
				if k > 0: 
					k -= 1
				print("Press K-")
			# run button
			if 850 < mouse_x < 1000 and 150 < mouse_y < 200:
				labels = []
				#  assign points to closest cluster
				if clusters == []:
					continue

				for p in points:
					distances_to_cluster = [] 
					for c in clusters:
						dis = distance(p,c)
						distances_to_cluster.append(dis)

					min_distance = min(distances_to_cluster)
					label = distances_to_cluster.index(min_distance)
					labels.append(label)

				# Update clusters
				for i in range(k):
					sum_x = 0
					sum_y = 0
					count = 0
					for j in range(len(points)):
						if labels[j] == i:
							sum_x += points[j][0]
							sum_y += points[j][1]
							count += 1

					if count != 0:				
						new_cluster_x = sum_x/count
						new_cluster_y = sum_y/count
						clusters[i] = [new_cluster_x, new_cluster_y]


				print("run pressed")

			# random button
			if 850 < mouse_x < 1000 and 250 < mouse_y < 300:
				labels = []
				clusters = []
				for i in range(k):
					random_point = [randint(0,700), randint(0, 500)]
					clusters.append(random_point)

				print("random pressed")

			# algorithm
			if 850 < mouse_x < 1000 and 450 < mouse_y < 500:
				try:
					kmeans = KMeans(n_clusters = k).fit(points)
					clusters = kmeans.cluster_centers_
					labels = kmeans.predict(points)
				except:
					print("Error")


				print("algorithm pressed")

			# reset button 
			if 850 < mouse_x < 1000 and 550 < mouse_y < 600:
				k = 0
				error = 0
				clusters = []
				points = []
				labels = []
				print("reset pressed")

	#  Draw point
	for i in range(len(points)):
		pygame.draw.circle(screen, BLACK, (points[i][0] + 50, points[i][1] + 50), 6)

		if labels == []:
			pygame.draw.circle(screen, WHITE, (points[i][0] + 50, points[i][1] + 50), 5)
		else:
			pygame.draw.circle(screen, colors[labels[i]], (points[i][0] + 50, points[i][1] + 50), 5)

	#  Draw clusters
	for i in range(len(clusters)):
		pygame.draw.circle(screen, colors[i], (int(clusters[i][0]) + 48, int(clusters[i][1]) + 48), 8)

	# Calculate and draw errors
	error = 0
	if clusters != [] and labels != []:
		for i in range(len(points)):
			error += distance(points[i], clusters[labels[i]])

	text_error = font.render("Error = " + str(int(error)), True, BLACK)
	screen.blit(text_error, (850, 350))


	pygame.display.flip()

pygame.quit()