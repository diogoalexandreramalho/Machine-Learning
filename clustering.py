import numpy as np
from numpy.linalg import inv,det
import math

nr_clusters = int(input("Number of clusters: "))
nr_points = int(input("Number of points: "))
dimension = int(input("Dimension: "))

points = []
for x in range(nr_points):
	point = []
	input_point = input("x{} = ".format(x+1)).split(',')
	for num in input_point:
		point += [[int(num)]]
	points += [np.array(point)]


avg = []
cov = []
for c in range(nr_clusters):
	c_avg = []
	input_avg = input("µ{} = ".format(c+1)).split(',')
	for num in input_avg:
		c_avg += [[int(num)]]
	avg += [np.array(c_avg)]

	cov_lst = []
	input_cov = input("∑{} = ".format(c+1)).split(';')
	for line in input_cov:
		line_split = line.split(',')
		c_line = []
		for num in line_split:
			c_line += [int(num)]
		cov_lst += [c_line]
	cov += [np.array(cov_lst)]

priors = input("Priors: ").split(',')
priors = [float(num) for num in priors]


posteriors = []

inv_cov = []
det_cov = []
for i in range(nr_clusters):
	inv_cov += [inv(cov[i])]
	det_cov += [det(cov[i])]



def compute_likelihood(x,c):
	subtraction = np.subtract(points[x],avg[c])
	mul_1 = np.matmul(subtraction.transpose(),inv_cov[c])
	mul_2 = np.matmul(mul_1, subtraction)
	likelihood = math.exp((-0.5 * mul_2)[0,0]) / (math.sqrt(det_cov[c]) * math.pow(2 * math.pi,dimension/2))
	return likelihood



print("\n\nE-Step\n")

# perform the expectation step
for x in range(nr_points):
	print("\nFor x{}".format(x+1)) 
	joints = []
	for c in range(nr_clusters):
		prior = priors[c]
		likelihood = compute_likelihood(x,c)
		joint = prior * likelihood
		joints.append(joint)

		print("\tFor cluster C{}".format(c+1))
		print("\t\tPrior: p(C={}) = {:.7f}".format(c+1,prior))
		print("\t\tLikelihood: p(x({})|C={}) = {:.7f}".format(x+1,c+1,likelihood))
		print("\t\tJoint: p(C={},x({})) = {:.7f}".format(c+1,x+1,joint))

	print("\tCompute the normalized posteriors for each cluster:")
	posterior = []
	for c in range(nr_clusters):
		post = joints[c] / sum(joints)
		posterior += [post]
		print("\t\tC = {}: p(C={}|x({})) = {:.7f}".format(c+1,c+1,x+1,post))

	posteriors += [posterior]



print("\n\n\nM-Step -> estimate the new parameters for each cluster\n")

# perform the maximization step
for c in range(nr_clusters):

	avg[c] = np.zeros(avg[c].shape)
	sum_posts = 0
	for x in range(nr_points):
		avg[c] += posteriors[x][c] * points[x]
		sum_posts += posteriors[x][c]
	avg[c] /= sum_posts

	cov_lst = [[] for i in range(dimension)]
	for i in range(dimension):
		for j in range(dimension):
			value = 0
			for x in range(nr_points):
				value += posteriors[x][c] * (points[x][i] - avg[c][i]) * (points[x][j] - avg[c][j])
			value /= sum_posts
			cov_lst[i] += [value[0]]
	
	cov[c] = np.array(cov_lst)

	priors[c] = sum_posts / nr_points

	print("For cluster C{}".format(c+1))
	print("\tµ{} = {}".format(c+1,np.array2string(avg[c], precision=7)))
	print("\t∑{} = {}".format(c+1,np.array2string(cov[c], precision=7)))
	print("\tThe new likelihood is p(x|C={}) = N(x|µ{}=...,∑{}=...)".format(c+1,c+1,c+1))
	print("\tFor the prior: p(C={}) = {:.7f}\n".format(c+1,priors[c]))







