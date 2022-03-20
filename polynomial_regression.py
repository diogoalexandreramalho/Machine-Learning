import numpy as np
from numpy.linalg import inv
import math


x_lst = []
t_lst = []

x_dimension = int(input("Insert number of lines of X: "))

for i in range(x_dimension):
	temp = input().split()
	temp = [float(i) for i in temp]
	x_lst += [temp]

t_dimension = int(input("Insert number of lines of T: "))

for i in range(t_dimension):
	temp = input().split()
	temp = [float(i) for i in temp]
	t_lst += [temp]


X = np.array(x_lst)
T = np.array(t_lst)


W = np.matmul(np.matmul(inv(np.matmul(X.transpose(),X)),X.transpose()),T)

#W = np.matmul(np.matmul(inv(np.matmul(X.transpose(),X) + 4 * np.identity(3)),X.transpose()),T)

print(W)

