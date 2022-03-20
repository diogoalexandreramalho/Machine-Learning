import numpy as np
from numpy.linalg import inv
import math


lst = []

dimension = int(input("Insert dimension: "))

for i in range(dimension):
	temp = input().split()
	temp = [float(i) for i in temp]
	lst += [temp]


X = np.array(lst)

print("Matrix:")
print(X)

inverse = inv(X)

print("\nInverse")
print(inverse)
