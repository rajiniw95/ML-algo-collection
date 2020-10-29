import math
import numpy as np

#task 01

x = [3,4] 
y = np.linalg.norm(x) # 5.0
print(y) 

# Compute the direction of a vector x.
def direction(x):
	return x/np.linalg.norm(x)

u_1 = np.array([3,4])
w_1 = direction(u_1) 
print(w_1) # [0.6 , 0.8]

u_2 = np.array([30,40])
w_2 = direction(u_2) 
print(w_2) # [0.6 , 0.8]

#same answer for both u_1 and u_2

#norm of a direction vector = unit vector

#task 02

def geometric_dot_product(x,y, theta):
	x_norm = np.linalg.norm(x)
	y_norm = np.linalg.norm(y)
	return x_norm * y_norm * math.cos(math.radians(theta))

theta = 45
x = [3,5] 
y = [8,2] 
print(geometric_dot_product(x,y,theta)) # 34.0

#task 03

def dot_product(x,y): 
	result = 0
	for i in range(len(x)):
		result = result + x[i]*y[i]
	return result

x_1=[3,5] 
y_1=[8,2]
print(dot_product(x_1,y_1)) # 34.0

#task 04

#FUNCTIONAL MARGIN

# Compute the functional margin of an example (x,y) 
# with respect to a hyperplane defined by w and b. 
def example_functional_margin(w, b, x, y):
	result = y * (np.dot(w, x) + b)
	return result
	# Compute the functional margin of a hyperplane 
	# for examples X with labels y.

def functional_margin(w, b, X, y):
	return np.min([example_functional_margin(w, b, x, y[i])
		for i, x in enumerate(X)])

x = np.array([1, 1]) 
y=1

b_1 = 5
w_1 = np.array([2, 1])

w_2 = w_1 * 10
b_2 = b_1 * 10

print(example_functional_margin(w_1, b_1, x, y)) # 8 
print(example_functional_margin(w_2, b_2, x, y)) # 80

#GEOMETRIC MARGIN

# Compute the geometric margin of an example (x,y) 
# with respect to a hyperplane defined by w and b. 
def example_geometric_margin(w, b, x, y):
	norm = np.linalg.norm(w)
	result = y * (np.dot(w/norm, x) + b/norm) 
	return result

# Compute the geometric margin of a hyperplane 
# for examples X with labels y.
def geometric_margin(w, b, X, y):
	return np.min([example_geometric_margin(w, b, x, y[i]) 
		for i, x in enumerate(X)])

x = np.array([1,1]) 
y=1

b_1 = 5
w_1 = np.array([2,1]) 

w_2 = w_1*10
b_2 = b_1*10

print(example_geometric_margin(w_1, b_1, x, y)) # 3.577708764 
print(example_geometric_margin(w_2, b_2, x, y)) # 3.577708764

# Compare two hyperplanes using the geometrical margin.
positive_x = [[2,7],[8,3],[7,5],[4,4],[4,6],[1,3],[2,5]] 
negative_x = [[8,7],[4,10],[9,7],[7,10],[9,6],[4,8],[10,10]]

X = np.vstack((positive_x, negative_x))
y = np.hstack((np.ones(len(positive_x)), -1*np.ones(len(negative_x))))

w = np.array([-0.4, -1])
b=8

# change the value of b
print(geometric_margin(w, b, X, y)) # 0.185695338177 
print(geometric_margin(w, 8.5, X, y)) # 0.64993368362
