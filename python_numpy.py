# calculate mean over all other axises
tmp = np.mean(batch,axis =(0,1,2))


# using pickle save numpy array to disk
import pickle
with open('tiny_test.pickle', 'wb') as handle:
    # pickle.dump(images_decode, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_decode, handle)

with open('tiny_train.pickle', 'rb') as handle:
    images_decode = pickle.load(handle)
   


# 改成one-hot编码
a = np.array([1, 0, 3])
b = np.zeros((3, 4))
b[np.arange(3), a] = 1
print(b)

# 检查一个object的类型是否为 np.ndarray
# version1: isinstance(obj, np.ndarray)
# version2: type(obj) is np.ndarray


# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:13:53 2018

@author: Sirius


"""
#========================================================
# version 1
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y

import numpy as np
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)


#========================================================
# version 2
# equivalent to forming a matrix vv by stacking multiple copies of v vertivally
# perform elementwise summation of x and vv

import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"


#========================================================
# version 3
# numpy works it out automatically
# y = x + v works even though x has shape (4,3) 
# v has shape (3,)
# this line works as if v actually had shape (4,3) where each row was a copy of v 

import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"

