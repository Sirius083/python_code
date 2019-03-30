#=================================
# numpy array boardcasting
#=================================
# can only be performed 
# when the shape of each dimension in the arrays are
# (1) equal
# (2) one has the dimension size of 1
# ***: dimensionas are considered in reverse order(starting with the trailing dimension)
# numpy pad missing dimensions with size of 1 when comparing arrays

# 1. scalar and one-dimensional array
from numpy import array 
a = array([1, 2, 3])
print(a)
b = 2
print(b)
c = a + b
print(c)

# 2. scalar and two-dimensional array
# scalar and two-dimensional
from numpy import array
A = array([[1, 2, 3], [1, 2, 3]])
print(A)
b = 2
print(b)
C = A + b
print(C)

# 3.one dimensional and two dimensional arrays
# one-dimensional and two-dimensional
from numpy import array
A = array([[1, 2, 3], [1, 2, 3]])
print(A)
b = array([1, 2, 3])
print(b)
C = A + b
print(C)


# example
'''
A.shape = (2 x 3)
b.shape = (3)

A.shape = (2 x 3)
b.shape = (1 x 3)
'''
