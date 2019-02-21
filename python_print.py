# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:07:19 2018

@author: Sirius

python string format
"""

# print line with certain length
# 10s: format a string   with 10 spaces,      default left
# 3d:  format an integer with 3 spaces,       default right
# 7.2f format a float,   with 7 spaces,       default right
print('{:10s} {:3d}  {:7.2f}'.format('xxx', 123, 98))
print('{:10s} {:3d}  {:7.2f}'.format('yyyy', 3, 1.0))
print('{:10s} {:3d}  {:7.2f}'.format('zz', 42, 123.34))

# format python3.6 new features
lang = 'Python'
adj = 'Very Good'
width = 20
num = 123.45567
f'{lang:<{width}}: my name is {adj:>{width}}: {num:<7.2f}' # format with given length

sammy_string = "Sammy loves {:20s} {}, and has {} {}."                      #4 {} placeholders
print(sammy_string.format("open-source", "software", 5, "balloons"))    #Pass 4 strings into method


# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:07:19 2018

@author: Sirius

python string format
"""

# print line with certain length
# 10s: format a string   with 10 spaces,      default left
# 3d:  format an integer with 3 spaces,       default right
# 7.2f format a float,   with 7 spaces,       default right
print('{:10s} {:3d}  {:7.2f}'.format('xxx', 123, 98))
print('{:10s} {:3d}  {:7.2f}'.format('yyyy', 3, 1.0))
print('{:10s} {:3d}  {:7.2f}'.format('zz', 42, 123.34))

# format python3.6 new features
lang = 'Python'
adj = 'Very Good'
width = 20
num = 123.45567
f'{lang:<{width}}: my name is {adj:>{width}}: {num:<7.2f}' # format with given length

sammy_string = "Sammy loves {:20s} {}, and has {} {}."                      #4 {} placeholders
print(sammy_string.format("open-source", "software", 5, "balloons"))    #Pass 4 strings into method


# format percentage  
# print ("{0:.0%}".format(1./3))  
# print ("{:.2%}".fotmat(15.84)) # print percentage with two decimal
    




