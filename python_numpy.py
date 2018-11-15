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
            
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 03 09:32:21 2017

@author: Sirius
"""
# numpy 加速
# https://pypi.python.org/pypi/Bottleneck/0.3.0
x.shape # 维数
x.size # 总的元素个数
x.itemsize # 每个元素的bytes数
x.strides # he strides of a numpy array describe the size in bytes of the steps that must be taken to increment one value along a given axis. 

# tuple 可以相加
(1,2,3) + (4,5)

# numpy 数组可以直接加list
numpy.array([1,2,3,4]) + 5 # 是在每个元素上加5

# numpy 直接初始化为nan
[np.nan] * 5

# 查看CPU内存占用率
import psutil
psutil.cpu_percent() # CPU占用率
psutil.virtual_memory().percent # 内存占用率

# two boolean np.array 'and' operator
a & b
 
# np.where 返回的值是tuple 所有要加[0]   
 
# Returns the indices of the maximum values along an axis.
import numpy as np
np.argmax(array, axis) # axis = 0 列，1 行

# 返回到目前长度为止的最大值
np.maximum.accumulate(array)

# 找到np.array 数字最大值的横坐标
np.argmax(array)

# efficient way to apply a function to numpy array
# https://stackoverflow.com/questions/7701429/efficient-evaluation-of-a-function-at-every-cell-of-a-numpy-array
# 对numpy.array的每一个元素作用一个function
def f(x):
    return x * x + 3 * x - 2 if x > 0 else x * 5 + 8

F = np.vectorize(f)  # or use a different name if you want to keep the original f
result_array = F(A)  # if A is your Numpy array

F = np.vectorize(f, otypes=[np.float]) # 指明输出的类型

#------------------------------------------------------------------------------
# 给ndarray最后一维加上window
# 要求 1 <= window <= a.shape[-1]
# https://stackoverflow.com/questions/4923617/efficient-numpy-2d-array-construction-from-1d-array
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    

import numpy as np
def rolling_window(a, window):
   """
   Make an ndarray with a rolling window of the last dimension

   Parameters
   ----------
   a : array_like
       Array to add rolling window to
   window : int
       Size of rolling window

   Returns
   -------
   Array that is a view of the original array with a added dimension
   of size w.

   Examples
   --------
   >>> x=np.arange(10).reshape((2,5))
   >>> rolling_window(x, 3)
   array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
          [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

   Calculate rolling mean of last dimension:
   >>> np.mean(rolling_window(x, 3), -1)
   array([[ 1.,  2.,  3.],
          [ 6.,  7.,  8.]])

   """
   if window < 1:
       raise ValueError, "`window` must be at least 1."
   if window > a.shape[-1]:
       raise ValueError, "`window` is too long."
   shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
   strides = a.strides + (a.strides[-1],)
   return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

a = np.arange(10)
print rolling_window(a, 3)

# 判断是不是nan
np.isnan(np.nan)
np.isnan(float('NaN'))
np.isnan(False)
np.isnan([float('NaN'),float('NaN'),1])
np.isnan([float('NaN'),float('NaN'),1])
all(np.isnan([float('NaN'),float('NaN'),1]))
any(np.isnan([float('NaN'),float('NaN'),1]))

# 在list后面加list
x = [1,2,3]
y = [4,5,6]
x.extend(y)
x = x + y

# df.datetime 是 pd.Series 格式
# 如果需要pop 先转化为list

# 判断一列series是否全部都是nan
np.nansum(yourSeries)

# 初始化一列 np.nan 有 column index 和 datetime 作为row index
d = pd.DataFrame([np.nan] * dfday.shape[1] , index = dfday.columns).T
DF = DF.append(d)
DF.set_index(pd.DatetimeIndex(nodemin))


# list index by another list
T = [L[i] for i in Idx]
# [acvolume[i] for i in acvolumeInd] = dfday.acvolume.tolist() # 这样不行，会报错
# 将 list 转化为 array， 就可以用list作为索引了，就可以赋值了

# 给dataframe的列重新赋值时，不能用list直接赋值
tmp.loc[:,'volume'] = pd.Series(range(10))
tmp.loc[:,'volume'] = np.array(range(10))

# 将dataframe的column列填充为零
df.colname1.fillna(df.colname2, inplace=True)
df.volume.fillna(0) # 给DF的某一列填充为零


# 将输出结果写出到文件中，方便查看
with open('datetime0915.txt','w') as f:
    for d in dfday.datetime:
        if d in nodemin:
           f.write('---------------------------------------' + '\n')
        f.write(d.strftime("%Y-%m-%d %H:%M:%S") + '\n')
# dataframe.ix 不支持负数的indexing
        
# 对于有重复行的，保留第一行
dfday.drop_duplicates(subset='datetime', keep="first")

# np.arange with stepsize
np.arange(60 , step = 5).tolist()

# 对dataframe进行索引时
df.ix[ind,] # 这里用的index是df本身的index, 如果用了drop)duplicate，需要对index重新进行赋值
df.iloc[ind,]

# 对于np.array中是否出现小于零的判断
any(volume < 0) # 正确
any(volume) < 0 # 错误


# python derivatives https://ojensen.wordpress.com/

# expression subscribe, 对表达式中的字符串进行替换
expr = x ** y
expr.subs(y, x ** y)
sum([f.subs(dict(k=k)) for k in u])

# pandas read_csv
# http://www.cnblogs.com/datablog/p/6127000.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

import pandas as pd
pd.__version__ # check package version


# 更新 package 的名字
# pip install pandas --upgrade
# 在安装新的package时，必须必须要关闭spyder, 否则会报错 WindowsError: [Error 5] Access is denied:

# True False Array Find True Index
[i for i, x in enumerate(t) if x]

# list sub directory inside directory
subdir = [x[0] for x in os.walk(directory)] # sub directory inside directory

# list all files inside directory
allfilename = os.listdir(directory)

# 找到某一个column为给定值的row
df.loc[df['column_name'] == some_value]

df = pd.concat(df, ignore_index=True)


# 由于append以后，dataframe的排序不是按照之前的排列的，这里可以重新定义排序顺序
cols = df.columns.tolist()
df = df[cols]

# 给dataframe增加列，需要添加index
df['datetime'] = pd.Series(dt, index = df.index) # 将datetime添加到列


# 按照dataframe的行进行遍历
for index, row in df.iterrows(): # 按照行对dataframe进行遍历
    print row
    
# np.array 类型的报错可能会报 ValueError: could not convert string to float
# a[1] = 'stringType'
# 但是转化为list就不会报错了
# dataframe 选列，不需要loc
df[['LASTPX','OPENINTS','S1','B1','volume']]

# 按照dataframe的某一列进行排序
df.sort_values(by = 'datetime', ascending = False)

# data frame的行数
len(df)

# while 条件不满足时执行
while cond:
    print 'condition satisfied'
else: # 条件不满足
    print 'condition not satisfied'
    
# dataframe 给列名重新赋值
a.rename(columns={'A':'a', 'B':'b', 'C':'c'}, inplace = True)

# 对于np.array可以用np.where, 但是对于list不能用
np.where(myarray < 0) # 对
np.where(mylist  < 0) # 错

# dict中找到给定值对应的键
print mydict.keys()[mydict.values().index(max(mydict.values()))]

# dataframe将不满足条件的列删掉
df = df[df['volume'] >= 0]

# pd.Series 多个条件and操作
result = result[(result['var']>0.25) | (result['var']<-0.25)]

# 删除dict的元素
del mydict[mykey]

# 改变dict中key的名字 mydict[newkey] = mydict.pop(oldkey)
mydict = {'a':1,'b':2,'c':3}
mydict['d'] = mydict.pop('c')

# scipy求最小值函数，可以设置等式，不等式约束
# 
from scipy.optimize import minimize # minimize(fun,initial_value)
# fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
fun = lambda x: (x-1)**2 + 1
fun = lambda tao: GaussKernelEstimatorExpr(tao,h,p)
minimize(fun,(1, p)) # 初值的维数应该与自变量个数相同


# 查看myobject是不是string类型
isinstance(myobject, basestring)

# list elmentwise multiply
[a*b for a,b in zip(lista,listb)]

# 字典获取元素
mydict.get('a') # 'a' 是key

# 集合添加元素
myset.add('a')

# np.nan是一个数 if np.nan会判断为True
# None是一个数 if None, 会判断为False


# apply array to function without known length: map
# map 的第一个参数是 
def abc(a, b, c):
    return a*10000 + b*100 + c
map(abc, (1, 2), (4, 5),(6,7)) # 将每一个对应位置的参数写进一个括号里


# dict 键值不存在
from collections import defaultdict
mydict = defaultdict(list)
print mydict['a'] # 如果key不存在，会默认初始化一个

# ellipise
import numpy as np
a = np.array([
       [[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16]],       
       [[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16]]])
a[:,:,1]
a[...,1]


# 非常神奇的zip(*var)用法
mylist = [['a','b','c'],[1,2,3]]
zip(*mylist)

# 对数据进行打乱处理
order = np.random.permutation(num_data)
data = data[order, ...]
label = label[order]



# pickle and load file from data

import pickle
my_data = {'a': [1, 2.0, 3, 4+6j],
           'b': ('string', u'Unicode string'),
           'c': None}
output = open('data.pkl', 'wb')
pickle.dump(data1, output)
output.close()

import pprint, pickle
pkl_file = open('data.pkl', 'rb')
data1 = pickle.load(pkl_file)
pprint.pprint(data1)
pkl_file.close()




