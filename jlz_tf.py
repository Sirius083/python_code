# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:19:35 2018

@author: Sirius
"""

import tensorflow as tf
a = tf.constant(2,name = 'a')
b = tf.constant(3,name = 'b')
x = tf.add(a,b,name = 'add')

writer = tf.summary.FileWriter('./graphs',tf.get_default_graph()) # before running session

with tf.Session() as sess:
     # writer = tf.summary.FileWriter('./graphs', sess.graph)
     print(sess.run(x))
writer.close()

tf.zeros(shape = [2,3], dtype = tf.float32)
tf.zeros_like(input_tensor, dtype)
tf.ones()
tf.ones_like()
tf.fill(dims = [2,3], value = 8, name = None) # fill a tensor with a specific value
tf.lin_space(start, stop, num, name) # tf.lin_space(10.0,13.0,4)
tf.range(start, limit = None, delta = 1, dtype = None, namge = 'range') # not iterable
tf.range(3,18,3) # ==> [3,4,9,12,15]

tf.set_random_seed(seed)
# loading graphs expensive when constants are big
# only use constants for primitive types
# use variables or readers for more data that requires more memory

s = tf.Variable(2, name = 'scalar')
m = tf.Variable([[0,1],[2,3]], name = "matrix")
W = tf.Variable(tf.zeros([784,10]))

x = tf.Variable()
x.initializer() # init op
x.value()    # read op
x.assign()   # write op
x.assign_add() 

# initialize your variables
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
    sess.run(tf.variables_initializer([a,b]))
    
    sess.run(W.initializer) # initialize one value
    print(W.eval()) # sess.run(W)

    W = tf.Variable(10)
    assign_op = W.assign(100) # creates an op(1)
    with tf.Session() as sess:
    	sess.run(W.initializer)
    	print(W.eval()) # 10
    	sess.run(assign_op) # need to be executed in a session to take effect

# 注意
my_var = tf.Variable(2, name = "my_var")
my_var_times_two = my_var.assign(2*my_var)

with tf.Session() as sess:
	sess.run(my_var.initializer)
	sess.run(my_var_times_two) # 4
	sess.run(my_var_times_two) # 8
	sess.run(my_var_times_two) # 16
	sess.run(my_var.assign_add(10)) # 20
	sess.run(my_var.assign_sun(2))  # 18

# each session maintains its own copy of variables

# control dependencies: define which ops should be run first
tf.Graph.control_dependencies(control_inputs)

# if graph g have 5 ops: a,b,c,d,e
g = tf.get_default_graph()
with g.control_dependencies([a,b,c]):
	# 'd' and 'e' will only run after 'a','b','c'
    d = ...
    e = ...

# 
a = tf.placeholder(tf.float32, shape = [3])
b = tf.constant([5,5,5], tf.float32)
c = a + b
with tf.Session() as sess:
	print(sess.run(c, feed_dict = {a:[1,2,3]}))
	# tf.Graph.is_feedable(tensor): true if and only if tensor is feedable

# extremely helpful for testing: feed in dummy values to test parts of a large graph
# the trap of lazy loading?
# lazy loading: defer creating/initializing an object until it is needed
# Attention: separate definition of ops from computing/running ops
# Note: use python property to ensure function is also loaded once the first time it is called

# Normal loading: 
x = tf.Variable(10, name = 'x')
y = tf.Variable(20, name = 'y')
z = tf.add(x,y)

writer = tf.summary.FileWriter('./graphs/normal_loading', tf.get_default_graph())
with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     for _ in range(10):
         sess.run(z)
     writer.close()


# lazy loading:
x = tf.Variable(10, name = 'x')
y = tf.Variable(20, name = 'y')

writer = tf.summary.FileWriter('./graphs/normal_loading', tf.get_default_graph())
with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     for _ in range(10):
         sess.run(tf.add(x,y)) # this op will be constructed many times
writer.close()

# graph definition
tf.get_default_graph().as_graph_def()

# distributed computation
with tf.device('/gpu:2'):
	a = tf.constant()
sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))

# 条件 tf.eager.execution
tf.cond(pred, fn1, fn2, name = None)
def huber_loss(labels, predictions, delta = 14.0):
	residual = tf.abs(labels - predictions)
	def f1(): return 0.5*tf.square(residual)
	def f2(): return delta*residual -0.5 * tf.square(delta)
	return tf.cond(residual < delta, f1, f2)

#============================================ Data
# Placeholder:
# pros: put the data processing outside tensorflow, making it easy to do in python
# cons: users often end up processing their data in a single thread and ctreating data 
#       bottleneck that slows execution down

# tf.data.Dataset
tf.data.Dataset.from_tensor_slices((features, labels))
dataset = tf.data.Dataset.from_tensor_slices((data[:,0],data[:,1]))

tf.data.Dataset.from_generator(gen, output_types, output_shapes)

# tf.data.Iterator
iterator = dataset.make_one_shot_iterator() # no need to initialization
X,Y = iterator.get_next()

iterator = dataset.make_initializable_iterator() # need to initialize with each epoch
for i in range(100):
	sess.run(iterator.initializer)
	total_loss = 0
	try:
		while True:
			sess.run([optimizer])

		except tf.errors.OutOfRangeError:
			pass
dataset = dataset.shuffle(1000)
dataset = dataset.repeat(100)
dataset = dataset.batch(128)
dataset = dataset.map(lambda x: tf.one_hot(x,10))

# session looks all trainable variables that loss depends on and update them
optimizer = tf.train.GradientDecentOptimizer(learning_rate = 0.001).minimize(loss)

_,l = sess.run([optimizer, loss], feed_dict = {X:x, Y:y})


# define same iterator for train and test set
iterator = tf.data.Iterator.from_structure(train_data.output_types,train_data.output_shapes)
img, label = iterator.get_next()
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data) 

# initialize iterator with dataset you want
with tf.Session() as sess:
	...
	for i in range(n_epochs):
		sess.run(train_init)
		try:
			while True:
				_,l = sess.run([optimizer, loss])
		except tf.errors.OutOfRangeError:
			pass

	# test the model
	sess.run(test_init)
	try:
		while True:
			sess.run(accuracy)
		except: tf.errors.OutOfRangeError:
			pass

# loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits)
loss = tf.reduce_mean(entropy)


# eager execution
import tensorflow
import tensorflow.contrib.eager as tfe # program start up
tfe.enable_eager_execution()

import tensorflow as tf
# x = tf.placeholder(tf.float32, shape = [1,1])
x = [[2.0]]
m = tf.matmul(x,x)
print(m)

# read in files from queues
filename_queue = tf.train.string_input_producer(["heart.csv"])
reader = tf.TextLineReader(skip_header_lines = 1)
key,value = reader.read(filename_queue)
with tf.Session() as sess:
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runner(coord = coord)
	for _ in range(1): # generate 1 example
	   key,value = sess.run([key,value])
	   print(value)
	   print(key)
	coord.request_stop()
	coord.join(threads)


# why tf.constant but tf.Variable? ==> tf.constant is an op ; tf.Variable is a class with many ops
# recommand
s = tf.get_variable("scalar", initializer = tf.constant(2))
m = tf.get_variable("matrix", initializer = tf.constant([[0,1],[2,3]]))
W = tf.get_variable("big_matrix", shape = (784,10), initializer = tf.zeros_initializer())




# tensorboard
# tensorboard --logdir=./graphs --port 6006  # using a linux style path
tensorboard --logdir=./normal_loading --port 6006  # using a linux style path
