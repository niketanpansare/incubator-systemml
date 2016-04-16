#!/usr/bin/python

import sys
import tensorflow as tf

max_iterations = int(sys.argv[1])
setup = int(sys.argv[2])
# print(str(max_iterations) + " " + str(setup))
numFilters = -1
numChannels = -1
filterSize = -1
pad = 'NONE'
if setup == 1:
        numFilters = 20
        numChannels = 1
        filterSize = 5
        pad = 'VALID'
elif setup == 2:
        numFilters = 50
        numChannels = 20
        filterSize = 5
        pad = 'VALID'
elif setup == 3: 
        numFilters = 20
        numChannels = 1
        filterSize = 3
        pad = 'SAME'
elif setup == 4:
        numFilters = 50
        numChannels = 20
        filterSize = 3
        pad = 'SAME'
else:
        raise ValueError('Incorrect setup (needs to be [1, 4]).')

n = 60000
imgSize = 28
# Using tf.float32 throws TypeError :( 
X = tf.random_uniform([n,imgSize,imgSize,numChannels], dtype=tf.float32)
w = tf.random_uniform([filterSize,filterSize,numChannels,numFilters], dtype=tf.float32)
batch_size = 64
foo = 0

with tf.Session() as sess:
        result = tf.fill([1], 0.0)
        for iter in xrange(max_iterations):
                beg = ((iter+1) * batch_size) % n
                end = min(n-1, beg + batch_size)
                X_batch = X[beg:end,:,:,:]
                convOut_1 = tf.nn.conv2d(X_batch, w, padding=pad, strides=[1,1,1,1])
                # 'minimum' to avoid any overflow
                result = tf.add(result, tf.minimum(0.1, tf.reduce_sum(convOut_1)))
        foo = sess.run([result])[0]
print(foo)
