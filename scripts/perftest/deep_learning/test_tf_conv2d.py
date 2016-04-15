import tensorflow as tf
# ---------------------------------------
# Set following before running the script
max_iterations = 1000
numFilters = 20
numChannels = 1
filterSize = 5
pad = 'VALID'
# ---------------------------------------

n = 60000
imgSize = 28
X = tf.random_uniform([n,imgSize,imgSize,numChannels], dtype=tf.float64)
w = tf.random_uniform([filterSize,filterSize,numChannels,numFilters], dtype=tf.float64)
batch_size = 64
foo = 0

with tf.Session() as sess:
        result = tf.fill([1], 0.0)
        for iter in xrange(max_iterations):
                beg = ((iter+1) * batch_size) % n
                end = min(n-1, beg + batch_size)
                X_batch = X[beg:end,:,:,:]
                convOut_1 = tf.nn.conv2d(X_batch, w, padding=pad, strides=[1,1,1,1])
                result = tf.add(result, tf.reduce_sum(convOut_1))
                foo = foo + result[0]
        r1 = sess.run([result])
        foo = r1[0]
print(foo)
