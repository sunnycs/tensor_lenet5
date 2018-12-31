import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

minist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weights_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, stride=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

xs =tf.placeholder(tf.float32, [None, 28*28])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob =tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28*28 , 1])

## conv1 layer
W_conv1 = weights_variable([5,5,1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
## pool1 layer
h_pool1 = max_pool_2x2(h_conv1) # outpt size 14x14x32

## conv2 layer
W_conv2 = weights_variable([5, 5, 32, 64]) # patch 5x5, in size 32, output size 54
b_conv2 = bias_variable([64])
h_conv2 = tf.relu(conv2d(h_pool1, W_conv2) + b_conv2)# output size 14x14x64
## pool2 layer
h_pool2 = max_pool_2x2(h_conv2) # output size 7x7x64
##
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) #[n_samples, 7,7,64]=>n_samplesx3136
## fc1 layer
W_fc1 = weights_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
## drop out
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer
W_fc2 = weights_variable([1024, 101])
b_fc2 = bias_variable([101])
h_fc2 =  tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# h_drop2 = tf.nn.drop(h_fc2, keep_drop)

##
prediction = tf.nn.softmax(tf.matmul(h_fc2, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(prediction*tf.log(ys), reduce_indices=[1]))
# loss = tf.reduce_mean(-tf.reduce_sum(y_prediction*tf.log(y_data), reduce_indices=[1]))


# optimizer = tf.train.GradientDescentOptimizer(0.5)
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(cross_entropy)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    result = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return result



init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

batchSize = 100

for step in range(1001):
    batch_xs, batch_ys = minist.train.next_batch(batchSize)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})
    if step % 2 == 0:
        print(compute_accuracy(minist.test.images, minist.test.labels))