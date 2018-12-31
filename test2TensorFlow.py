import tensorflow as tf
    
import numpy as np
import matplotlib.pyplot as pyplot


with tf.name_scope('data'):
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data*0.3+0.1

with tf.name_scope('parameters'):
    with tf.name_scope('weights'):
        weights = tf.Variable(tf.random_uniform([1],-1.0, 1.0,), name='weight')
        tf.summary.histogram('weights', weights)
    with tf.name_scope('bias'):
        bias = tf.Variable(tf.zeros([1]))
        tf.summary.histogram('bias', bias)

with tf.name_scope('prediction'):
    y_prediction = x_data*weights + bias
    # tf.summary.scalar('y_prediction', y_prediction)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y_prediction - y_data))
    tf.summary.scalar('loss', loss)

optimizer =tf.train.GradientDescentOptimizer(0.05)
with tf.name_scope('train'):
    train = optimizer.minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter('logs', sess.graph)
merge  = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

for step in range(201):
    sess.run(train)
    rs = sess.run(merge)
    writer.add_summary(rs, step)
    if step % 2  == 0:
        print('weights:', sess.run(weights), 'bias:', sess.run(bias), 'loss:', sess.run(loss))


sess.close()