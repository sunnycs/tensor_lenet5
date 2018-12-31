import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

# numpy generates training data in the memory:
with tf.name_scope('data'):
    x_data = np.random.rand(100).astype(np.float32)
    y_data = 0.3*x_data+0.1

# define the graph elements: parameters, variables:
with tf.name_scope('parameters'):
    weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='weight')
    bias = tf.Variable(tf.zeros([1]))

# define the graph elements: output:
with tf.name_scope('prediction'):  
    y_prediction = x_data*weight + bias

# define the graph elemetns: loss:
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y_data-y_prediction))

# define the graph elements: optimizer object:
optimizer = tf.train.GradientDescentOptimizer(0.5)

#define the graph elements: optimizer unit:
with tf.name_scope('train'):
    train = optimizer.minimize(loss)


# with tf.name_scope('init'):
#     init = tf.global_variables_initializer()

init = tf.global_variables_initializer()

sess = tf.Session()

writer = tf.summary.FileWriter('logs', sess.graph)

sess.run(init)

# training:
for step in range(101):
    sess.run(train)
    if step % 10 == 0:
        print(step, 'weight:', sess.run(weight), 'bias:', sess.run(bias), 'loss:', sess.run(loss))


plt.figure()
plt.scatter(x_data, y_data, s=50, alpha=0.5)
x = np.linspace(0, 1, 10)
y = sess.run(weight)*x+sess.run(bias)
plt.plot(x,y, color='k')
plt.savefig('data_model.png')
plt.show()