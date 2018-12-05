import tensorflow as tf
import numpy as np
x = tf.Variable(np.random.randn(2,2), trainable=True)
f_x = 2 * x* x - 5 *x + 4

loss = f_x
opt = tf.train.GradientDescentOptimizer(0.1).minimize(f_x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        print(sess.run([x,loss]))
        sess.run(opt)