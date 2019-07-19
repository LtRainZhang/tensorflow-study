# coding=utf-8
# one hiden layer
import tensorflow as tf
import numpy as np

Batch_size =8
seed = 23455
learning_rate = 0.001


rdm = np.random.RandomState(seed)
X =rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]
#Y = [[int(x0+x1 < 1)] for (x0, x1) in X]
print(X)
print(Y_)

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random.normal([2,1], stddev=1, seed=1))
#w2 = tf.Variable(tf.random.normal([3,1], stddev=1, seed=1))

y = tf.matmul(x,w1)
#y = tf.matmul(a,w2)

#loss = tf.reduce_mean(tf.square(y_-y))
#自定义loss function
COST = 9
PROFIT = 1

loss = tf.compat.v1.reduce_sum(tf.compat.v1.where(tf.greater(y,y_),COST*(y-y_),PROFIT*(y_-y)))  #选择函数
train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.compat.v1.Session() as sess:
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
#    print("y in tf3_3 is: \n", sess.run(y, feed_dict={x: [[0.7, 0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
    print("w1:\n", sess.run(w1))
    steps = 20000
    for i in range(steps):
        start = (i*Batch_size) %32
        end = start + Batch_size
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y_[start: end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y_})
            print("After %d training steps, loss on all data is %g" % (i, total_loss))
            print(sess.run(w1),"\n")
    print ("\n")
    print("Final w1: \n", sess.run(w1))
