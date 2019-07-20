#coding:utf-8
import tensorflow as tf

w1 = tf.Variable(0,dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)
Moving_Average_Decay = 0.99
ema = tf.train.ExponentialMovingAverage(Moving_Average_Decay, global_step)
ema_op = ema.apply(tf.compat.v1.trainable_variables())

with tf.compat.v1.Session() as sess:
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(tf.compat.v1.assign(w1, 1))
    sess.run(ema_op)
    print (sess.run([w1, ema.average(w1)]))

    sess.run(tf.compat.v1.assign(global_step, 100))
    sess.run(tf.compat.v1.assign(w1, 10))
    sess.run(ema_op)
    print (sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
