# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
seed = 2


# 定义产生数据集
def generateds():
    rdm = np.random.RandomState(seed)
    X= rdm.randn(300, 2)        # 产生两百组Gaussian分布随机数
    Y_ = [int(x0*x0 + x1*x1 < 2) for (x0, x1) in X]    # 将0或1 赋值给Y_,代表红蓝出现的概率
    Y_c = [['red' if y else 'blue'] for y in Y_]       # 赋值颜色
    X = np.vstack(X).reshape(-1, 2)                     # 数据整形
    Y_ = np.vstack(Y_).reshape(-1, 1)
    return X, Y_, Y_c


# 定义获得权重值，包含正则化
def get_weight(shape, regularizer):
    w = tf.compat.v1.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


# 定义获得置b
def get_bias(shape):
    b = tf.compat.v1.Variable(tf.constant(0.01, shape=shape))
    return b


# 定义前向传播过程
def forward(x, regularizer):
    w1 = get_weight([2, 11], regularizer)
    b1 = get_bias([11])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)  # relu，非线性函数拟合

    w2 = get_weight([11, 1], regularizer)
    b2 = get_bias([1])
    y = tf.matmul(y1,w2) + b2     # 通过前向传播计算y
    return y


STEPS = 40000
Batch_size = 30
Learning_rate_base = 0.001
Learning_rate_decay = 0.999
Regularizer = 0.01     # 正则化系数


# 定义反向传播过程
def backward():
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
    X, Y_, Y_c = generateds()
    y = forward(x, Regularizer)
    global_step = tf.compat.v1.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        Learning_rate_base,
        global_step,
        300/Batch_size,
        Learning_rate_decay,
        staircase=True)

    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.compat.v1.Session() as sess:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i*Batch_size) % 300
            end = start + Batch_size
            sess.run(train_step, feed_dict={x: X[start: end], y_: Y_[start: end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
                print("After %d steps, loss is: %f" % (i,loss_v))

        xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.contour(xx,yy,probs, levels=[0.5])
    plt.show()


if __name__=='__main__':
    backward()



