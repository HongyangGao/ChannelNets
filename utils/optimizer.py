import tensorflow as tf


def rmsprop(lr, epsilon=1e-10):
    return tf.train.RMSPropOptimizer(lr, momentum=0.9, epsilon=epsilon)


def adam(lr, epsilon=1e-10):
    return tf.train.AdamOptimizer(lr, epsilon=epsilon)


def sgd(lr, **kw):
    return tf.train.RMSPropOptimizer(lr)