import tensorflow as tf
import numpy as np


def rev_conv2d(outs, kernel, scope, keep_r=1.0, train=True):
    outs = tf.transpose(outs, perm=[0, 3, 1, 2], name=scope+'/trans1')
    pre_shape = [-1] + outs.shape.as_list()[1:]
    new_shape = [-1, tf.shape(outs)[1]] + [np.prod(outs.shape.as_list()[2:])]
    outs = tf.reshape(outs, new_shape, name=scope+'/reshape1')
    num_outs = outs.shape.as_list()[-1]
    outs = conv1d(outs, num_outs, kernel, scope+'/conv1d', 1, keep_r, train)
    outs = tf.reshape(outs, pre_shape, name=scope+'/reshape2')
    outs = tf.transpose(outs, perm=[0, 2, 3, 1], name=scope+'/trans2')
    return outs


def single_block(outs, block_num, keep_r, is_train, scope, *args):
    num_outs = outs.shape[3].value
    for i in range(block_num):
        outs = dw_block(
            outs, num_outs, 1, scope+'/conv_%s' % i, keep_r, is_train)
    return outs


def simple_group_block(outs, block_num, keep_r, is_train, scope, group, *args):
    results = []
    split_outs = tf.split(outs, group, 3, name=scope+'/split')
    for g in range(group):
        cur_outs = single_block(
            split_outs[g], block_num, keep_r, is_train, scope+'/group_%s' % g)
        results.append(cur_outs)
    results = tf.concat(results, 3, name=scope+'/concat')
    return tf.add(outs, results, name=scope+'/add')


def conv_group_block(outs, block_num, keep_r, is_train, scope, group, *args):
    num_outs = int(outs.shape[3].value/group)
    results = []
    for g in range(group):
        cur_outs = pure_conv2d(
            outs, num_outs, (1, 1, group), scope+'/group_%s_conv0' % g, keep_r,
            is_train)
        cur_outs = single_block(
            cur_outs, block_num, keep_r, is_train, scope+'/group_%s' % g)
        results.append(cur_outs)
    results = tf.concat(results, 3, name=scope+'/concat')
    return tf.add(outs, results, name=scope+'/add')


def dw_block(outs, num_outs, stride, scope, keep_r, is_train, use_rev_conv=False):
    outs = dw_conv2d(outs, (3, 3), stride, scope+'/conv1', keep_r, is_train)
    if use_rev_conv:
        outs = rev_conv2d(outs, 3, scope+'/conv2', keep_r, is_train)
    else:
        outs = conv2d(outs, num_outs, (1, 1), scope+'/conv2', 1, keep_r, is_train)
    return outs


def out_block(outs, scope, class_num, is_train):
    kernel = outs.shape.as_list()[1:-1]
    outs = pool2d(outs, kernel, scope+'/pool', is_train, 'VALID')
    outs = dense(outs, class_num, scope)
    return outs


def conv_out_block(outs, scope, class_num, is_train):
    kernel = (3, 3, 9)
    for i in range(3):
        outs = dw_conv2d(outs, (3, 3), 1, scope+'/dw_conv_%s' % i, act_fn=lrelu)
        if i == 2:
            act_fn = None
        else:
            act_fn = lrelu
        outs = pure_conv2d(
            outs, outs.shape[3].value, kernel, scope+'/pure_%s' % i,
            padding='VALID', act_fn=act_fn)
    outs = tf.squeeze(outs, axis=[1, 2], name=scope+'/squeeze')
    return outs


def pure_conv2d(outs, num_outs, kernel, scope, keep_r=1.0, train=True, padding='SAME',
        weight_decay=2e-4, act_fn=tf.nn.relu6):
    stride = int(outs.shape[3].value/num_outs)
    outs = tf.expand_dims(outs, axis=-1, name=scope+'/expand_dims')
    shape = list(kernel) + [1, 1]
    weights = tf.get_variable(
        scope+'/conv/weights', shape,
        initializer=tf.random_normal_initializer())
    outs = tf.nn.conv3d(
        outs, weights, (1, 1, 1, stride, 1), padding=padding,
        name=scope+'/conv')
    outs = tf.squeeze(outs, axis=[-1], name=scope+'/squeeze')
    if act_fn:
        outs = act_fn(outs, scope+'/relu6')
    return outs


def conv1d(outs, num_outs, kernel, scope, stride=1, keep_r=1.0, train=True, weight_decay=2e-4):
    outs = tf.layers.conv1d(
        outs, num_outs, kernel, stride, padding='same', use_bias=False,
        kernel_initializer=tf.random_normal_initializer(),
        name=scope+'/conv1d')
    if keep_r < 1.0:
        outs = tf.contrib.layers.dropout(
            outs, keep_r, is_training=train, scope=scope)
    return batch_norm(outs, scope, train)


def conv2d(outs, num_outs, kernel, scope, stride=1, keep_r=1.0, train=True, weight_decay=2e-4):
    l2_func = tf.contrib.layers.l2_regularizer(weight_decay, scope)
    outs = tf.contrib.layers.conv2d(
        outs, num_outs, kernel, scope=scope, stride=stride,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.09),
        weights_regularizer=l2_func, biases_initializer=None, activation_fn=None)
    if keep_r < 1.0:
        outs = tf.contrib.layers.dropout(
            outs, keep_r, is_training=train, scope=scope)
    return batch_norm(outs, scope, train)


def dw_conv2d(outs, kernel, stride, scope, keep_r=1.0, train=True, weight_decay=2e-4,
        act_fn=tf.nn.relu):
    l2_func = tf.contrib.layers.l2_regularizer(weight_decay, scope)
    shape = list(kernel)+[outs.shape[3].value, 1]
    weights = tf.get_variable(
        scope+'/conv/weights', shape,
        initializer=tf.truncated_normal_initializer(stddev=0.09),
        regularizer=l2_func)
    outs = tf.nn.depthwise_conv2d(
        outs, weights, [1, stride, stride, 1], 'SAME', name=scope+'/depthwise_conv2d')
    if keep_r < 1.0:
        outs = tf.contrib.layers.dropout(
            outs, keep_r, is_training=train, scope=scope)
    return batch_norm(outs, scope, train, act_fn)


def pool2d(outs, kernel, scope, train, padding='SAME'):
    outs = tf.contrib.layers.avg_pool2d(
        outs, kernel, scope=scope, padding=padding)
    return batch_norm(outs, scope, train)


def dense(outs, dim, scope, weight_decay=2e-4):
    l2_func = tf.contrib.layers.l2_regularizer(weight_decay, scope)
    outs = tf.squeeze(outs, axis=[1, 2], name=scope+'/squeeze')
    outs = tf.contrib.layers.fully_connected(
        outs, dim, activation_fn=None, scope=scope+'/dense',
        weights_regularizer=l2_func)
    return outs


def layer_norm(outs, scope):
    outs = tf.contrib.layers.layer_norm(
        outs, scope=scope+'/layer_norm')
    return lrelu(outs, scope+'/lrelu')


def batch_norm(outs, scope, is_training=True, act_fn=tf.nn.relu6):
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9997, scale=True, activation_fn=act_fn,
        epsilon=1e-3, is_training=is_training, scope=scope+'/batch_norm')


def lrelu(x, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x, 0.2*x)
