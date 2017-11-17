import tensorflow as tf
import numpy as np


def rev_conv2d(outs, kernel, scope, keep_r=1.0, train=True,
               data_format='NHWC'):
    if data_format == 'NHWC':
        outs = tf.transpose(outs, perm=[0, 3, 1, 2], name=scope+'/trans1')
    pre_shape = [-1] + outs.shape.as_list()[1:]
    hw_dim = np.prod(outs.shape.as_list()[2:])
    new_shape = [-1, outs.shape.as_list()[1]] + [hw_dim]
    outs = tf.reshape(outs, new_shape, name=scope+'/reshape1')
    num_outs = outs.shape.as_list()[-1]
    outs = conv1d(outs, num_outs, kernel, scope+'/conv1d', 1, keep_r, train)
    outs = tf.reshape(outs, pre_shape, name=scope+'/reshape2')
    if data_format == 'NHWC':
        outs = tf.transpose(outs, perm=[0, 2, 3, 1], name=scope+'/trans2')
    return batch_norm(outs, scope, train, data_format=data_format)


def single_block(outs, block_num, keep_r, is_train, scope, data_format, *args):
    num_outs = outs.shape[data_format.index('C')].value
    for i in range(block_num):
        cur_outs = dw_block(
            outs, num_outs, 1, scope+'/conv_%s' % i, keep_r, is_train,
            data_format=data_format)
        outs = tf.add(outs, cur_outs, name=scope+'/add_%s' % i)
    return outs


def simple_group_block(outs, block_num, keep_r, is_train, scope, data_format,
                       group, *args):
    results = []
    split_outs = tf.split(outs, group, 3, name=scope+'/split')
    for g in range(group):
        cur_outs = single_block(
            split_outs[g], block_num, keep_r, is_train, scope+'/group_%s' % g,
            data_format)
        results.append(cur_outs)
    results = tf.concat(results, 3, name=scope+'/concat')
    return tf.add(outs, results, name=scope+'/add')


def conv_group_block(outs, block_num, keep_r, is_train, scope, data_format,
                     group, *args):
    num_outs = int(outs.shape[data_format.index('C')].value/group)
    shape = [1, 1, 4*group] if data_format == 'NHWC' else [4*group, 1, 1]
    results = []
    for g in range(group):
        cur_outs = pure_conv2d(
            outs, num_outs, shape, scope+'/group_%s_conv0' % g, keep_r,
            is_train, act_fn=None, data_format=data_format)
        cur_outs = single_block(
            cur_outs, block_num, keep_r, is_train, scope+'/group_%s' % g,
            data_format)
        results.append(cur_outs)
    results = tf.concat(results, data_format.index('C'), name=scope+'/concat')
    return tf.add(outs, results, name=scope+'/add')


def dw_block(outs, num_outs, stride, scope, keep_r, is_train,
             use_rev_conv=False, data_format='NHWC'):
    outs = dw_conv2d(
        outs, (3, 3), stride, scope+'/conv1', keep_r, is_train,
        data_format=data_format, act_fn=tf.nn.relu6)
    if use_rev_conv:
        outs = rev_conv2d(
            outs, 64, scope+'/conv2', keep_r, is_train, data_format)
    else:
        outs = conv2d(
            outs, num_outs, (1, 1), scope+'/conv2', 1, keep_r, is_train,
            data_format=data_format)
    return outs


def out_block(outs, scope, class_num, is_train, data_format='NHWC'):
    axes = [2, 3] if data_format == 'NCHW' else [1, 2]
    outs = tf.reduce_mean(outs, axes, name=scope+'/pool')
    outs = dense(outs, class_num, scope, data_format=data_format)
    return outs


def conv_out_block(outs, scope, class_num, is_train):
    # need to change the format
    kernel = (3, 3, 9)
    for i in range(3):
        outs = dw_conv2d(outs, (3, 3), 1, scope+'/dw_conv_%s' % i)
        act_fn = None if i == 2 else tf.nn.relu6
        outs = pure_conv2d(
            outs, outs.shape[3].value, kernel, scope+'/pure_%s' % i,
            padding='VALID', act_fn=act_fn)
    outs = tf.squeeze(outs, axis=[1, 2], name=scope+'/squeeze')
    return outs


def pure_conv2d(outs, num_outs, kernel, scope, keep_r=1.0, train=True,
                padding='SAME', act_fn=tf.nn.relu6, data_format='NHWC'):
    stride = int(outs.shape[data_format.index('C')].value/num_outs)
    if data_format == 'NHWC':
        strides = (1, 1, stride)
        outs = tf.expand_dims(outs, axis=-1, name=scope+'/expand_dims')
        df = 'channels_last'
    else:
        strides = (stride, 1, 1)
        outs = tf.expand_dims(outs, axis=1, name=scope+'/expand_dims')
        df = 'channels_first'
    outs = tf.layers.conv3d(
        outs, 1, kernel, strides, padding=padding, activation=act_fn,
        use_bias=False, data_format=df, name=scope+'/pure_conv',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.09))
    if data_format == 'NHWC':
        outs = tf.squeeze(outs, axis=[-1], name=scope+'/squeeze')
    else:
        outs = tf.squeeze(outs, axis=[1], name=scope+'/squeeze')
    return outs


def conv1d(outs, num_outs, kernel, scope, stride=1, keep_r=1.0, train=True,
           data_format='NHWC'):
    df = 'channels_last' if data_format == 'NHWC' else 'channels_first'
    outs = tf.layers.conv1d(
        outs, num_outs, kernel, stride, padding='same', use_bias=False,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.09),
        data_format=df, name=scope+'/conv1d')
    if keep_r < 1.0:
        outs = tf.contrib.layers.dropout(
            outs, keep_r, is_training=train, scope=scope)
    return outs


def conv2d(outs, num_outs, kernel, scope, stride=1, keep_r=1.0, train=True,
           data_format='NHWC'):
    outs = tf.contrib.layers.conv2d(
        outs, num_outs, kernel, scope=scope, stride=stride,
        data_format=data_format, activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.09),
        biases_initializer=None)
    if keep_r < 1.0:
        outs = tf.contrib.layers.dropout(
            outs, keep_r, is_training=train, scope=scope)
    return batch_norm(outs, scope, train, data_format=data_format)


def dw_conv2d(outs, kernel, stride, scope, keep_r=1.0, train=True,
              act_fn=tf.nn.relu6, data_format='NHWC'):
    shape = list(kernel)+[outs.shape[data_format.index('C')].value, 1]
    weights = tf.get_variable(
        scope+'/conv/weight_depths', shape,
        initializer=tf.truncated_normal_initializer(stddev=0.09))
    if data_format == 'NCHW':
        strides = [1, 1, stride, stride]
    else:
        strides = [1, stride, stride, 1]
    outs = tf.nn.depthwise_conv2d(
        outs, weights, strides, 'SAME', name=scope+'/depthwise_conv2d',
        data_format=data_format)
    return act_fn(outs) if act_fn else outs


def dense(outs, dim, scope, data_format='NHWC'):
    outs = tf.contrib.layers.fully_connected(
        outs, dim, activation_fn=None, scope=scope+'/dense',
        weights_initializer=tf.truncated_normal_initializer(stddev=0.09))
    return outs


def batch_norm(outs, scope, is_training=True, act_fn=tf.nn.relu6,
               data_format='NHWC'):
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9997, scale=True, activation_fn=act_fn, fused=True,
        epsilon=1e-3, is_training=is_training, data_format=data_format,
        scope=scope+'/batch_norm')


def global_pool(outs, scope, data_format):
    if data_format == 'NHWC':
        kernel = outs.shape.as_list()[1:3]
    else:
        kernel = outs.shape.as_list()[2:]
    outs = tf.contrib.layers.avg_pool2d(
        outs, kernel, scope=scope, data_format=data_format)
    return outs
