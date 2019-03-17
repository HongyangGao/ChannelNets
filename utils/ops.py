import tensorflow as tf
import tensorpack as tp
import numpy as np


def rev_conv2d(outs, scope, rev_kernel_size, keep_r=1.0, train=True, data_format='NHWC'):
    if data_format == 'NHWC':
        outs = tf.transpose(outs, perm=[0, 3, 1, 2], name=scope+'/trans1')
    pre_shape = [-1] + outs.shape.as_list()[1:]
    hw_dim = np.prod(outs.shape.as_list()[2:])
    new_shape = [-1, outs.shape.as_list()[1]] + [hw_dim]
    outs = tf.reshape(outs, new_shape, name=scope+'/reshape1')
    num_outs = outs.shape.as_list()[-1]
    kernel = rev_kernel_size
    outs = conv1d(
        outs, num_outs, kernel, scope+'/conv1d', 1, keep_r, train)
    outs = tf.reshape(outs, pre_shape, name=scope+'/reshape2')
    if data_format == 'NHWC':
        outs = tf.transpose(outs, perm=[0, 2, 3, 1], name=scope+'/trans2')
    return outs


def single_block(outs, block_num, keep_r, is_train, scope, data_format, *args):
    num_outs = outs.shape[data_format.index('C')].value
    for i in range(block_num):
        outs = dw_block(
            outs, num_outs, 1, scope+'/conv_%s' % i, keep_r, is_train,
            data_format=data_format)
        #outs = tf.add(outs, cur_outs, name=scope+'/add_%s' % i)
    return outs


def simple_group_block(outs, block_num, keep_r, is_train, scope, data_format,
                       group, *args):
    results = []
    split_outs = tf.split(outs, group, data_format.index('C'), name=scope+'/split')
    for g in range(group):
        cur_outs = single_block(
            split_outs[g], block_num, keep_r, is_train, scope+'/group_%s' % g,
            data_format)
        results.append(cur_outs)
    results = tf.concat(results, data_format.index('C'), name=scope+'/concat')
    return results


def conv_group_block(outs, block_num, keep_r, is_train, scope, data_format,
                     group, *args):
    num_outs = int(outs.shape[data_format.index('C')].value/group)
    shape = [1, 1, 4*group] if data_format == 'NHWC' else [4*group, 1, 1]
    results = []
    conv_outs = pure_conv2d(
        outs, num_outs, shape, scope+'/pure_conv', keep_r,
        is_train, chan_num=group, data_format=data_format)
    axis = -1 if data_format=='NHWC' else 1
    conv_outs = tf.unstack(conv_outs, axis=axis, name=scope+'/unstack')
    for g in range(group):
        cur_outs = single_block(
            conv_outs[g], block_num, keep_r, is_train, scope+'/group_%s' % g,
            data_format)
        results.append(cur_outs)
    results = tf.concat(results, data_format.index('C'), name=scope+'/concat')
    return results


def out_block(outs, scope, class_num, is_train, data_format='NHWC'):
    axes = [2, 3] if data_format == 'NCHW' else [1, 2]
    outs = tf.reduce_mean(outs, axes, name=scope+'/pool')
    outs = dense(outs, class_num, scope, is_train, data_format=data_format)
    return outs


def conv_out_block(outs, scope, class_num, is_train, data_format='NHWC'):
    if data_format == 'NHWC':
        outs = tf.transpose(outs, perm=[0, 3, 1, 2], name=scope+'/trans')
    kernel = (25, 7, 7)
    outs = dw_conv2d(
        outs, (3, 3), 1, scope+'/conv', data_format='NCHW')
    outs = pure_conv2d(
        outs, outs.shape[1].value, kernel, scope+'/pure',
        padding='VALID', data_format='NCHW')
    outs = tf.squeeze(outs, axis=[2, 3], name=scope+'/squeeze')
    return outs


def pure_conv2d(outs, num_outs, kernel, scope, keep_r=1.0, train=True,
                padding='SAME', chan_num=1, data_format='NHWC'):
    stride = int(outs.shape[data_format.index('C')].value/num_outs)
    if data_format == 'NHWC':
        strides = (1, 1, stride)
        axis = -1
        df = 'channels_last'
    else:
        strides = (stride, 1, 1)
        axis = 1
        df = 'channels_first'
    outs = tf.expand_dims(outs, axis=axis, name=scope+'/expand_dims')
    outs = tf.layers.conv3d(
        outs, chan_num, kernel, strides, padding=padding, use_bias=False,
        data_format=df, name=scope+'/pure_conv',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.09))
    if keep_r < 1.0:
        outs = dropout(outs, keep_r, scope=scope)
    if chan_num == 1:
        outs = tf.squeeze(outs, axis=[axis], name=scope+'/squeeze')
    return outs


def conv1d(outs, num_outs, kernel, scope, stride=1, keep_r=1.0, train=True,
           data_format='NHWC', padding='same'):
    df = 'channels_last' if data_format == 'NHWC' else 'channels_first'
    outs = tf.layers.conv1d(
        outs, num_outs, kernel, stride, padding=padding, use_bias=False,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.09),
        data_format=df, name=scope+'/conv1d')
    if keep_r < 1.0:
        outs = dropout(outs, keep_r, scope=scope)
    return outs


def conv2d(outs, num_outs, kernel, scope, stride=1, keep_r=1.0, train=True,
           act_fn=tf.nn.relu6, data_format='NHWC'):
    outs = tf.contrib.layers.conv2d(
        outs, num_outs, kernel, scope=scope, stride=stride,
        data_format=data_format, activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.09),
        biases_initializer=None)
    if keep_r < 1.0:
        outs = dropout(outs, keep_r, scope=scope)
    return batch_norm(outs, scope, train, act_fn=act_fn, data_format=data_format)


def dw_conv2d(outs, kernel, stride, scope, keep_r=1.0, train=True,
              act_fn=None, data_format='NHWC'):
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


def dense(outs, dim, scope, train=True, data_format='NHWC'):
    outs = tf.contrib.layers.fully_connected(
        outs, dim, activation_fn=None, scope=scope+'/dense',
        weights_initializer=tf.truncated_normal_initializer(stddev=0.09))
    return outs


def dw_block(outs, num_outs, stride, scope, keep_r, is_train,
             use_rev_conv=False, rev_kernel_size=64, act_fn=tf.nn.relu6,
             data_format='NHWC'):
    outs = dw_conv2d(
        outs, (3, 3), stride, scope+'/conv1', keep_r, is_train,
        data_format=data_format)
    if use_rev_conv:
        outs = rev_conv2d(
            outs, scope+'/conv2', rev_kernel_size, keep_r, is_train, data_format)
    else:
        outs = conv2d(
            outs, num_outs, (1, 1), scope+'/conv2', 1, keep_r, is_train,
            act_fn=act_fn, data_format=data_format)
    return outs


def batch_norm_old(outs, scope, is_training=True, act_fn=tf.nn.relu6,
                   data_format='NHWC', not_final=True):
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9997, scale=not_final, center=not_final, activation_fn=act_fn,
        fused=True, epsilon=1e-3, is_training=is_training, data_format=data_format,
        scope=scope+'/batch_norm')


def batch_norm(outs, scope, is_training=True, act_fn=tf.nn.relu6,
               data_format='NHWC', not_final=True):
    df = 'channels_last' if data_format=='NHWC' else 'channels_first'
    outs = tp.BatchNorm(scope+'/bn', outs, data_format=df)
    return outs if not act_fn else act_fn(outs, name=scope+'/act')


def dropout(outs, keep_r, scope):
    if keep_r < 1.0:
        outs = tp.Dropout(scope+'/dropout', outs, rate=1-keep_r)
    return outs


def global_pool(outs, scope, data_format):
    if data_format == 'NHWC':
        kernel = outs.shape.as_list()[1:3]
    else:
        kernel = outs.shape.as_list()[2:]
    outs = tf.contrib.layers.avg_pool2d(
        outs, kernel, scope=scope, data_format=data_format)
    return outs


def skip_pool(outs, scope, data_format):
    df = 'channels_last' if data_format=='NHWC' else 'channels_first'
    outs = tf.layers.average_pooling2d(
        outs, 2, 2, data_format=df, name=scope+'/avg')
    return outs
