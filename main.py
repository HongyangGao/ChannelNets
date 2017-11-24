import os
import time
import argparse
import tensorflow as tf
from model import MobileNet
from tensor_net import run


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_string('data_format', 'NCHW', 'data format for training')
    flags.DEFINE_bool('fake', False, 'use fake data for test or benchmark')
    flags.DEFINE_bool('is_train', True, 'train or not')
    # data
    flags.DEFINE_string('data_dir', 'DATADIR', 'Name of data directory')
    flags.DEFINE_integer('batch', 64, 'batch size')
    flags.DEFINE_integer('class_num', 1000, 'output class number')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('reload_step', '', 'Reload step to continue training')
    flags.DEFINE_string('test_step', '', 'Test or predict model at this step')
    # network architecture
    flags.DEFINE_integer('ch_num', 32, 'channel number')
    flags.DEFINE_integer('block_num', 5, 'block number')
    flags.DEFINE_integer('group_num', 4, 'group number')
    flags.DEFINE_float('keep_r', 0.9999, 'dropout keep rate')

    flags.DEFINE_bool('use_rev_conv', False, 'use reverse conv or not')
    flags.DEFINE_string(
        'block_func', 'conv_group_block',
        'single_block or simple_group_block or conv_group_block')
    flags.DEFINE_string('out_func', 'out_block', 'out_block or conv_out_block')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    conf = configure()
    model = MobileNet(conf)
    run(model)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['TENSORPACK_PIPEDIR'] = '/tmp'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
    tf.app.run()
