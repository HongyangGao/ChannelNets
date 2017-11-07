import os
import time
import argparse
import tensorflow as tf
from model import MobileNet
from tensor_net import run


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_string('data_format', 'NHWC', 'data format for training')
    flags.DEFINE_bool('fake', False, 'use fake data for test or benchmark')
    # data
    flags.DEFINE_string('data_dir', '/tempspace2/hgao/data/imagenet/', 'Name of data directory')
    flags.DEFINE_integer('batch', 64, 'batch size')
    flags.DEFINE_integer('class_num', 1000, 'output class number')
    # Debug
    flags.DEFINE_string('logdir', './logdir1', 'Log dir')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 0, 'Test or predict model at this step')
    # network architecture
    flags.DEFINE_integer('ch_num', 32, 'channel number')
    flags.DEFINE_integer('block_num', 5, 'block number')
    flags.DEFINE_integer('group_num', 4, 'group number')
    flags.DEFINE_float('keep_r', 0.999, 'dropout keep rate')

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
    conf.option = args.option
    conf.is_train = args.option == 'train'
    model = MobileNet(conf)
    run(model)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['TENSORPACK_PIPEDIR'] = '/tmp'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    tf.app.run()
