import os
import time
import argparse
import tensorflow as tf
from model import MobileNet
from tensor_net import run


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 2000000, '# of step for training')
    flags.DEFINE_integer('test_interval', 100, '# of interval to test a model')
    flags.DEFINE_integer('save_interval', 100, '# of interval to save model')
    flags.DEFINE_integer('summary_interval', 10, '# of step to save summary')
    flags.DEFINE_string('opt_name', 'adam', 'adam, rmsprop, or sgd')
    flags.DEFINE_float('learning_rate', 1e-1, 'learning rate')
    flags.DEFINE_float('decay_steps', 500000, 'decay steps')
    flags.DEFINE_float('decay_factor', 0.1, 'decay factor')
    flags.DEFINE_float('epsilon', 1.0, 'epsilon in optimizer')
    flags.DEFINE_string('data_format', 'NHWC', 'data format for training')
    flags.DEFINE_bool('fake', False, 'use fake data for test or benchmark')
    # data
    flags.DEFINE_string('data_dir', '/tempspace2/hgao/data/imagenet/', 'Name of data directory')
    flags.DEFINE_integer('batch', 64, 'batch size')
    flags.DEFINE_integer('channel', 3, 'channel size')
    flags.DEFINE_integer('height', 224, 'height size')
    flags.DEFINE_integer('width', 224, 'width size')
    flags.DEFINE_integer('class_num', 1000, 'output class number')
    # Debug
    flags.DEFINE_string('logdir', './logdir1', 'Log dir')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 0, 'Test or predict model at this step')
    # network architecture
    flags.DEFINE_integer('ch_num', 32, 'channel number')
    flags.DEFINE_integer('block_num', 5, 'block number')
    flags.DEFINE_integer('group_num', 4, 'group number')
    flags.DEFINE_float('keep_r', 0.5, 'dropout keep rate')
    flags.DEFINE_float('regu_r', 1e-4, 'regulation loss rate')
    flags.DEFINE_bool('use_rev_conv', False, 'use reverse conv or not')
    flags.DEFINE_string(
        'block_func', 'conv_group_block',
        'Use which block: single_block or simple_group_block or conv_group_block')
    flags.DEFINE_string(
        'out_func', 'out_block',
        'Use which block: out_block or conv_out_block')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='train',
                        help='actions: train, test')
    args = parser.parse_args()
    conf = configure()
    conf.option = args.option
    conf.is_train = args.option == 'train'
    if args.option not in ['train', 'test']:
        print('invalid option: ', args.option)
        print("Please input a option: train, test")
    else:
        model = MobileNet(conf)
        run(model)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['TENSORPACK_PIPEDIR'] = '/tmp'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    tf.app.run()
