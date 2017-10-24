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
    flags.DEFINE_string('data_format', 'NCHW', 'data format for training')
    flags.DEFINE_bool('fake', False, 'use fake data for test or benchmark')
    # data
    flags.DEFINE_string('data_dir', '/tempspace2/hgao/data/imagenet/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'cifar10train.h5', 'Training data')
    flags.DEFINE_string('valid_data', 'cifar10valid.h5', 'Validation data')
    flags.DEFINE_string('test_data', 'cifar10valid.h5', 'Testing data')
    flags.DEFINE_integer('batch', 128, 'batch size')
    flags.DEFINE_integer('channel', 3, 'channel size')
    flags.DEFINE_integer('height', 32, 'height size')
    flags.DEFINE_integer('width', 32, 'width size')
    flags.DEFINE_integer('class_num', 10, 'output class number')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 0, 'Test or predict model at this step')
    # network architecture
    flags.DEFINE_integer('ch_num', 32, 'channel number')
    flags.DEFINE_integer('block_num', 5, 'block number')
    flags.DEFINE_integer('group_num', 4, 'group number')
    flags.DEFINE_float('keep_r', 0.8, 'dropout keep rate')
    flags.DEFINE_float('regu_r', 1e-4, 'regulation loss rate')
    flags.DEFINE_bool('use_rev_conv', False, 'use reverse conv or not')
    flags.DEFINE_string(
        'block_func', 'conv_group_block',
        'Use which block: single_block or simple_group_block or conv_group_block')
    flags.DEFINE_string(
        'out_func', 'conv_out_block',
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
    conf.is_train = args.option === 'train'
    if args.option not in ['train', 'test']:
        print('invalid option: ', args.option)
        print("Please input a option: train, test")
    else:
        model = MobileNet(tf.Session(), conf)
        getattr(model, args.option)()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()
