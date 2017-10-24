import os
import time
import numpy as np
import tensorflow as tf
from utils.data_reader import H5DataLoaderCrop
from utils import ops, optimizer


class MobileNet(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    def def_params(self):
        self.axis, self.channel_axis = (1, 2), 3
        self.input_shape = [
            self.conf.batch, self.conf.height, self.conf.width,
            self.conf.channel]
        self.output_shape = [self.conf.batch]

    def configure_networks(self):
        self.build_network()
        self.cal_loss()
        global_step = tf.Variable(0, trainable=False)
        learning_rate = self.get_learning_rate(global_step)
        optimizer = self.get_optimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(
                self.loss_op, global_step=global_step, name='train_op')
        tf.set_random_seed(int(time.time()))
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        if self.conf.option == 'train':
            self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.print_params_num()

    def get_learning_rate(self, global_step):
        return tf.train.exponential_decay(
            self.conf.learning_rate, global_step, self.conf.decay_steps,
            self.conf.decay_factor, staircase=True, name='learning_rate')

    def get_optimizer(self, learning_rate):
        return getattr(optimizer, self.conf.opt_name)(
            learning_rate, epsilon=self.conf.epsilon)

    def build_network(self):
        self.inputs = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.labels = tf.placeholder(
            tf.int64, self.output_shape, name='labels')
        self.preds = self.inference(self.inputs)

    def inference(self, inputs):
        cur_out_num = self.conf.ch_num # 112 * 112 * 32
        outs = ops.conv2d(inputs, cur_out_num, (3, 3), 'conv_s', stride=2)
        cur_out_num *= 2
        outs = ops.dw_block(# 112 * 112 * 64
            outs, cur_out_num, 1, 'conv_1_0', self.conf.keep_r, self.conf.is_train)
        cur_out_num *= 2
        outs = ops.dw_block(# 56 * 56 * 128
            outs, cur_out_num, 2, 'conv_1_1', self.conf.keep_r, self.conf.is_train)
        outs = ops.dw_block(# 56 * 56 * 128
            outs, cur_out_num, 1, 'conv_1_2', self.conf.keep_r, self.conf.is_train)
        cur_out_num *= 2
        outs = ops.dw_block(# 28 * 28 * 256
            outs, cur_out_num, 2, 'conv_1_3', self.conf.keep_r, self.conf.is_train)
        outs = ops.dw_block(# 28 * 28 * 256
            outs, cur_out_num, 1, 'conv_1_4', self.conf.keep_r, self.conf.is_train)
        cur_out_num *= 2
        outs = ops.dw_block(# 14 * 14 * 512
            outs, cur_out_num, 2, 'conv_1_5', self.conf.keep_r, self.conf.is_train)
        outs = self.get_block_func()(# 14 * 14 * 512
            outs, self.conf.block_num, self.conf.keep_r, self.conf.is_train, 'conv_2',
            self.conf.group_num)
        cur_out_num *= 2
        outs = ops.dw_block(# 7 * 7 * 1024
            outs, cur_out_num, 2, 'conv_3_0', self.conf.keep_r, self.conf.is_train)
        outs = ops.dw_block(# 7 * 7 * 1024
            outs, cur_out_num, 1, 'conv_3_1', self.conf.keep_r, self.conf.is_train,
            self.conf.use_rev_conv)
        outs = self.get_out_func()(outs, 'out', self.conf.class_num, self.conf.is_train)
        return outs

    def get_block_func(self):
        return getattr(ops, self.conf.block_func)

    def get_out_func(self):
        return getattr(ops, self.conf.out_func)

    def cal_loss(self):
        with tf.variable_scope('loss'):
            self.class_loss = tf.losses.sparse_softmax_cross_entropy(
                logits=self.preds, labels=self.labels)
            self.regu_loss = tf.losses.get_regularization_loss(name='regu_loss_op')
            self.loss_op = self.class_loss + self.conf.regu_r * self.regu_loss
        with tf.variable_scope('accuracy'):
            self.dec_preds = tf.argmax(self.preds, 1)
            self.accuracy_op = tf.reduce_mean(
                tf.cast(tf.equal(self.dec_preds, self.labels),
                    tf.float32))
            self.accuracy_top_1 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.preds, self.labels, 1),
                        tf.float32))
            self.accuracy_top_5 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.preds, self.labels, 5),
                        tf.float32))

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/class_loss', self.class_loss))
        summarys.append(tf.summary.scalar(name+'/regu_loss', self.regu_loss))
        summarys.append(tf.summary.scalar(name+'/accuracy_op', self.accuracy_op))
        summarys.append(tf.summary.scalar(name+'/accuracy_top_1', self.accuracy_top_1))
        summarys.append(tf.summary.scalar(name+'/accuracy_top_5', self.accuracy_top_5))
        summary = tf.summary.merge(summarys)
        return summary

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        train_reader = H5DataLoaderCrop(
            self.conf.data_dir+self.conf.train_data, self.input_shape[1:-1])
        valid_reader = H5DataLoaderCrop(
            self.conf.data_dir+self.conf.valid_data, self.input_shape[1:-1])
        for epoch_num in range(self.conf.max_step+1):
            if epoch_num and epoch_num % self.conf.test_interval == 0:
                inputs, labels = valid_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.labels: labels}
                loss, summary = self.sess.run(
                    [self.loss_op, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
                print('----testing loss', loss)
            if epoch_num and epoch_num % self.conf.summary_interval == 0:
                inputs, labels = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.labels: labels}
                loss, _, summary = self.sess.run(
                    [self.loss_op, self.train_op, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
            else:
                inputs, labels = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.labels: labels}
                loss, _ = self.sess.run(
                    [self.loss_op, self.train_op], feed_dict=feed_dict)
                print('----training loss', loss)
            if epoch_num and epoch_num % self.conf.save_interval == 0:
                self.save(epoch_num+self.conf.reload_step)

    def test(self):
        print('---->testing ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step in main.py")
            return
        test_reader = H5DataLoaderCrop(
            self.conf.data_dir+self.conf.test_data, self.input_shape[1:-1],
            False)
        accuracies = []
        while True:
            inputs, labels = test_reader.next_batch(self.conf.batch)
            if inputs is None or inputs.shape[0] < self.conf.batch:
                break
            feed_dict = {self.inputs: inputs, self.labels: labels}
            accur, preds  = self.sess.run(
                [[self.accuracy_top_1, self.accuracy_top_5], self.dec_preds],
                feed_dict=feed_dict)
            print('--------------->', accur, preds, labels)
            accuracies.append(accur)
        print('accuracy top1 is ', np.mean(accuracies, axis=0))

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)

    def print_params_num(self):
        total_params = 0
        for var in tf.trainable_variables():
            if 'layer_norm' not in var.name:
                total_params += var.shape.num_elements()
        print("The total number of params --------->", total_params)