import os
import tensorflow as tf
from tensorpack import (
    logger, ModelSaver, InferenceRunner, imgaug, dataset, InputDesc,
    ModelDesc, AugmentImageComponent, BatchData, PrefetchDataZMQ,
    StopTraining, ScalarStats, StatMonitorParamSetter)
from tensorpack.train import TrainConfig, QueueInputTrainer
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.models import regularize_cost


class Model(ModelDesc):

    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.weight_decay = 4e-5

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 30, 30, 3), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        if self.net.net.conf.data_format == 'NHWC':
            image = tf.transpose(image, [0, 3, 1, 2])
        image = image / 128.0 - 1
        logits = self.net.net.inference(image)
        cost = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=label, scope='cross_entropy_loss')
        #correct = tf.to_float(tf.nn.in_top_k(logits, label, 1), name='correct')
        correct = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, 1), label)))
        # monitor training error
        add_moving_summary(tf.reduce_mean(correct, name='accuracy'))
        wd_cost = regularize_cost(
            '.*/weights', tf.contrib.layers.l2_regularizer(self.weight_decay),
            name='regularize_loss')
        add_moving_summary(cost, wd_cost)
        add_param_summary(('.*/weights', ['histogram']))
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable(
            'learning_rate', initializer=1e-2, trainable=False)
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr, epsilon=1e-3)


def get_data(train_or_test, batch):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar100(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.RandomCrop((30, 30)),
            imgaug.Flip(horiz=True),
            imgaug.Brightness(63),
            imgaug.Contrast((0.2, 1.8)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.CenterCrop((30, 30)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, batch, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 5)
    return ds


def get_config(model, conf):
    dataset_train = get_data('train', conf.batch)
    dataset_test = get_data('test', conf.batch)

    def lr_func(lr):
        if lr < 3e-5:
            raise StopTraining()
        return lr * 0.31
    config = TrainConfig(
        model=Model(model),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            ScalarStats(['accuracy', 'cost'])),
            StatMonitorParamSetter('learning_rate', 'val_error', lr_func,
                                   threshold=0.001, last_k=10)],
        max_epoch=150,
    )
    return config


def run(net):
    instance = Model(net)
    logger.set_logger_dir(net.conf.logdir)
    config = get_config(instance, net.conf)
    if net.conf.reload_step:
        config.session_init = get_model_loader(
            net.conf.logdir+'/'+net.conf.reload_step)
    QueueInputTrainer(config).train()
