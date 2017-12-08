from utils import ops


class MobileNet(object):

    def __init__(self, conf):
        self.conf = conf

    def inference(self, images):
        cur_out_num = self.conf.ch_num
        outs = ops.conv2d(
            images, cur_out_num, (3, 3), 'conv_s', train=self.conf.is_train,
            stride=2, data_format=self.conf.data_format)
        cur_out_num *= 2
        outs = ops.dw_block(  # 112 * 112 * 64
            outs, cur_out_num, 1, 'conv_1_0', self.conf.keep_r,
            self.conf.is_train, data_format=self.conf.data_format)
        cur_out_num *= 2
        outs = ops.dw_block(  # 56 * 56 * 128
            outs, cur_out_num, 2, 'conv_1_1', self.conf.keep_r,
            self.conf.is_train, data_format=self.conf.data_format)
        outs = ops.dw_block(  # 56 * 56 * 128
            outs, cur_out_num, 1, 'conv_1_2', self.conf.keep_r,
            self.conf.is_train, data_format=self.conf.data_format)
        cur_out_num *= 2
        outs = ops.dw_block(  # 28 * 28 * 256
            outs, cur_out_num, 2, 'conv_1_3', self.conf.keep_r,
            self.conf.is_train, data_format=self.conf.data_format)
        outs = ops.dw_block(  # 28 * 28 * 256
            outs, cur_out_num, 1, 'conv_1_4', self.conf.keep_r,
            self.conf.is_train, data_format=self.conf.data_format)
        cur_out_num *= 2
        outs = ops.dw_block(  # 14 * 14 * 512
            outs, cur_out_num, 2, 'conv_1_5', self.conf.keep_r,
            self.conf.is_train, data_format=self.conf.data_format)
        outs = self.get_block_func()(  # 14 * 14 * 512
            outs, self.conf.block_num, self.conf.keep_r, self.conf.is_train,
            'conv_2', self.conf.data_format, self.conf.group_num)
        cur_out_num *= 2
        outs = ops.dw_block(  # 7 * 7 * 1024
            outs, cur_out_num, 2, 'conv_3_0', self.conf.keep_r,
            self.conf.is_train, data_format=self.conf.data_format)
        outs = ops.dw_block(  # 7 * 7 * 1024
            outs, cur_out_num, 1, 'conv_3_1', self.conf.keep_r,
            self.conf.is_train, self.conf.use_rev_conv,
            self.conf.rev_kernel_size,
            data_format=self.conf.data_format)
        outs = self.get_out_func()(
            outs, 'out', self.conf.class_num, self.conf.is_train,
            data_format=self.conf.data_format)
        return outs

    def get_block_func(self):
        return getattr(ops, self.conf.block_func)

    def get_out_func(self):
        return getattr(ops, self.conf.out_func)
