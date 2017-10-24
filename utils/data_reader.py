import h5py
import numpy as np


class H5DataLoader(object):

    def __init__(self, data_path, is_train=True):
        self.is_train = is_train
        data_file = h5py.File(data_path, 'r')
        self.images, self.labels = data_file['data'], data_file['labels']
        self.gen_indexes()

    def gen_indexes(self):
        if self.is_train:
            self.indexes = np.random.permutation(range(self.images.shape[0]))
        else:
            self.indexes = np.array(range(self.images.shape[0]))
        self.cur_index = 0

    def next_batch(self, batch_size):
        next_index = self.cur_index+batch_size
        cur_indexes = list(self.indexes[self.cur_index:next_index])
        self.cur_index = next_index
        if len(cur_indexes) < batch_size and self.is_train:
            self.gen_indexes()
            return self.next_batch(batch_size)
        cur_indexes.sort()
        if cur_indexes:
            return self.images[cur_indexes], self.labels[cur_indexes]
        else:
            return None, None


class H5DataLoaderCrop(object):

    def __init__(self, data_path, shape, is_train=True):
        self.is_train = is_train
        self.out_shape = shape
        data_file = h5py.File(data_path, 'r')
        self.images, self.labels = data_file['data'], data_file['labels']
        self.in_shape = self.images.shape[1:-1]
        self.gen_indexes()

    def gen_indexes(self):
        if self.is_train:
            self.indexes = np.random.permutation(range(self.images.shape[0]))
        else:
            self.indexes = np.array(range(self.images.shape[0]))
        self.cur_index = 0

    def next_batch(self, batch_size):
        next_index = self.cur_index+batch_size
        cur_indexes = list(self.indexes[self.cur_index:next_index])
        self.cur_index = next_index
        if len(cur_indexes) < batch_size and self.is_train:
            self.gen_indexes()
            return self.next_batch(batch_size)
        elif len(cur_indexes) < batch_size:
            return None, None
        cur_indexes.sort()
        w_s, h_s = self.get_rand()
        images = self.images[cur_indexes, w_s:w_s+self.out_shape[0], h_s:h_s+self.out_shape[1]]
        return images, self.labels[cur_indexes]

    def get_rand(self):
        if self.is_train:
            w_s = np.random.randint(self.in_shape[0]-self.out_shape[0]+1)
            h_s = np.random.randint(self.in_shape[1]-self.out_shape[1]+1)
        else:
            w_s = int((self.in_shape[0]-self.out_shape[0])/2)
            h_s = int((self.in_shape[1]-self.out_shape[1])/2)
        return w_s, h_s
