import os
import sys
import pickle
import tarfile
import h5py
import urllib.request
import numpy as np


DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def maybe_download_and_extract(dest_directory):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count*block_size)/float(total_size)*100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        print('\nSuccessfully download', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def build_h5_dataset(data_dir, out_dir, shape=(32, 32, 3), name='cifar10'):
    dataset = h5py.File(out_dir+name+'train.h5', 'w')
    dataset.create_dataset('data', (50000, *shape), dtype='f')
    dataset.create_dataset('labels', (50000,), dtype='i')
    for i in range(5):
        file = os.path.join(
            data_dir, 'cifar-10-batches-py', 'data_batch_%s' % (i + 1))
        data = unpickle(file)
        for index in range(10000):
            dataset['data'][i*10000+index] = np.reshape(
                data[b'data'][index]/128.0-1, shape, 'F')
            dataset['labels'][i*10000+index] = data[b'labels'][index]
    dataset.close()
    dataset = h5py.File(out_dir+name+'valid.h5', 'w')
    dataset.create_dataset('data', (10000, *shape), dtype='f')
    dataset.create_dataset('labels', (10000,), dtype='i')
    file = os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch')
    data = unpickle(file)
    for index in range(10000):
        dataset['data'][index] = np.reshape(
            data[b'data'][index]/128.0-1, shape, 'F')
        dataset['labels'][index] = data[b'labels'][index]
    dataset.close()


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


if __name__ == '__main__':
    # maybe_download_and_extract('../dataset/')
    build_h5_dataset('../dataset/', '../dataset/')
