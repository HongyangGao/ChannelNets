import os
import glob
import h5py
import numpy as np
import scipy.misc

MEAN = np.array([104., 117., 124.])


def get_files(data_dir):
    return glob.glob(data_dir+'*.tar')


def extract_imgs(target_dir, files, seperate_dir=True):
    for file in files:
        target_folder = target_dir
        if seperate_dir:
            file_name = file.rsplit('/', 1)[-1].split('.')[0]
            target_folder += file_name
            os.system('mkdir %s' % target_folder)
        print(file, 'tar -C %s -xvf %s' % (target_folder, file))
        os.system('tar -C %s -xvf %s' % (target_folder, file))


def get_image_names(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines


def transform(img_path, shape, img_mean):
    img = scipy.misc.imread(img_path).astype(np.float64)
    img = scipy.misc.imresize(img, shape)
    print(img_path, shape, img.shape)
    if len(img.shape) < 3:
        new_img = np.ones(shape)
        new_img[:,:,0] = img
        new_img[:,:,1] = img
        new_img[:,:,2] = img
        img = new_img
    if img.shape[2] != 3:
        img = img[:,:,:3]
    img = img - img_mean 
    img = img / 127.5 - 1.
    return img


def build_h5(file_name, data_dir, out_dir, shape=(256, 256, 3), name='train'):
    dataset = h5py.File(out_dir+name+'.h5', 'w')
    image_files = get_image_names(file_name)
    dataset.create_dataset('data', (len(image_files), *shape), dtype='f')
    dataset.create_dataset('labels', (len(image_files),), dtype='i')
    for i, line in enumerate(image_files):
        img_path, label = line.strip().split()
        img = transform(data_dir+img_path, shape, MEAN)
        dataset['data'][i] = img
        dataset['labels'][i] = int(label)
        if i % 1000 == 0:
            print('processing------------------->', i)
    dataset.close()


if __name__ == '__main__':
    #files = get_files('/tempspace/hgao/data/imagenet/train/')
    #extract_imgs('/tempspace/hgao/data/imagenet/train/', files)
    build_h5(
        '/tempspace/hgao/data/imagenet/files/train.txt',
        '/tempspace/hgao/data/imagenet/train/',
        '/tempspace/hgao/data/imagenet/', name='train')
    #build_h5(
    #    '/tempspace2/hgao/data/imagenet/files/test.txt',
    #    '/tempspace2/hgao/data/imagenet/test/',
    #    '/tempspace2/hgao/data/imagenet/', name='test')
    #build_h5(
    #    '/tempspace2/hgao/data/imagenet/files/val.txt',
    #    '/tempspace2/hgao/data/imagenet/val/',
    #    '/tempspace2/hgao/data/imagenet/', name='val')
