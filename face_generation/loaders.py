import cPickle
import os
import cv2
from utils import create_dir_if_not_exist
from progressbar import ETA, ProgressBar
import numpy as np
import theano
from random import shuffle


class MNISTLoader(object):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            train_set, valid_set, test_set = cPickle.load(f)
        self.train_x, self.train_y = train_set
        self.valid_x, self.valid_y = valid_set
        self.test_x, self.test_y = test_set

    def generate_batches(self, batch_size=128, mode='train'):
        x = getattr(self, '%s_x' % mode)
        for i in xrange(0, x.shape[0], batch_size):
            current_batch_size = min(batch_size, x.shape[0] - i)
            yield x[i:i + current_batch_size]


def transform(images_paths, size, output_dir_path, mode):
    widgets = ['Processing: ', ETA()]
    progress_bar = ProgressBar(widgets=widgets, maxval=len(images_paths)).start()

    for i, img_path in enumerate(images_paths):
        img = cv2.imread(img_path)
        if mode == 'grayscale':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        dest_img_path = os.path.join(output_dir_path, os.path.basename(img_path))
        cv2.imwrite(dest_img_path, img)
        progress_bar.update(i + 1)
    progress_bar.finish()


def transform_lfw(input_dir_path, output_dir_path, size, mode, split=True, ratio=0.9):
    """
    :param ratio: how much of train set we use if split is True
    :param split: should we split into train set and test set
    :param input_dir_path: input path to lfw images (in default lfw format)
    :param output_dir_path: output dir for processed images
    :param size: should be tuple (x, y) opencv style
    :param mode: grayscale supported, for others nothing happens
    :return:
    """
    images_paths = []
    create_dir_if_not_exist(output_dir_path)

    for dir_name in os.listdir(input_dir_path):
        dir_path = os.path.join(input_dir_path, dir_name)
        if os.path.isdir(dir_path):
            images_names = [img_name for img_name in os.listdir(dir_path) if img_name.endswith('.jpg')]
            for img_name in images_names:
                img_path = os.path.join(dir_path, img_name)
                images_paths.append(img_path)

    shuffle(images_paths)

    if split:
        train_dir = os.path.join(output_dir_path, 'train')
        test_dir = os.path.join(output_dir_path, 'test')
        create_dir_if_not_exist(train_dir)
        create_dir_if_not_exist(test_dir)
        train_end = int(len(images_paths) * ratio)
        train_images_paths = images_paths[:train_end]
        test_images_paths = images_paths[train_end:]
        transform(train_images_paths, size, train_dir, mode)
        transform(test_images_paths, size, test_dir, mode)
    else:
        transform(images_paths, size, output_dir_path, mode)


def load_images(images_dir_path):
    images = [cv2.imread(os.path.join(images_dir_path, img_name), cv2.CV_LOAD_IMAGE_GRAYSCALE).flatten() for img_name in
              os.listdir(images_dir_path) if
              img_name.endswith('.jpg')]
    shuffle(images)
    images = np.array(images, dtype=theano.config.floatX)
    return images


class LFWLoaderInMemory(object):
    def __init__(self, input_dir_path):
        train_dir_path = os.path.join(input_dir_path, 'train')
        if os.path.exists(train_dir_path):
            self.load_train_images(train_dir_path)
            test_dir_path = os.path.join(input_dir_path, 'test')
            self.load_test_images(test_dir_path)
        else:
            self.load_train_images(input_dir_path)

    def load_train_images(self, train_dir_path):
        images = load_images(train_dir_path)
        self.min_val, self.max_val = images.min(axis=0), images.max(axis=0)
        # normalisation to 0-1
        self.train_x = (images - self.min_val) / (self.max_val - self.min_val)

    def load_test_images(self, test_dir_path):
        images = load_images(test_dir_path)
        self.test_x = self.normalise(images)

    def generate_batches(self, batch_size=128, mode='train'):
        x = getattr(self, '%s_x' % mode)
        for i in xrange(0, x.shape[0], batch_size):
            current_batch_size = min(batch_size, x.shape[0] - i)
            yield x[i:i + current_batch_size]

    def normalise(self, x):
        return (x - self.min_val) / (self.max_val - self.min_val)
