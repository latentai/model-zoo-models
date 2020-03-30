#!/usr/bin/python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../')
sys.path.append(dir_path + '/../../')
sys.path.append(dir_path)
sys.path.append(dir_path.replace('ssd_kerasV2', ''))
sys.path.append('.')

import keras
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt

import numpy as np
import pickle
import argparse
import yaml
from random import shuffle
from PIL import Image

from model.ssd300MobileNetV2Lite import SSD
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)


class Generator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3. / 4., 4. / 3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y + h, x:x + w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                    y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets

    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:
                img_path = self.path_prefix + key
                img = Image.open(img_path)
                img = img.resize(self.image_size, Image.ANTIALIAS)
                img = np.array(img).astype('float32')
                y = self.gt[key].copy()

                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)

                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_settings', help='Path to dataset')
    parser.add_argument('--pretrained_weights', help='Model pretrained weights')

    NUM_CLASSES = 21
    args = parser.parse_args()

    with open(args.path_to_settings, 'r') as fp:
        sets = yaml.safe_load(fp)

    input_shape = (sets['img_height'], sets['img_width'], 3)
    batch_size = sets['batch_size']

    priors = pickle.load(open(os.path.join(dir_path, 'priorFiles/prior_boxes_ssd300MobileNetV2.pkl'), 'rb'))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    gt = pickle.load(open(os.path.join(dir_path, 'voc_2007.pkl'), 'rb'))
    keys = sorted(gt.keys())
    num_train = int(round(0.8 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)

    path_prefix = os.path.join(sets['dataset_dir'], 'VOC2007/JPEGImages/')
    gen = Generator(gt, bbox_util, batch_size, path_prefix,
                    train_keys, val_keys,
                    (input_shape[0], input_shape[1]), do_crop=False)

    init_op = tf.initialize_all_variables()
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                                            device_count={'GPU': 0}
                                            )

    with tf.Session(graph=tf.get_default_graph(), config=session_conf) as sess:
        # sess.run(init_op)
        model = SSD(input_shape, num_classes=NUM_CLASSES)
        if args.pretrained_weights:
            model.load_weights(args.pretrained_weights, by_name=True)

        freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
                  'conv2_1', 'conv2_2', 'pool2',
                  'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

        for L in model.layers:
            if L.name in freeze:
                L.trainable = False

        def schedule(epoch, decay=0.9):
            return base_lr * decay ** (epoch)


        callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(sets['tf_model_path'], 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                                     verbose=1, save_weights_only=True),
                     # keras.callbacks.LearningRateScheduler(schedule)
                     ]

        base_lr = sets['learning_rate']
        optim = keras.optimizers.Adam(lr=base_lr)
        model.compile(optimizer=optim, loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)
        # import pdb
        # pdb.set_trace()
        # graph = tf.compat.v1.get_default_graph()
        # [n.name for n in graph.as_graph_def().node]
        #
        # graph.get_tensor_by_name()

        nb_epoch = sets['epochs']
        history = model.fit_generator(gen.generate(True), gen.train_batches,
                                      nb_epoch, verbose=1,
                                      callbacks=callbacks,
                                      validation_data=gen.generate(False),
                                      nb_val_samples=gen.val_batches,
                                      nb_worker=1)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(sets['tf_model_path'], 'model'))
        tf.summary.FileWriter(os.path.join(sets['tf_model_path'], 'model'), sess.graph)

    print('\n\nTraining completed')





