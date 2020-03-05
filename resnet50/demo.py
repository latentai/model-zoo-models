#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_definition import image_size

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_path',
    type=str,
    default=None,
    required=True,
    help='Path to folders of labeled images. Expects "train" and "eval" subfolders'
)
parser.add_argument(
    '--image_file',
    type=str,
    default=None,
    required=True,
    help='Path to image to test'
)
parser.add_argument(
    '--input_model_path',
    type=str,
    default='trained_model.h5',
    required=False,
    help='Where to load the trained model.'
)

args = parser.parse_args()
train_data_dir = os.path.join(args.dataset_path, 'train')

model = keras.models.load_model(args.input_model_path)

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=1,
    class_mode='categorical', shuffle=True
)
label_map = (train_generator.class_indices)
class_idx_to_label = {v: k for k, v in label_map.items()}

img = image.load_img(args.image_file, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)[0]
print('Predicted:', preds)

for i in range(len(preds)):
    label = class_idx_to_label[i]
    score = preds[i]
    print("{}\t{}".format(score, label))
