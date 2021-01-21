#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

parser = argparse.ArgumentParser()
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
    default='trained_model/trained_model.h5',
    required=False,
    help='Where to load the trained model.'
)
parser.add_argument(
    '--input_class_names_path',
    type=str,
    default='trained_model/class_names.txt',
    required=False,
    help='Where to load the class names used by the trained model.'
)

args = parser.parse_args()

model = keras.models.load_model(args.input_model_path)

class_names = open(args.input_class_names_path, 'r').read().split('\n')

img = image.load_img(args.image_file, target_size=(224, 224), interpolation='lanczos')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)[0]
#print('Predicted:', preds)

top_values_index = sorted(range(len(preds)), key=lambda i: preds[i])[-5:]

for i in top_values_index:
    label = class_names[i]
    score = preds[i]
    print("index  {}\t{}\t{}".format(i, score, label))
