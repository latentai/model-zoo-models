#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    '--image_file',
    type=str,
    default='workspace/datasets/mnist/eval/images/eight/eight.jpg',
    required=False,
    help='Path to image to test'
)
parser.add_argument(
    '--input_model',
    type=str,
    default='pretrained_models/fp32_model/',
    required=False,
    help='Where to load the trained model from.'
)
parser.add_argument(
    '--input_class_names_path',
    type=str,
    default='workspace/datasets/mnist/eval/class_names.txt',
    required=False,
    help='Where to load the class names used by the trained model.'
)

args = parser.parse_args()

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.InteractiveSession()
print("Graph file Dir: {}".format(args.input_model))

# Start the Session
try:
    tf.compat.v1.saved_model.loader.load(sess, tags=['train'],
                                                export_dir=args.input_model)
except RuntimeError:
    try:
        tf.compat.v1.saved_model.loader.load(sess, tags=['serve'],
                                                export_dir=args.input_model)
    except RuntimeError:
        raise RuntimeError("The Saved Model has no tags, \
                            ['train'] or ['serve']")

print('Model is loaded')

class_names = open(args.input_class_names_path, 'r').read().split('\n')

img = image.load_img(args.image_file,color_mode='grayscale')
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
input_tensor = sess.graph.get_tensor_by_name('Placeholder:0')
output = sess.graph.get_tensor_by_name('Softmax:0')
dict_eval = {input_tensor : x}
preds = sess.run(output, feed_dict = dict_eval)
preds = preds[0]
print('Predicted:', preds)

top_values_index = sorted(range(len(preds)), key=lambda i: preds[i])[-5:]

for i in top_values_index:
    label = class_names[i]
    score = preds[i]
    print("index  {}\t{}\t{}".format(i, score, label))
