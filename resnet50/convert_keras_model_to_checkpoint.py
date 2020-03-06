#!/usr/bin/env python3
import argparse
import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
from tensorflow.keras import backend as K

if __name__ == '__main__':
    # constants

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_model_path',
        type=str,
        default=None,
        required=True,
        help='Load keras model at path'
    )
    parser.add_argument(
        '--output_model_path',
        type=str,
        default='checkpoint',
        required=False,
        help='Where to save the converted model.'
    )

    args = parser.parse_args()

    model = keras.models.load_model(args.input_model_path)

    sess = tf.compat.v1.keras.backend.get_session()
    saver = tf.compat.v1.train.Saver()
    os.makedirs(args.output_model_path, exist_ok=True)
    saver.save(sess, os.path.join(args.output_model_path, 'converted'))
