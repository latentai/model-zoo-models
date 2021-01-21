#!/usr/bin/env python3
import argparse
import os
import shutil

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras

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
        '--input_class_names_path',
        type=str,
        default='class_names.txt',
        required=False,
        help='Where to load the class names used by the trained model.'
    )
    parser.add_argument(
        '--output_model_path',
        type=str,
        default='checkpoint',
        required=False,
        help='Where to save the converted model.'
    )

    args = parser.parse_args()
    tf.keras.backend.set_learning_phase(0)

    model = keras.models.load_model(args.input_model_path)
    model.save('tmp.h5', include_optimizer=False)
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)
    model2 = keras.models.load_model('tmp.h5')
    os.remove('tmp.h5')

    sess = tf.compat.v1.keras.backend.get_session()
    saver = tf.compat.v1.train.Saver()
    os.makedirs(args.output_model_path, exist_ok=True)
    if os.path.exists(args.input_class_names_path) and os.path.isfile(args.input_class_names_path):
        shutil.copyfile(args.input_class_names_path, os.path.join(args.output_model_path, 'class_names.txt'))
    else:
        print('Warning! No class_names.txt file found. This converted checkpoint will not have an associated class_names.txt file present')
    saver.save(sess, os.path.join(args.output_model_path, 'converted'))
