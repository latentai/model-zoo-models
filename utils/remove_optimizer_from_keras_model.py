#!/usr/bin/env python3
import argparse

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
        '--output_model_path',
        type=str,
        default=None,
        required=True,
        help='Where to save the slimmed keras model.'
    )

    args = parser.parse_args()
    tf.keras.backend.set_learning_phase(0)

    model = keras.models.load_model(args.input_model_path)

    model.save(args.output_model_path, include_optimizer=False)
