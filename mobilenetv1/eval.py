#!/usr/bin/env python3

import argparse
import os

import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_definition import image_size, preprocess_imagenet


def top1_acc(labels, logits):
    return keras.metrics.top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=1)


def top5_acc(labels, logits):
    return keras.metrics.top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=5)



if __name__ == '__main__':
    # constants

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        required=True,
        help='Path to folders of labeled images. '
    )
    parser.add_argument(
        '--input_model_path',
        type=str,
        default=None,
        required=True,
        help='Load keras model at path'
    )
    args = parser.parse_args()

    model = keras.models.load_model(args.input_model_path)

    validation_data_dir = args.dataset_path
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_imagenet
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=image_size,
        batch_size=1,
        class_mode='categorical', shuffle=True,
        interpolation='lanczos'
    )

    model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=Adadelta(),
                          metrics=[top1_acc, top5_acc])

    print('\n# Evaluate')
    result = model.evaluate(validation_generator)
    print(dict(zip(model.metrics_names, result)))
