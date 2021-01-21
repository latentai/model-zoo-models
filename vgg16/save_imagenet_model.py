#!/usr/bin/env python3
import argparse

import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adadelta


def get_model():
    input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

    # create the base pre-trained model
    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=True)

    return base_model


def compile_model(compiledModel):
    compiledModel.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=Adadelta(),
                          metrics=['accuracy'])


def saveImagenetModel():
    fitModel = get_model()
    compile_model(fitModel)

    fitModel.save(output_model_path, include_optimizer=False)
    print("Saved trained model to {}".format(output_model_path))


def main():
    saveImagenetModel()


if __name__ == '__main__':
    # constants

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_model_path',
        type=str,
        default='trained_model.h5',
        required=False,
        help='Where to save the trained model.'
    )
    parser.add_argument(
        '--output_class_names_path',
        type=str,
        default='class_names.txt',
        required=False,
        help='Where to save the class names used by the trained model.'
    ) # NOT USED YET

    args = parser.parse_args()
    output_model_path = args.output_model_path
    output_class_names_path = args.output_class_names_path
    main()
