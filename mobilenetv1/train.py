#!/usr/bin/env python3
import argparse
import json
import math
import os

import tensorflow.keras as keras
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_definition import image_size, preprocess_imagenet


def get_model(num_classes):
    input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

    # create the base pre-trained model
    base_model = MobileNet(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # for layer in base_model.layers:
    #     layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(
        x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    x = Dense(num_classes, activation='softmax')(x)  # final layer with softmax activation

    updatedModel = Model(base_model.input, x)

    return updatedModel


def compile_model(compiledModel):
    compiledModel.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=Adadelta(),
                          metrics=['accuracy'])


def modelFitGenerator():

    train_datagen = ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.4,
        preprocessing_function=preprocess_imagenet
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_imagenet
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', shuffle=True,
        interpolation='lanczos'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', shuffle=True,
        interpolation='lanczos'
    )

    num_train_samples = len(train_generator.classes)
    num_valid_samples = len(validation_generator.classes)

    num_train_steps = math.floor(num_train_samples / batch_size)
    num_valid_steps = math.floor(num_valid_samples / batch_size)

    train_classes = len(set(train_generator.classes))
    test_classes = len(set(validation_generator.classes))

    if train_classes != test_classes:
        print('number of classes in train and test do not match, train {}, test {}'.format(train_classes, test_classes))
        exit(1)

    # save class names list before training

    label_map = train_generator.class_indices
    class_idx_to_label = {v: k for k, v in label_map.items()}
    labels = []
    for i in range(len(class_idx_to_label)):
        label = class_idx_to_label[i]
        labels.append(label)

    labels_txt = u"\n".join(labels)
    with open(output_class_names_path, 'w') as classes_f:
        classes_f.write(labels_txt)
    print("Saved class names list file to {}".format(output_class_names_path))

    fitModel = get_model(num_classes=train_classes)
    compile_model(fitModel)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    fitModel.fit_generator(
        train_generator,
        steps_per_epoch=num_train_steps,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=num_valid_steps,
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss]
    )

    fitModel.save(output_model_path, include_optimizer=False)
    print("Saved trained model to {}".format(output_model_path))


def main():
    modelFitGenerator()


if __name__ == '__main__':
    # constants

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        required=True,
        help='Path to folders of labeled images. Expects "train" and "eval" subfolders'
    )
    parser.add_argument(
        '--eval_dataset_path',
        type=str,
        default=None,
        required=True,
        help='Path to folders of labeled eval images'
    )
    parser.add_argument(
        '--output_model_path',
        type=str,
        default='trained_model',
        required=False,
        help='Where to save the trained model.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of training epochs, full passes through the dataset'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Training batch size. Number of images to process at each gradient descent step.'
    )

    args = parser.parse_args()
    train_data_dir = args.dataset_path
    validation_data_dir = args.eval_dataset_path
    nb_epoch = args.epochs
    batch_size = args.batch_size
    output_model_path = args.output_model_path
    output_class_names_path = os.path.join(output_model_path, 'class_names.txt')

    os.makedirs(output_model_path, exist_ok=True)

    with open(os.path.join(output_model_path,'model_schema.json'), 'w') as schema_f:
        schema_f.write(json.dumps({
            "preprocessor": "imagenet",
            "input_shapes": "1,224,224,3",
            "task": "classifier",
            "dataset": "custom"
        }, indent=4))

    output_model_path = os.path.join(output_model_path, 'model.h5')

    main()
