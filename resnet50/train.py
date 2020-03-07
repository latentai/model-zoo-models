#!/usr/bin/env python3
import argparse

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adadelta
import tensorflow.keras as keras
import math, os, sys
import matplotlib.pyplot as plt

from model_definition import image_size


def get_model(num_classes):
    input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

    # create the base pre-trained model
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dense(num_classes, activation='softmax')(x)

    updatedModel = Model(base_model.input, x)

    return updatedModel


def compile_model(compiledModel):
    compiledModel.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=Adadelta(),
                          metrics=['accuracy'])


def modelFitGenerator():
    num_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])

    num_train_steps = math.floor(num_train_samples / batch_size)
    num_valid_steps = math.floor(num_valid_samples / batch_size)

    train_datagen = ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.4)

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', shuffle=True
    )

    train_classes = len(set(train_generator.classes))
    test_classes = len(set(validation_generator.classes))

    if train_classes != test_classes:
        print('number of classes in train and test do not match, train {}, test {}'.format(train_classes, test_classes))
        exit(1)

    # save class names list before training


    label_map = (train_generator.class_indices)
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
    history = fitModel.fit_generator(
        train_generator,
        steps_per_epoch=num_train_steps,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=num_valid_steps)

    # printGraph(history)
    fitModel.save(output_model_path)
    print("Saved trained model to {}".format(output_model_path))


def printGraph(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


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
    train_data_dir = os.path.join(args.dataset_path, 'train')
    validation_data_dir = os.path.join(args.dataset_path, 'eval')
    nb_epoch = args.epochs
    batch_size = args.batch_size
    output_model_path = args.output_model_path
    output_class_names_path = args.output_class_names_path
    main()
