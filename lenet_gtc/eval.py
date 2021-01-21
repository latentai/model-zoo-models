import os
import tensorflow as tf
import imageio
import logging
import argparse
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

def top1_acc(labels, logits):
    return keras.metrics.top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=1)


def top5_acc(labels, logits):
    return keras.metrics.top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=5)

def evaluateMNIST(args):
    input_model = args.input_path
    ###############################
    #   Load the model
    ###############################
    print("Graph file Dir: {}".format(input_model))

    # Start the Session
    sess = tf.compat.v1.InteractiveSession()
    logging.debug(sess)
    try:
        tf.compat.v1.saved_model.loader.load(sess, tags=['train'],
                                                    export_dir=input_model)
    except RuntimeError:
        try:
            tf.compat.v1.saved_model.loader.load(sess, tags=['serve'],
                                                    export_dir=input_model)
        except RuntimeError:
            raise RuntimeError("The Saved Model has no tags, \
                                ['train'] or ['serve']")

    print('Model is loaded')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape from rank 3 tensor to rank 4 tensor
    x_train = np.reshape(x_train, (x_train.shape[0],28,28,1))
    x_test = np.reshape(x_test, (x_test.shape[0],28,28,1))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('Dataset Statistics')
    print('Training data shape', x_train.shape)
    print('Testing data shape', x_test.shape)

    # data generator
    datagen = ImageDataGenerator(rescale=1. / 255)
    print('Number of training samples', x_train.shape[0])
    print('Number of test samples', x_test.shape[0], '\n')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    input_tensor = sess.graph.get_tensor_by_name('Placeholder:0')
    output = sess.graph.get_tensor_by_name('Softmax:0')
    dict_eval = {input_tensor : x_train}
    train_prediction = sess.run(output, feed_dict = dict_eval)
    
    print('top1 accuracy for train set: ', np.sum(top1_acc(y_train,train_prediction).eval())/len(x_train))
    print('top5 accuracy for train set: ', np.sum(top5_acc(y_train,train_prediction).eval())/len(x_train))
    dict_eval = {input_tensor : x_test}
    test_prediction = sess.run(output, feed_dict = dict_eval)
    print('top1 accuracy for test set: ', np.sum(top1_acc(y_test,test_prediction).eval())/len(x_test))
    print('top5 accuracy for test set: ', np.sum(top5_acc(y_test,test_prediction).eval())/len(x_test))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Path to pb model.')
    parser.add_argument('--basedirectory', help='Directory for the output')
    args = parser.parse_args()

    print('\n')
    print(args)
    print('\n')


    print('Statistics on the mnist test datase')
    evaluateMNIST(args)
    print('Evaluation complete')
