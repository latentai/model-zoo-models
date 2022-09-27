import json
import logging
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
from config import config
from leip_tf_model import L1
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops import variables

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

from leip.compress.training.gtc.conv2d import Conv2d as gtcConv2d
from leip.compress.training.gtc.dense import Dense as gtcDense


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


set_tf_loglevel(logging.FATAL)


def print_summary_short(model, data_dict):
    outputs = ['hp_cross_entropy', 'total_loss', 'distillation_loss', 'bit_loss', 'regularization_term', 'lp_accuracy',
               'hp_accuracy']
    for k, v in model._summary.items():
        if k in outputs:
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    if isinstance(v1, tuple):
                        print(k, k1, v1[0].eval(data_dict))
                    else:
                        print(k, k1, v1.eval(data_dict))
            else:
                print(k, v.eval(data_dict))


def main():
    # define model, optimizer, training, hyperparams
    args = config()
    _BASEDIR = args.basedirectory
    # print('*'*20)
    # print(args)
    # print('*'*20)

    print("Name of the Experiment", args.name_of_experiment, '\n')
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape from rank 3 tensor to rank 4 tensor
    x_train = np.reshape(x_train, (x_train.shape[0], args.image_size[0],
                                   args.image_size[1], args.image_size[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], args.image_size[0],
                                 args.image_size[1], args.image_size[2]))

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
    y_train = keras.utils.to_categorical(y_train, args.num_classes)
    y_test = keras.utils.to_categorical(y_test, args.num_classes)

    x_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        [None, args.image_size[0], args.image_size[1], args.image_size[2]])
    learning_rate = tf.compat.v1.placeholder(tf.float32, name='lr')

    # create the model layers
    # try:
    #    with tf.Session(graph=tf.Graph()) as sess:
    #        tf.saved_model.loader.load(
    #            sess, tags=["train"], export_dir=model_path)
    # except:

    model = L1(args, x_placeholder)
    model.compile()

    print('Compiled Model')
    model.print_model()
    # y_placeholder_dict = {}
    # for op_key, op_val in model._output_layers.items():
    #    y_placeholder_dict[op_key] = tf.placeholder(
    #        tf.float32, shape=(op_val['hp'].shape))
    y_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 10))
    # loss functions
    # High precision Cross entropy loss
    # print('saveable obj', variables._all_saveable_objects())
    # print('local vars', variables.local_variables())
    # print(' global vars', variables.global_variables())
    # print('model vars', variables.model_variables())
    # print('trainable varts', variables.trainable_variables())
    # print('moving average vars', variables.moving_average_variables())
    hp_cross_entropy_dict = model.hp_cross_entropy(y_placeholder)
    hp_cross_entropy = tf.reduce_sum(input_tensor=list(hp_cross_entropy_dict.values()))

    distillation_loss_dict = model.distillation_loss()
    distillation_loss = tf.reduce_sum(input_tensor=list(distillation_loss_dict.values()))

    bit_loss = model.bit_loss()
    regularization_loss = model.L2_regularization()

    # regularization parameter
    regularization_term = 0
    for k in model._layers_objects.keys():
        if isinstance(model._layers_objects[k]['layer_obj'],
                      gtcConv2d) or isinstance(
            model._layers_objects[k]['layer_obj'], gtcDense):
            regularization_term = regularization_term + tf.nn.l2_loss(
                model._layers_objects[k]['layer_obj'].hp_weights)
            # print('---REGULARIZATION', regularization_term)

    lambda_distillation_loss = tf.constant(args.lambda_distillation_loss)
    lambda_bit_loss = tf.constant(args.lambda_bit_loss)
    lambda_regularization = tf.constant(args.lambda_regularization)

    # Total loss
    losses = {}

    lambda_vars = {}
    lambda_vars['lambda_bit_loss'] = lambda_bit_loss
    lambda_vars['lambda_distillation_loss'] = lambda_distillation_loss
    lambda_vars['lambda_regularization'] = lambda_regularization

    # optimizer
    losses['total_loss'] = hp_cross_entropy + distillation_loss * lambda_vars[
        'lambda_distillation_loss'] + bit_loss * lambda_vars[
                               'lambda_bit_loss'] + lambda_vars[
                               'lambda_regularization'] * regularization_loss

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
        losses['total_loss'])

    metrices = {}

    model.lp_accuracy(y_placeholder)
    model.hp_accuracy(y_placeholder)
    model.compile_training_info(
        loss=losses, optimizer=optimizer, metrices=metrices)

    # print('check inputs to network')
    # [print(k, v) for k, v in model._layers_strings.items()]
    # print('check forward inputs')
    # [print(k, v) for k, v in model._forward_prop_inputs.items()]
    # print('check layer objects')
    # [print(k, v) for k, v in model._layers_objects.items()]

    # summary
    summary_ops = model.get_summary()
    for k, v in summary_ops.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                # print(k, k1, v1, '\n')
                tf.compat.v1.summary.scalar('_'.join([k, k1]), v1)
        else:
            # print(k, v, '\n')
            tf.compat.v1.summary.scalar('_'.join(k), v)
    # print('SUMMARY')
    # [print(k, v) for k, v in summary_ops.items()]

    validation_log_path = _BASEDIR + '/' + args.name_of_experiment + '/' + 'logs/' + 'validation'
    training_log_path = _BASEDIR + '/' + args.name_of_experiment + '/' + 'logs/' + 'train'
    model_path = _BASEDIR + '/' + args.name_of_experiment + '/' + 'model/' + 'model.ckpt'
    lp_model_path = _BASEDIR + '/' + args.name_of_experiment + '/' + 'model/'
    init = tf.compat.v1.global_variables_initializer()
    # saving lp  model

    merge = tf.compat.v1.summary.merge_all()
    var_list = variables._all_saveable_objects(
    ) + model._layers_objects['conv2d']['layer_obj']._save_params()
    # print('variable list', var_list)
    # saver = tf.train.Saver(var_list=var_list)
    sys.stdout.flush()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        train_writer = tf.compat.v1.summary.FileWriter(training_log_path, sess.graph)
        validation_writer = tf.compat.v1.summary.FileWriter(validation_log_path,
                                                            sess.graph)
        test_batches = 1
        batches = 1
        # igraphdef = tf.saved_model.loader.load(
        #        sess, tags=["train"], export_dir=model_path)
        for e in range(args.num_epochs):

            test_average = {}
            test_average['lp_accuracy', 'flatten_1'] = []
            test_average['hp_accuracy', 'flatten_1'] = []
            test_average['total_loss', 'total_loss'] = []
            test_average['bit_loss'] = []
            test_average['distillation_loss', 'flatten_1'] = []
            test_average['hp_cross_entropy', 'flatten_1'] = []
            test_average['regularization_term'] = []

            for x_batch, y_batch in datagen.flow(
                    x_train, y_train, batch_size=args.batch_size):
                data_dict = {
                    x_placeholder: x_batch,
                    y_placeholder: y_batch,
                    learning_rate: args.learning_rate
                }
                model.train_on_batch(data_dict)

                if batches % 100 == 0:
                    print('------------ Epoch', e + 1, '/', args.num_epochs, 'batch',
                          batches, '  ---------------------------------')
                    print_summary_short(model, data_dict)
                    param_summary = merge.eval(data_dict)
                    train_writer.add_summary(param_summary, batches)

                batches = batches + 1

                if batches % 5000 == 0:
                    args.learning_rate /= 2.

                if batches / ((e + 1) * (len(x_train) / args.batch_size)) > 1:

                    sys.stdout.flush()
                    # checkpoint.save(file_prefix=checkpoint_prefix)
                    # saver.save(sess, model_path)
                    print('Saving the Model')
                    model._save_model(
                        path=lp_model_path + 'training/', int_model=False)
                    for x_batch, y_batch in datagen.flow(
                            x_test, y_test, batch_size=args.batch_size):
                        data_dict = {
                            x_placeholder: x_batch,
                            y_placeholder: y_batch
                        }
                        param_summary = merge.eval(data_dict)
                        validation_writer.add_summary(param_summary,
                                                      test_batches)
                        test_batches = test_batches + 1
                        for k1 in test_average.keys():
                            if isinstance(k1, tuple):
                                test_average[k1].append(
                                    summary_ops[k1[0]][k1[1]].eval(data_dict))
                            else:
                                test_average[k1].append(
                                    summary_ops[k1].eval(data_dict))

                        if test_batches / (
                                (e + 1) * (len(x_test) / args.batch_size)) > 1:
                            print('*' * 20)
                            print("Test results after epoches", e)
                            sys.stdout.flush()
                            [
                                print(k,
                                      sum(v) / len(v))
                                for k, v in test_average.items()
                            ]

                            break
                    break
        if args.print_layers_bits:
            print('----------------- Number of bits used per each layer ------------------------')
            bits_per_layer = model._bits_per_layer()
            for k, lb in bits_per_layer.items():
                if ('conv' in k) or ('dense' in k):
                    print('k = ', k, '  lb =  ', lb.eval())
        path_expt = os.path.join(_BASEDIR, args.name_of_experiment + '/training_model_final/')
        if os.path.exists(path_expt):
            os.rmdir(path_expt)
        os.makedirs(path_expt)

        model.save(path=path_expt, int_model=False)
        with open(os.path.join(path_expt, 'model_schema.json'), 'w') as schema_f:
            schema_f.write(json.dumps({
                "dataset": "custom",
                "input_names": "Placeholder",
                "input_shapes": "1,28,28,1",
                "output_names": "Softmax",
                "preprocessor": "rgbtogray",
                "task": "classifier"
            }, indent=4))
        int_path_expt = os.path.join(_BASEDIR, args.name_of_experiment + '/int_model_final/')
        if os.path.exists(int_path_expt):
            os.rmdir(int_path_expt)
        os.makedirs(int_path_expt)

        model.save(path=int_path_expt, int_model=True)
        with open(os.path.join(int_path_expt, 'model_schema.json'), 'w') as schema_f:
            schema_f.write(json.dumps({
                "dataset": "custom",
                "input_names": "Placeholder",
                "input_shapes": "1,28,28,1",
                "output_names": "Softmax",
                "preprocessor": "rgbtogray",
                "task": "classifier"
            }, indent=4))

if __name__ == "__main__":
    main()
