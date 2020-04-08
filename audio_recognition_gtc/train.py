import os
import sys
import yaml
import logging

from collections import namedtuple

import tensorflow.compat.v1 as tf
import numpy as np

from tf_flags import FLAGS, unparsed
from leip_tf_model import prepare_model_settings, create_leip_gmodel

import input_data

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

from leip.compress.training.utility_scripts.weight_readers import CheckpointReader

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


def main(_):

    session_conf = tf.compat.v1.ConfigProto(
      intra_op_parallelism_threads=4,
      inter_op_parallelism_threads=4)
    sess = tf.compat.v1.InteractiveSession(config=session_conf)

    model_settings = prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess)

    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir,
        FLAGS.silence_percentage, FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings, summaries_dir=None)

    input_frequency_size = model_settings['fingerprint_width']
    input_time_size = model_settings['spectrogram_length']

    # deal with pathes and directories
    # --------------------------------
    exp_dir = 'train_data'

    if FLAGS.exp_dir is not None:
        exp_dir = FLAGS.exp_dir

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    summaries_dir = os.path.join(exp_dir, 'summaries')
    if not os.path.exists(summaries_dir):
        os.mkdir(summaries_dir)

    validation_log_path = os.path.join(summaries_dir, 'validation')
    if not os.path.exists(validation_log_path):
        os.mkdir(validation_log_path)

    training_log_path = os.path.join(summaries_dir, 'train')
    if not os.path.exists(training_log_path):
        os.mkdir(training_log_path)

    lp_export_model_path = os.path.join(exp_dir, 'model')
    if not os.path.exists(lp_export_model_path):
        os.mkdir(lp_export_model_path)
    # --------------------------------

    # configuration
    leip_args = {
        'batch_size': FLAGS.batch_size,
        'num_epochs': 1,
        'learning_rate': 0.0001,
        'num_channels': 1,
        'lambda_bit_loss': FLAGS.lambda_bit_loss,
        'lambda_distillation_loss': FLAGS.lambda_distillation_loss
    }

    print('-' * 10)
    print(leip_args)
    print('-' * 10)

    with open(os.path.join(exp_dir, 'leip_args.yaml'), 'w') as fp:
        yaml.dump(leip_args, fp)

    leip_args = namedtuple('LeipArgs', leip_args.keys())(*leip_args.values())

    fingerprint_size = model_settings['fingerprint_size']
    print('fingerprint size:', fingerprint_size)

    input_placeholder = tf.compat.v1.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    fingerprint_4d = tf.compat.v1.reshape(input_placeholder,
                                [-1, input_time_size, input_frequency_size, 1])
    ground_truth_input = tf.compat.v1.placeholder(
        tf.int64, [None, model_settings['label_count']], name='groundtruth_input')
    dropout = tf.compat.v1.placeholder(
        tf.float32,
        name="dropout"
    )
    model = create_leip_gmodel(fingerprint_4d, model_settings)
    model.compile()


    print('Compiled Model')
    model.print_model()

    hp_cross_entropy_dict = model.hp_cross_entropy(ground_truth_input)
    hp_cross_entropy = tf.reduce_sum(list(hp_cross_entropy_dict.values()))
    distillation_loss_dict = model.distillation_loss()
    distillation_loss = tf.reduce_sum(list(distillation_loss_dict.values()))
    bit_loss = model.bit_loss()

    logits = model._forward_prop_outputs['dense']['hp']
    predicted_indices = tf.argmax(input=logits, axis=1)
    labels = tf.argmax(ground_truth_input, axis=0)
    correct_prediction = tf.equal(predicted_indices, labels)

    evaluation_step = tf.reduce_mean(input_tensor=tf.cast(correct_prediction,
                                                          tf.float32))
    with tf.compat.v1.get_default_graph().name_scope('eval'):
        tf.compat.v1.summary.scalar('cross_entropy', hp_cross_entropy)
        tf.compat.v1.summary.scalar('accuracy', evaluation_step)

    lambda_distillation_loss = tf.constant(leip_args.lambda_distillation_loss)
    lambda_bit_loss = tf.constant(leip_args.lambda_bit_loss)

    # Total loss
    losses = {}

    lambda_vars = {}
    lambda_vars['lambda_bit_loss'] = lambda_bit_loss
    lambda_vars['lambda_distillation_loss'] = lambda_distillation_loss
    lambda_vars['lambda_regularization'] = .01

    # optimizer
    losses['total_loss'] = hp_cross_entropy + distillation_loss * lambda_vars[
        'lambda_distillation_loss'] + bit_loss * lambda_vars['lambda_bit_loss']

    learning_rate = tf.compat.v1.placeholder(tf.float32, name='lr')
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
        losses['total_loss'])
    metrices = {}

    model.lp_accuracy(ground_truth_input)
    model.hp_accuracy(ground_truth_input)
    model.compile_training_info(
        loss=losses, optimizer=optimizer, metrices=metrices)

    print('SUMMARY')
    with tf.compat.v1.get_default_graph().name_scope('gtc'):
        summary_ops = model.get_summary()
        for k, v in summary_ops.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    tf.summary.scalar('_'.join([k, k1]), v1)
            else:
                tf.summary.scalar('_'.join(k), v)

    init = tf.compat.v1.global_variables_initializer()
    merge_gtc_summary = tf.compat.v1.summary.merge_all(scope="gtc")

    sess.run(init)
    train_writer = tf.compat.v1.summary.FileWriter(
        training_log_path, sess.graph)
    validation_writer = tf.compat.v1.summary.FileWriter(
        validation_log_path, sess.graph)

    test_batches = 0
    batches = 1
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

    test_average = {}
    test_average['lp_accuracy', 'dense'] = []
    test_average['hp_accuracy', 'dense'] = []
    test_average['total_loss', 'total_loss'] = []
    test_average['bit_loss'] = []
    test_average['distillation_loss', 'dense'] = []
    test_average['hp_cross_entropy', 'dense'] = []

    if FLAGS.tf_checkpoint is not None:
        print('Loading from checkpoint: {}'.format(FLAGS.tf_checkpoint))
        model_reader = CheckpointReader(FLAGS.tf_checkpoint)
        model.load_pretrained_weights_from_reader(model_reader)
        print('Checkpoint restored.')

    for e in range(FLAGS.train_steps):
        print('Train step: {}'.format(e+1))
        sys.stdout.flush()

        x_batch, y_train = audio_processor.get_data(
            leip_args.batch_size, 0, model_settings, FLAGS.background_frequency,
            FLAGS.background_volume, time_shift_samples, 'training', sess)
        y_train = y_train.astype(int)
        y_batch = np.zeros(
            [leip_args.batch_size, model_settings['label_count']], dtype=int)
        y_batch[np.arange(leip_args.batch_size), y_train] = 1

        data_dict = {
            input_placeholder: x_batch,
            ground_truth_input: y_batch,
            learning_rate: leip_args.learning_rate,
            dropout: 0.2
        }
        model.train_on_batch(data_dict)

        if batches % 100 == 0:
            print('Step', e + 1, '/', leip_args.num_epochs, 'checkup',
                  batches, model.print_summary(data_dict))
            if merge_gtc_summary is not None:
                param_summary = merge_gtc_summary.eval(data_dict)
                train_writer.add_summary(param_summary, batches)
            else:
                print('Evaluation summary has not been saved.')

        if batches % 500 == 0:
            print('Saving the Model')
            model.save(path=os.path.join(lp_export_model_path, 'training_{:0>5d}/'.format(batches)), int_model=False)
            print('Saving the integer Model')
            model.save(path=os.path.join(lp_export_model_path, 'int_model_{:0>5d}/'.format(batches)), int_model=True)

        if batches % FLAGS.eval_step_interval == 0 or batches == 1:
            set_size = audio_processor.set_size('validation')
            print("################# VALIDATION ###################")
            for i in range(0, set_size, leip_args.batch_size):
                x_valid_batch, y_valid = audio_processor.get_data(
                    leip_args.batch_size, i, model_settings, 0.0, 0.0, 0, 'validation', sess)

                this_batch_size = np.min([leip_args.batch_size, x_valid_batch.shape[0]])
                # if len(y_valid)!=leip_args.batch_size:
                #     break
                y_valid = y_valid.astype(int)
                y_valid_batch = np.zeros([this_batch_size, model_settings['label_count']], dtype=int)
                y_valid_batch[np.arange(this_batch_size), y_valid] = 1

                valid_data_dict = {
                    input_placeholder: x_valid_batch,
                    ground_truth_input: y_valid_batch,
                    dropout: 0.0
                }
                if merge_gtc_summary is not None:
                    param_summary = merge_gtc_summary.eval(valid_data_dict)
                    validation_writer.add_summary(param_summary, test_batches + 1)

                for k1 in test_average.keys():
                    if isinstance(k1, tuple):
                        test_average[k1].append(
                            summary_ops[k1[0]][k1[1]].eval(valid_data_dict))
                    else:
                        test_average[k1].append(
                            summary_ops[k1].eval(valid_data_dict))

                print('Epoch', e + 1, 'test batch', test_batches, model.print_summary(valid_data_dict))

                test_batches += 1


            print("Test results after steps:", e)
            [
                print(k,
                      sum(v) / len(v))
                for k, v in test_average.items()
            ]
            print("################# VALIDATION ###################")

        batches = batches + 1

    model.save(path=os.path.join(lp_export_model_path, 'training_final/'), int_model=False)
    model.save(path=os.path.join(lp_export_model_path, 'int_model_final/'), int_model=True)


if __name__ == "__main__":
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

