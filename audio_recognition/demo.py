# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os.path
import sys

import numpy as np
import tensorflow.compat.v1 as tf

import input_data
import tf_audio_models as models


FLAGS = None


def main(_):
    # Set the verbosity based on flags (default is INFO, so we see all messages)
    tf.logging.set_verbosity(FLAGS.verbosity)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess)
    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir,
        FLAGS.silence_percentage, FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings, None)

    wav_file = FLAGS.wav

    fingerprint_size = model_settings['fingerprint_size']

    # Figure out the learning rates for each training phase. Since it's often
    # effective to have high learning rates at the start of training, followed by
    # lower levels towards the end, the number of steps and learning rates can be
    # specified as comma-separated lists to define the rate at each stage. For
    # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
    # will run 13,000 training loops in total, with a rate of 0.001 for the first
    # 10,000, and 0.0001 for the final 3,000.
    training_steps_list = list(
        map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                       len(learning_rates_list)))
    input_placeholder = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')
    if FLAGS.quantize:
        fingerprint_min, fingerprint_max = input_data.get_features_range(
            model_settings)
        fingerprint_input = tf.quantization.fake_quant_with_min_max_args(
            input_placeholder, fingerprint_min, fingerprint_max)
    else:
        fingerprint_input = input_placeholder

    print('fingerprint input:', fingerprint_input)

    logits = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        is_training=False,
    )

    # Define loss and optimizer
    ground_truth_input = tf.placeholder(
        tf.int64, [None], name='groundtruth_input')

    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits)
    if FLAGS.quantize:
        tf.contrib.quantize.create_training_graph(quant_delay=0)
    predicted_indices = tf.argmax(input=logits, axis=1)
    correct_prediction = tf.equal(predicted_indices, ground_truth_input)
    evaluation_step = tf.reduce_mean(input_tensor=tf.cast(correct_prediction,
                                                          tf.float32))
    with tf.get_default_graph().name_scope('eval'):
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        tf.summary.scalar('accuracy', evaluation_step)

    global_step = tf.train.get_or_create_global_step()

    tf.global_variables_initializer().run()

    start_step = 1

    if FLAGS.checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)
        start_step = global_step.eval(session=sess)
        tf.logging.info(
            'Checkpoint: {}'.format(FLAGS.checkpoint))

    tf.logging.info('Recovering checkpoint from step: {}'.format(start_step))


    input_features = audio_processor.get_features_for_wav(wav_file, model_settings, sess)
    print('features:', input_features)
    print('features:', len(input_features))
    input_features = input_features[0]
    print('features:', input_features.shape)
    input_features = np.expand_dims(input_features.flatten(), 0)

    y_pred = sess.run(
                predicted_indices,
                feed_dict={
                    fingerprint_input: input_features
                })

    print('Predict:', y_pred)
    print('Label:', audio_processor.words_list[y_pred[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        help="Place where speech commands dataset is located")
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be silence.
      """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be unknown words.
      """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectogram timeslices.',)
    parser.add_argument(
        '--feature_bin_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint',
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default='15000,20000',
        help='How many training loops to run',)
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.001,0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once',)
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='up,down,left,right,one,two,three,four,five,six,seven,eight,nine,zero,go,stop,cat,dog,bird,bed,wow,sheila,happy,house,marvin,yes,no,off,on,tree',
        help='Words to use (others will be added to an unknown label)',)
    parser.add_argument(
        '--train_dir',
        type=str,
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=100,
        help='Save Model checkpoint every save_steps.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='If specified, restore this pretrained Model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='conv',
        help='What Model architecture to use')
    parser.add_argument(
        '--quantize',
        type=bool,
        default=False,
        help='Whether to train the Model for eight-bit deployment')
    parser.add_argument(
        '--preprocess',
        type=str,
        default='mfcc',
        help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"')
    parser.add_argument(
        '--wav',
        type=str,
        default='dataset/cat/030ec18b_nohash_1.wav',
        help='Path to WAV file to make a prediction')

    # Function used to parse --verbosity argument
    def verbosity_arg(value):
        """Parses verbosity argument.

        Args:
          value: A member of tf.logging.
        Raises:
          ArgumentTypeError: Not an expected value.
        """
        value = value.upper()
        if value == 'INFO':
            return tf.logging.INFO
        elif value == 'DEBUG':
            return tf.logging.DEBUG
        elif value == 'ERROR':
            return tf.logging.ERROR
        elif value == 'FATAL':
            return tf.logging.FATAL
        elif value == 'WARN':
            return tf.logging.WARN
        else:
            raise argparse.ArgumentTypeError('Not an expected value')
    parser.add_argument(
        '--verbosity',
        type=verbosity_arg,
        default=tf.logging.INFO,
        help='Log verbosity. Can be "INFO", "DEBUG", "ERROR", "FATAL", or "WARN"')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
