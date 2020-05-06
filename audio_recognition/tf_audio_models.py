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
"""
Model definitions for simple speech recognition.
"""

# TODO: we should leave only single conv architecture. All other can go.
# TODO: this file is still needs clean up

import math

import tensorflow.compat.v1 as tf


def _next_power_of_two(x):
  """Calculates the smallest enclosing power of two for an input.

  Args:
    x: Positive float or integer number.

  Returns:
    Next largest power of two integer.
  """
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, feature_bin_count,
                           preprocess):
  """Calculates common settings needed for all rough-models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    feature_bin_count: Number of frequency bins to use for analysis.
    preprocess: How the spectrogram is processed to produce features.

  Returns:
    Dictionary containing common settings.

  Raises:
    ValueError: If the Preprocessing mode isn't recognized.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  if preprocess == 'average':
    fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
    average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
    fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
  elif preprocess == 'mfcc':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  elif preprocess == 'micro':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  else:
    raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                     ' "average", or "micro")' % (preprocess))
  fingerprint_size = fingerprint_width * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'fingerprint_width': fingerprint_width,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'preprocess': preprocess,
      'average_window_width': average_window_width,
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  """Builds a Model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of rough-models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible Model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the Model.
    model_architecture: String specifying which kind of Model to create.
    is_training: Whether the Model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, for now only "conv"' +
                    ' is supported.')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional Model.

  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the Model.
    is_training: Whether the Model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  dropout_prob = 0 
  if is_training:
    dropout_prob = 0.5
  
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.get_variable(
      name='first_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.get_variable(
      name='first_bias',
      initializer=tf.zeros_initializer,
      shape=[first_filter_count])
  first_conv = tf.nn.conv2d(input=fingerprint_4d,
                            filter=first_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
  first_conv = tf.math.add(first_conv, first_bias)
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(value=first_dropout,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.get_variable(
      name='second_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[
          second_filter_height, second_filter_width, first_filter_count,
          second_filter_count
      ])
  second_bias = tf.get_variable(
      name='second_bias',
      initializer=tf.zeros_initializer,
      shape=[second_filter_count])
  second_conv = tf.nn.conv2d(input=max_pool,
                             filter=second_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
  second_conv = tf.math.add(second_conv, second_bias)
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(
      name='final_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[second_conv_element_count, label_count])
  final_fc_bias = tf.get_variable(
      name='final_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights)
  final_fc = tf.math.add(final_fc, final_fc_bias)
  return final_fc

