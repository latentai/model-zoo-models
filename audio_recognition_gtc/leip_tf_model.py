"""
This script contains example of google audio model
prepared with LEIP framework.
"""
import math

import tensorflow.compat.v1 as tf
from leip.compress.training.gtcModel import GTCModel
from leip.compress.training.quantization import GTCQuantization
from leip.compress.training.gtc.conv2d import Conv2d as gtcConv2d
from leip.compress.training.gtc.activation import Activation as gtcActivation
from leip.compress.training.quantization import IdentityQuantization
from leip.compress.training.gtc.pooling import MaxPooling2D as gtcMaxPooling2D
from leip.compress.training.gtc.flatten import Flatten as gtcFlatten
from leip.compress.training.gtc.dense import Dense as gtcDense
from leip.compress.training.gtc.dropout import Dropout as gtcDropout

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



def create_leip_gmodel(fingerprint_input, model_settings, dropout=0.25):
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
    tf.compat.v1.disable_eager_execution()

    model = GTCModel()

    input_frequency_size = model_settings['fingerprint_width']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.compat.v1.reshape(
        fingerprint_input, [-1, input_time_size, input_frequency_size, 1])
    print(fingerprint_4d)
    print('input_time_size, input_frequency_size',
          input_time_size, input_frequency_size)

    model.add(fingerprint_4d)

    first_filter_width = 8
    first_filter_count = 64

    model.add(
        gtcConv2d(
            filters=first_filter_count,
            kernel_size=first_filter_width,
            quantizer=GTCQuantization(),
            strides=1,
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.compat.v1.constant_initializer(0.0),
            input_shape=[input_time_size, input_frequency_size],
            use_bias=True,
            name='Conv2D')
    )

    model.add(
        gtcActivation(
            quantizer=IdentityQuantization(),
            activation='relu',
            trainable=False))

    # TODO: this is debug option, will be removed later
    use_dropout = False

    if use_dropout:
        model.add(gtcDropout(rate=dropout))

    model.add(
        gtcMaxPooling2D(
            pool_size=2,
            strides=2,
            quantizer=IdentityQuantization(),
            padding='SAME'
        ))

    second_filter_width = 4
    second_filter_count = 64

    model.add(
        gtcConv2d(
            filters=second_filter_count,
            kernel_size=second_filter_width,
            quantizer=GTCQuantization(),
            strides=1,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.compat.v1.zeros_initializer,
            input_shape=[input_time_size, input_frequency_size],
            use_bias=True,
            name='Conv2D_1')
    )

    model.add(
        gtcActivation(
            quantizer=IdentityQuantization(),
            activation='relu',
            trainable=False))

    if use_dropout:
        model.add(gtcDropout(rate=dropout))

    model.add(gtcFlatten())
    label_count = model_settings['label_count']

    model.add(
        gtcDense(
            units=label_count,
            quantizer=GTCQuantization(),
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.initializers.glorot_normal(),
            use_bias=True)
    )

    return model
