import argparse

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
  help="""\
  Where to download the speech training data to.
  """)
parser.add_argument(
  '--background_volume',
  type=float,
  default=0.1,
  help="""\
  How loud the background noise should be, between 0 and 1.
  """)
parser.add_argument(
  '--background_frequency',
  type=float,
  default=0.8,
  help="""\
  How many of the training samples have background noise mixed in.
  """)
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
  '--time_shift_ms',
  type=float,
  default=100.0,
  help="""\
  Range to randomly shift the training audio by in time.
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
  default='15000,3000',
  help='How many training loops to run',)
parser.add_argument(
  '--train_steps',
  type=int,
  default=4000,
  help='How many steps to train the model.',)
parser.add_argument(
  '--eval_step_interval',
  type=int,
  default=1000,
  help='How often to evaluate the training results.')
parser.add_argument(
  '--learning_rate',
  type=str,
  default='0.001,0.0001',
  help='How large a learning rate to use when training.')
parser.add_argument(
  '--lambda_bit_loss',
  type=float,
  default=0.0001,
  help='How lambda bit loss impact training.')
parser.add_argument(
  '--lambda_distillation_loss',
  type=float,
  default=0.01,
  help='How lambda distillation loss impact training.')
parser.add_argument(
  '--batch_size',
  type=int,
  default=2048,
  help='How many items to train with at once',)
parser.add_argument(
  '--wanted_words',
  type=str,
  default='up,down,left,right,one,two,three,four,five,six,seven,eight,nine,zero,go,stop,cat,dog,bird,bed,wow,sheila,happy,house,marvin,yes,no,off,on,tree',
  help='Words to use (others will be added to an unknown label)',)
parser.add_argument(
  '--save_step_interval',
  type=int,
  default=100,
  help='Save Model checkpoint every save_steps.')
parser.add_argument(
  '--start_checkpoint',
  type=str,
  default='',
  help='If specified, restore this pretrained Model before any training.')
parser.add_argument(
  '--model_architecture',
  type=str,
  default='conv',
  help='What Model architecture to use')
parser.add_argument(
  '--check_nans',
  type=bool,
  default=False,
  help='Whether to check for invalid numbers during processing')
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
  '--tf_checkpoint',
  type=str,
  help="Path to the pretrained tensorflow checkpoint.")
parser.add_argument(
  '--exp_dir',
  type=str,
  help="Custom path to experiment directory. If argument is omited, 'train_data' directory will be created.")

FLAGS, unparsed = parser.parse_known_args()
