#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras

tf.keras.backend.set_learning_phase(0)

model = keras.models.load_model('trained_model.h5')

sess = tf.compat.v1.keras.backend.get_session()
writer = tf.compat.v1.summary.FileWriter("/tmp/tmp_tensorboard", sess.graph)
#print(sess.run(h))
writer.close()

subprocess.check_call(['tensorboard', '--bind_all', '--logdir=/tmp/tmp_tensorboard'])
