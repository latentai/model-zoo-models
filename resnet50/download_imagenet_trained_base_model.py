#!/usr/bin/env python3
import subprocess

import dload

import os

os.makedirs("./weights/", exist_ok=True)

dload.save("http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW.tar.gz", "./weights/resnet_v2_fp32_savedmodel_NCHW.tar.gz")
print('Downloaded!')

subprocess.check_call(['tar', 'xzf', "resnet_v2_fp32_savedmodel_NCHW.tar.gz"], cwd="./weights/")
print('Extracted!')



