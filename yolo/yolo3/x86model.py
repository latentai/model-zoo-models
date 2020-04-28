#  Copyright (c) 2019 by LatentAI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "LatentAI Commercial Software License".
#  Please see the LICENSE file that should have been included as part of
#  this package.
#
# @file    x86_inference.py
#
# @author  Abelardo López Lagunas
#
# @date    Thu Aug 22 08:55 2019
#
# @brief   Load binary and run inference on an x86
#
# References:
#          Shows how to load and run a binary
#
# Restrictions:
#          There is no attempt to check input parameters
#
# Revision history:
#
#          Thu Aug 22 08:55 2019 -- File created
#          Thu Sep 19 12:42 2019 -- Fixed output path to binaries
#          Mon Sep 30 11:45 2019 -- Added path into latent-ai sdk
#          Thu Oct 03 16:16 2019 -- Fixed formatting & refactoring
#          Wed Oct 09 16:19 2019 -- Fixed reference to the model
#
# @note    Intended as support for the LatentAI demo day
#
import os
import tvm
from tvm.contrib import graph_runtime

import numpy as np
from PIL import Image
# from matplotlib import pyplot as plt
from keras.applications.resnet50 import preprocess_input
import json

class Model:
    """
    The model which imitates keras's model behavior.
    The model can be used to do predictions and evaluations in YOLO ecosystem.
    """
    def __init__(self):
        self._loaded_graph = None
        self._loaded_lib = None
        self._loaded_params = None

        self._input_name = 'input_1'

    def predict_on_batch(self, x):
        """
        Runs session on input data and returns numpy array
        """
        # Prepocessing is being done outside of the predict method
        # TODO: type casting should be done, maybe
        #data = np.array(image)[np.newaxis, :].astype(dtype)

        dtype = "float32"
        x = x.astype(dtype)
        print('Input type\shape:', x.dtype, x.shape)

        self._model.set_input(self._input_name, tvm.nd.array(x))

        # The actuall inference?
        ftimer = self._model.module.time_evaluator("run", self._ctx, number=1, repeat=1)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

        output = [self._model.get_output(i).asnumpy() for i in range(3)]

        print('outputs sum:', [np.sum(x) for x in output])

        return output

    def load(self, path):
        """
        Restore tensorflow checkpoint

        param path: path to the directory where the binary files are located.
        """

        #
        # Load the previously generated model, binary and parameters. Note that
        # the directory paths are hard coded.
        #
        base = path
        self._loaded_graph = open(os.path.join(base, "modelDescription.json")).read()
        self._loaded_lib = tvm.runtime.load_module(os.path.join(base, "modelLibrary.so"))
        self._loaded_params = bytearray(open(os.path.join(base, "modelParams.params"), "rb").read())
        #self._ctx = tvm.cpu(0)
        self._ctx = tvm.gpu(1)
        #
        # Get rid of the leip key
        #
        graphjson = json.loads(self._loaded_graph)
        if 'leip' in list(graphjson.keys()):
            del graphjson['leip']
            self._loaded_graph = json.dumps(graphjson)

        self._model = graph_runtime.create(self._loaded_graph, self._loaded_lib, self._ctx)
        self._model.load_params(self._loaded_params)


