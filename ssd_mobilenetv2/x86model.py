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
    def __init__(self, **kwargs):
        self._loaded_graph = None
        self._loaded_lib = None
        self._loaded_params = None

        self._dequantize = True
        self._input_name = 'input_1'
        # if 'dequantize' in kwargs.keys():
        #     self._dequantize = kwargs['dequantize']

    def predict_on_batch(self, x):
        """
        Runs session on input data and returns numpy array
        """
        # Prepocessing is being done outside of the predict method
        # TODO: type casting should be done, maybe
        #data = np.array(image)[np.newaxis, :].astype(dtype)

        dtype = "float32"
        x = x.astype(dtype)
        # print('Input type\shape:', x.dtype, x.shape)

        self._model.set_input(self._input_name, tvm.nd.array(x))

        # The actuall inference?
        ftimer = self._model.module.time_evaluator("run", self._ctx, number=1, repeat=1)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        # print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

        # output = [self._model.get_output(i).asnumpy() for i in range(3)]
        #
        # print('outputs sum:', [np.sum(x) for x in output])

        return self._model.get_output(0)

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
        self._ctx = tvm.cpu(0)
        #
        # Get rid of the leip key
        #
        graphjson = json.loads(self._loaded_graph)
        if 'leip' in list(graphjson.keys()):
            del graphjson['leip']
            self._loaded_graph = json.dumps(graphjson)

        if self._dequantize:
            from leip.cast.Cast import Cast
            from leip import Constants
            self._loaded_params_quant = bytearray(open(os.path.join(base, "quantParams.params"), "rb").read())
            castObject = Cast(self._loaded_params, 'ASYMMETRIC', 8, Constants.DATA_INT8)
            self._loaded_params = castObject.dequantize(self._loaded_params_quant)

        self._model = graph_runtime.create(self._loaded_graph, self._loaded_lib, self._ctx)
        self._model.load_params(self._loaded_params)
