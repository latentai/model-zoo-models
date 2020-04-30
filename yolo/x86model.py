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
#
# The example uses a hard-coded path based on the environment variable
# SAMPLE_MODEL_BASE_DIR, make sure it points to the correct place.
#
base = "/Users/alopez/Documents/Code/LatentAI/support/"
# base = os.environ["SAMPLE_MODEL_BASE_DIR"] + "/"
image_dir = base + "resources/images/imagenet_images/raw/"
name_dir = base + "resources/data/imagenet/"
#
# Load the test image, match the expected input shape of 224,224,3
#
img_name = image_dir + "Wiggums.JPG"
# img_name = image_dir + "IMG_9448.JPG"
# img_name = image_dir + "elephant-299.jpg"
# img_name = image_dir + "cat.png"
synset_name = name_dir + "imagenet1001.names"

with open(synset_name) as f:
    synset = f.readlines()

dtype = "float32"
image = Image.open(img_name).resize((224, 224))
data = np.array(image)[np.newaxis, :].astype(dtype)
data = preprocess_input(data[:, :, :, ::-1], mode="tf")
#
# Uncomment this to see the image, also uncomment matplotlib import above
#
# plt.imshow(image)
# plt.show()
#
# Model parameters & target
#
# When running from output compress the name of the input node is called
# Placeholder or input. Choose the right input_name below.
#
# input_name = "Placeholder"
input_name = "input"
# input_name = "data"
shape_dict = {input_name: data.shape}
ctx = tvm.cpu(0)
#
# Load the previously generated model, binary and parameters. Note that
# the directory paths are hard coded.
#
base = "./bin/"
loaded_graph = open(base + "modelDescription.json").read()
loaded_lib = tvm.runtime.load_module(base + "modelLibrary.so")
loaded_params = bytearray(open(base + "modelParams.params", "rb").read())
#
# Get rid of the leip key
#
graphjson = json.loads(loaded_graph)
if 'leip' in list(graphjson.keys()):
    del graphjson['leip']
    loaded_graph = json.dumps(graphjson)


#
# Create runtime and do inference. This could be in another file
# if the modelLibrary, modelParams and modelDescription are loaded
#
def run(graph, lib, params, ctx):
    #
    # Build runtime
    #
    print("\nBuilding runtime\n")
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(input_name, tvm.nd.array(data))
    m.load_params(loaded_params)
    print("\nDone building runtime\n")
    #
    # Execute runtime
    #
    print("\nEvaluate inference time cost\n")
    ftimer = m.module.time_evaluator("run", ctx, number=50, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))
    #
    # Get output
    #
    tvm_output = m.get_output(0)
    top1 = np.argmax(tvm_output.asnumpy()[0])

    output = tvm_output.asnumpy()[0]
    items = 5
    names = synset

    cn = np.argsort(output)
    cn = cn[-items:]
    results = []
    print("Top {0} classifications: ".format(items))
    for i in reversed(cn):
        results.append({
                "class": names[i],
                "confidence": round(float(output[i]), 4)
            })
        print("  ", names[i], round(output[i], 4))

    return top1


#
# Here is were the inference starts
#
top1 = run(loaded_graph,
           loaded_lib,
           loaded_params,
           ctx)

print("\nPrediction id {}, name: {}\n".format(top1,
                                              synset[top1]))
