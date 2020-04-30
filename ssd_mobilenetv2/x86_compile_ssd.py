#  Copyright (c) 2019 by LatentAI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "LatentAI Commercial Software License".
#  Please see the LICENSE file that should have been included as part of
#  this package.
#
# @file    x86_compile.py
#
# @author  Abelardo López Lagunas
#
# @date    Thu Aug 22 08:55 2019
#
# @brief   Example of MobileNet_v2 compilation for an x86
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
#          Wed Oct 09 12:12 2019 -- Fixed reference to the model
#
# @note    Intended as support for the LatentAI demo
#
import os
from tvm import relay
import tensorflow as tf
import logging
#
# Use latentai sdk
#
from leip.compress.quantizer import Importers
#
# Set the logger
#
logging.basicConfig(level=os.environ.get("LOGLEVEL", "ERROR"))

# log = logging.getLogger(__name__)
#
# The example uses a hard-coded path based on the environment variable
# SAMPLE_MODEL_BASE_DIR, make sure it points to the correct place.


base = "./"
model_dir = base + "saved_models/tf/"
output_name = ["predictions/concat"]
# model_dir = "/Users/alopez/Documents/Code/LatentAI/my_models/mobilenet_v1"
# model_dir = "/Users/alopez/Downloads/mobilenet_v2_1.0_224"

# model = Importers.ImportMeta(dir_path=model_dir, quantizer="", call_cmd=None)
model = Importers.ImportKeras(dir_path=model_dir, quantizer="", output_names=output_name)
gd = model.graph.as_graph_def()
#
# Model parameters
#
# When running from output compress the name of the input node is called
# Placeholder or input. Choose the right input_name below.
#
# input_name = "Placeholder"
input_name = "input"
dshape = (1, 224, 224, 3)
shape_dict = {input_name: dshape}
#
# Cleanup the graph by converting variables to constants
#
gdn = tf.compat.v1.graph_util.convert_variables_to_constants(
    model.sess,
    gd,
    output_name
)

print("\nGraph generated. Start compilation process\n")
#
# Target settings for x86
#
# myTarget = "llvm -mcpu=skylake"
myTarget = "llvm"
layout = "NCHW"
# layout = "NHWC"


#
# Ingest model into compiler then build. Note that Graph optimization
# should be done before building the model.
#
def build(target):
    print("\nBuilding code\n")

    sym, params = relay.frontend.from_tensorflow(gdn,
                                                 layout=layout,
                                                 shape={input_name: dshape},
                                                 outputs=output_name)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(sym, target, params=params)
        #
        # Export the model as a library, this is hardcoded for now
        #
        print("\nExporting the model\n")
        base = "./compiled_tvm_int8/bin/"
        lib.export_library(base + "modelLibrary.so")
        with open(base + "modelDescription.json", "w") as fo:
            fo.write(graph)
        with open(base + "modelParams.params", "wb") as fo:
            fo.write(relay.save_param_dict(params))

    return graph, lib, params


#
# Compile the model
#
graph, lib, params = build(myTarget)
print("Model created for the x86:", myTarget)
