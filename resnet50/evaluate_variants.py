#!/usr/bin/env python3
import json
import os
import subprocess

input_checkpoint = 'imagenet_checkpoint/'
dataset_index_file = "/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt"
class_names_file = '/data/sample-models/resources/data/imagenet/imagenet1000.names'
preprocessor = 'imagenet_caffe'
input_names = 'input_1'
output_names = 'probs/Softmax'
input_shapes = '1,224,224,3'

dry_run=True


commands_run = []

def logCmd(args):
    commands_run.append(args)
    if not dry_run:
        subprocess.check_call(args)

def getResults(dirname, variant_name):
    resultPath = os.path.join(dirname, 'results.json')
    data = json.loads(open(resultPath, 'r').read())
    print(data)

logCmd(["rm", "-rf", "variants"])
logCmd(["mkdir", "variants"])
logCmd(["mkdir", "baselineFp32Results"])

#", "Baseline", "TF", "FP32
logCmd(["leip", "evaluate", "--output_path","baselineFp32Results", "--framework", "tf2", "--input_path", input_checkpoint, "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor, "--input_shapes", input_shapes, "--input_names", input_names, "--output_names", output_names])
getResults("baselineFp32Results/", "Baseline TF FP32")
#", "LEIP", "Compress

logCmd(["leip", "compress", "--input_path", input_checkpoint, "--quantizer", "ASYMMETRIC", "--bits", "8", "--output_path", "variants/checkpointCompressed/"])
logCmd(["leip", "compress", "--input_path", input_checkpoint, "--quantizer", "POWER_OF_TWO", "--bits", "8", "--output_path", "variants/checkpointCompressedPow2/"])

#", "LEIP", "TF", "FP32
logCmd(["leip", "evaluate", "--output_path","variants/checkpointCompressed/", "--framework", "tf2", "--input_path", "variants/checkpointCompressed/model_save/", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor, "--input_shapes", input_shapes, "--input_names", input_names, "--output_names", output_names])
getResults("variants/checkpointCompressed/", "LEIP TF FP32")
#", "Baseline", "TVM", "INT8
logCmd(["rm", "-rf", "variants/compiled_tvm_int8"])
logCmd(["mkdir", "variants/compiled_tvm_int8"])
logCmd(["leip", "compile", "--input_path", input_checkpoint, "--input_shapes", input_shapes, "--output_path", "variants/compiled_tvm_int8/bin", "--input_types=uint8", "--data_type=int8"])
logCmd(["leip", "evaluate", "--output_path","variants/compiled_tvm_int8/","--framework", "tvm", "--input_names", input_names, "--input_types=uint8", "--input_shapes", input_shapes, "--input_path", "variants/compiled_tvm_int8/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
getResults("variants/compiled_tvm_int8/", "Baseline TVM INT8")

#", "Baseline", "TVM", "FP32
logCmd(["rm", "-rf", "variants/compiled_tvm_fp32"])
logCmd(["mkdir", "variants/compiled_tvm_fp32"])
logCmd(["leip", "compile", "--input_path", input_checkpoint, "--input_shapes", input_shapes, "--output_path", "variants/compiled_tvm_fp32/bin", "--input_types=float32", "--data_type=float32"])
logCmd(["leip", "evaluate", "--output_path","variants/compiled_tvm_fp32/","--framework", "tvm", "--input_names", input_names, "--input_types=float32", "--input_shapes", input_shapes, "--input_path", "variants/compiled_tvm_fp32/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
getResults("variants/compiled_tvm_fp32/", "Baseline TVM FP32")

#", "LEIP", "TVM", "INT8
logCmd(["rm", "-rf", "variants/leip_compiled_tvm_int8"])
logCmd(["mkdir", "variants/leip_compiled_tvm_int8"])
logCmd(["leip", "compile", "--input_path", "variants/checkpointCompressed/model_save/", "--input_shapes", input_shapes, "--output_path", "variants/leip_compiled_tvm_int8/bin", "--input_types=uint8", "--data_type=int8"])
logCmd(["leip", "evaluate", "--output_path","variants/leip_compiled_tvm_int8","--framework", "tvm", "--input_names", input_names, "--input_types=uint8", "--input_shapes", input_shapes, "--input_path", "variants/leip_compiled_tvm_int8/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
getResults("variants/leip_compiled_tvm_int8/", "LEIP TVM INT8")

#", "LEIP", "TVM", "FP32
logCmd(["rm", "-rf", "variants/leip_compiled_tvm_fp32"])
logCmd(["mkdir", "variants/leip_compiled_tvm_fp32"])
logCmd(["leip", "compile", "--input_path", "variants/checkpointCompressed/model_save/", "--input_shapes", input_shapes, "--output_path", "variants/leip_compiled_tvm_fp32/bin", "--input_types=float32", "--data_type=float32"])
logCmd(["leip", "evaluate", "--output_path","variants/leip_compiled_tvm_fp32","--framework", "tvm", "--input_names", input_names, "--input_types=float32", "--input_shapes", input_shapes, "--input_path", "variants/leip_compiled_tvm_fp32/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
getResults("variants/leip_compiled_tvm_fp32/", "LEIP TVM FP32")


for command in commands_run:
    line = ' '.join(command)
    print(line)
