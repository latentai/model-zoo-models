#!/usr/bin/env python3

import subprocess

input_checkpoint = 'imagenet_checkpoint/'
dataset_index_file = "/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt"
class_names_file = '/data/sample-models/resources/data/imagenet/imagenet1000.names'
preprocessor = 'imagenet_caffe'
input_names = 'input_1'
output_names = 'probs/Softmax'
input_shapes = '1,224,224,3'

subprocess.check_call(["rm", "-rf", "variants"])
subprocess.check_call(["mkdir", "variants"])
subprocess.check_call(["mkdir", "baselineFp32Results"])

#", "Baseline", "TF", "FP32
subprocess.check_call(["leip", "evaluate", "--output_path","baselineFp32Results", "--framework", "tf2", "--input_path", input_checkpoint, "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor, "--input_shapes", input_shapes, "--input_names", input_names, "--output_names", output_names])
#", "LEIP", "Compress

subprocess.check_call(["leip", "compress", "--input_path", input_checkpoint, "--quantizer", "ASYMMETRIC", "--bits", "8", "--output_path", "variants/checkpointCompressed/"])
subprocess.check_call(["leip", "compress", "--input_path", input_checkpoint, "--quantizer", "POWER_OF_TWO", "--bits", "8", "--output_path", "variants/checkpointCompressedPow2/"])

#", "LEIP", "TF", "FP32
subprocess.check_call(["leip", "evaluate", "--output_path","variants/checkpointCompressed/", "--framework", "tf2", "--input_path", "variants/checkpointCompressed/model_save/", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor, "--input_shapes", input_shapes, "--input_names", input_names, "--output_names", output_names])

#", "Baseline", "TVM", "INT8
subprocess.check_call(["rm", "-rf", "variants/compiled_tvm_int8"])
subprocess.check_call(["mkdir", "variants/compiled_tvm_int8"])
subprocess.check_call(["leip", "compile", "--input_path", input_checkpoint, "--input_shapes", input_shapes, "--output_path", "variants/compiled_tvm_int8/bin", "--input_types=uint8", "--data_type=int8"])
subprocess.check_call(["leip", "evaluate", "--output_path","variants/compiled_tvm_int8/","--framework", "tvm", "--input_names", input_names, "--input_types=uint8", "--input_shapes", input_shapes, "--input_path", "variants/compiled_tvm_int8/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
#", "Baseline", "TVM", "FP32
subprocess.check_call(["rm", "-rf", "variants/compiled_tvm_fp32"])
subprocess.check_call(["mkdir", "variants/compiled_tvm_fp32"])
subprocess.check_call(["leip", "compile", "--input_path", input_checkpoint, "--input_shapes", input_shapes, "--output_path", "variants/compiled_tvm_fp32/bin", "--input_types=float32", "--data_type=float32"])
subprocess.check_call(["leip", "evaluate", "--output_path","variants/compiled_tvm_fp32/","--framework", "tvm", "--input_names", input_names, "--input_types=float32", "--input_shapes", input_shapes, "--input_path", "variants/compiled_tvm_fp32/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
#", "LEIP", "TVM", "INT8
subprocess.check_call(["rm", "-rf", "variants/leip_compiled_tvm_int8"])
subprocess.check_call(["mkdir", "variants/leip_compiled_tvm_int8"])
subprocess.check_call(["leip", "compile", "--input_path", "variants/checkpointCompressed/model_save/", "--input_shapes", input_shapes, "--output_path", "variants/leip_compiled_tvm_int8/bin", "--input_types=uint8", "--data_type=int8"])
subprocess.check_call(["leip", "evaluate", "--output_path","variants/leip_compiled_tvm_int8","--framework", "tvm", "--input_names", input_names, "--input_types=uint8", "--input_shapes", input_shapes, "--input_path", "variants/leip_compiled_tvm_int8/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
#", "LEIP", "TVM", "FP32
subprocess.check_call(["rm", "-rf", "variants/leip_compiled_tvm_fp32"])
subprocess.check_call(["mkdir", "variants/leip_compiled_tvm_fp32"])
subprocess.check_call(["leip", "compile", "--input_path", "variants/checkpointCompressed/model_save/", "--input_shapes", input_shapes, "--output_path", "variants/leip_compiled_tvm_fp32/bin", "--input_types=float32", "--data_type=float32"])
subprocess.check_call(["leip", "evaluate", "--output_path","variants/leip_compiled_tvm_fp32","--framework", "tvm", "--input_names", input_names, "--input_types=float32", "--input_shapes", input_shapes, "--input_path", "variants/leip_compiled_tvm_fp32/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
