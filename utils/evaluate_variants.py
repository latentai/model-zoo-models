#!/usr/bin/env python3

import subprocess

input_checkpoint = 'imagenet_checkpoint/'
dataset_index_file = "/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt"
class_names_file = '/data/sample-models/resources/data/imagenet/imagenet1000.names'
preprocessor = 'imagenet_caffe'
input_names = 'input_1'
output_names = 'probs/Softmax'
input_shapes = '1,224,224,3'

subprocess.check_call("rm -rf variants")
subprocess.check_call("mkdir variants")

# Imagenet Baseline TF FP32
subprocess.check_call("leip evaluate -fw tf2 -in "+input_checkpoint+" --test_path="+dataset_index_file+" --class_names="+class_names_file+" --task=classifier --dataset=custom  --preprocessor "+preprocessor+" --input_shapes "+input_shapes+" --input_names "+input_names+" --output_names "+output_names+"")
# Imagenet LEIP Compress

subprocess.check_call("leip compress -in "+input_checkpoint+" -q ASYMMETRIC -b 8 -out variants/checkpointCompressed/")
subprocess.check_call("leip compress -in "+input_checkpoint+" -q POWER_OF_TWO -b 8 -out variants/checkpointCompressedPow2/")

# Imagenet LEIP TF FP32
subprocess.check_call("leip evaluate -fw tf2 -in variants/checkpointCompressed/model_save/ --test_path="+dataset_index_file+" --class_names="+class_names_file+" --task=classifier --dataset=custom  --preprocessor "+preprocessor+"  --input_shapes "+input_shapes+" --input_names "+input_names+" --output_names "+output_names+"")

# Imagenet Baseline TVM INT8
subprocess.check_call("rm -rf variants/compiled_tvm_int8")
subprocess.check_call("mkdir variants/compiled_tvm_int8")
subprocess.check_call("leip compile -in "+input_checkpoint+" -ishapes "+input_shapes+" -o variants/compiled_tvm_int8/bin --input_types=uint8  --data_type=int8")
subprocess.check_call("leip evaluate -fw tvm --input_names "+input_names+" --input_types=uint8 -ishapes "+input_shapes+" -in variants/compiled_tvm_int8/bin --test_path="+dataset_index_file+" --class_names="+class_names_file+" --task=classifier --dataset=custom  --preprocessor "+preprocessor)
# Imagenet Baseline TVM FP32
subprocess.check_call("rm -rf variants/compiled_tvm_fp32")
subprocess.check_call("mkdir variants/compiled_tvm_fp32")
subprocess.check_call("leip compile -in "+input_checkpoint+" -ishapes "+input_shapes+" -o variants/compiled_tvm_fp32/bin --input_types=float32  --data_type=float32")
subprocess.check_call("leip evaluate -fw tvm --input_names "+input_names+" --input_types=float32 -ishapes "+input_shapes+" -in variants/compiled_tvm_fp32/bin --test_path="+dataset_index_file+" --class_names="+class_names_file+" --task=classifier --dataset=custom  --preprocessor "+preprocessor)
# Imagenet LEIP TVM INT8
subprocess.check_call("rm -rf variants/leip_compiled_tvm_int8")
subprocess.check_call("mkdir variants/leip_compiled_tvm_int8")
subprocess.check_call("leip compile -in variants/checkpointCompressed/model_save/ -ishapes "+input_shapes+" -o variants/leip_compiled_tvm_int8/bin --input_types=uint8  --data_type=int8")
subprocess.check_call("leip evaluate -fw tvm --input_names "+input_names+" --input_types=uint8 -ishapes "+input_shapes+" -in variants/leip_compiled_tvm_int8/bin --test_path="+dataset_index_file+" --class_names="+class_names_file+" --task=classifier --dataset=custom  --preprocessor "+preprocessor)
# Imagenet LEIP TVM FP32
subprocess.check_call("rm -rf variants/leip_compiled_tvm_fp32")
subprocess.check_call("mkdir variants/leip_compiled_tvm_fp32")
subprocess.check_call("leip compile -in variants/checkpointCompressed/model_save/ -ishapes "+input_shapes+" -o variants/leip_compiled_tvm_fp32/bin --input_types=float32  --data_type=float32")
subprocess.check_call("leip evaluate -fw tvm --input_names "+input_names+" --input_types=float32 -ishapes "+input_shapes+" -in variants/leip_compiled_tvm_fp32/bin --test_path="+dataset_index_file+" --class_names="+class_names_file+" --task=classifier --dataset=custom  --preprocessor "+preprocessor)
