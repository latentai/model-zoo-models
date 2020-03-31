# MobileNetV2

# Download pretrained model on open images 10 classes

./dev_docker_run leip zoo download --model_id mobilenetv2 --variant_id keras-open-images-10-classes

# Download dataset for Transfer Learning training

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval

# Train

(Set --epochs and --batch_size to 1 for a quick training run.)

./dev_docker_run ./train.py --dataset_path datasets/open_images_10_classes_200/ --epochs 150

# Convert Trained Model to TF Checkpoint format for use in LEIP SDK

./dev_docker_run ./utils/convert_keras_model_to_checkpoint.py --input_model_path trained_model.h5

# Evaluate a trained model

./dev_docker_run ./eval.py --dataset_path latentai-zoo-models/datasets/open-images-10-classes/eval/eval/ --input_model_path trained_model.h5

# Demo

This runs inference on a single image.
./dev_docker_run ./demo.py --input_model_path trained_model.h5 --image_file test_images/dog.jpg

# Run multiple variant evaluation on imagenet
dev-leip-run leip-evaluate-variants --input_checkpoint imagenet_checkpoint --dataset_index_file /data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names_file /data/sample-models/resources/data/imagenet/imagenet1000.names --preprocessor imagenet --input_names input_1 --output_names Logits/Softmax --input_shapes 1,224,224,3

# Run multiple variant evaluation on open images 10 classes eval
dev-leip-run leip-evaluate-variants --input_checkpoint checkpoint --dataset_index_file latentai-zoo-models/datasets/open-images-10-classes/eval/eval/index.txt --class_names_file class_names.txt --preprocessor imagenet_caffe --input_names input_1 --output_names dense_3/Softmax --input_shapes 1,224,224,3


# Run a converted checkpoint on a single image within LEIP SDK

Assuming your checkpoint is in "checkpoint/" after converting with ./convert_keras_model_to_checkpoint.py :

dev-leip-run leip run -in checkpoint/ --class_names class_names.txt --framework tf --preprocessor imagenet_caffe --test_path test_images/dog.jpg

# Evaluate baseline model within LEIP SDK

dev-leip-run leip evaluate -fw tf -in checkpoint/ --test_path=latentai-zoo-models/datasets/open-images-10-classes/eval/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe


# Evaluate with TVM
### Baseline Compile with TVM INT8
rm -rf compiled_tvm_int8
mkdir compiled_tvm_int8
dev-leip-run leip compile -in checkpoint/ -ishapes "1, 224, 224, 3" -o compiled_tvm_int8/bin --input_types=uint8  --data_type=int8
### Run compiled model with INT8 on single image
dev-leip-run leip run -fw tvm --input_names input_1 --input_types=uint8 -ishapes "1, 224, 224, 3" -in compiled_tvm_int8/bin --class_names class_names.txt --preprocessor imagenet_caffe --test_path test_images/dog.jpg
### Evaluate compiled model with INT8
dev-leip-run leip evaluate -fw tvm --input_names input_1 --input_types=uint8 -ishapes "1, 224, 224, 3" -in compiled_tvm_int8/bin --test_path=latentai-zoo-models/datasets/open-images-10-classes/eval/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe

### Baseline Compile with TVM FP32
rm -rf compiled_tvm_fp32
mkdir compiled_tvm_fp32
dev-leip-run leip compile -in checkpoint/ -ishapes "1, 224, 224, 3" -o compiled_tvm_fp32/bin --input_types=float32  --data_type=float32
### Run compiled model with FP32 on single image
dev-leip-run leip run -fw tvm --input_names input_1 --input_types=float32 -ishapes "1, 224, 224, 3" -in compiled_tvm_fp32/bin --class_names class_names.txt --preprocessor imagenet_caffe --test_path test_images/dog.jpg
### Evaluate compiled model with FP32
dev-leip-run leip evaluate -fw tvm --input_names input_1 --input_types=float32 -ishapes "1, 224, 224, 3" -in compiled_tvm_fp32/bin --test_path=latentai-zoo-models/datasets/open-images-10-classes/eval/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe


# LEIP Compress

dev-leip-run leip compress -in checkpoint/ -q ASYMMETRIC -b 8 -out checkpointCompressed/
dev-leip-run leip compress -in checkpoint/ -q POWER_OF_TWO -b 8 -out checkpointCompressedPow2/

# Evaluate compressed with TF

dev-leip-run leip evaluate -fw tf -in checkpointCompressed/model_save/ --test_path=latentai-zoo-models/datasets/open-images-10-classes/eval/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe

# Evaluate compressed with TVM
### LEIP Compile with TVM INT8
rm -rf leip_compiled_tvm_int8
mkdir leip_compiled_tvm_int8
dev-leip-run leip compile -in checkpointCompressed/model_save/ -ishapes "1, 224, 224, 3" -o leip_compiled_tvm_int8/bin --input_types=uint8  --data_type=int8
### Run compiled model with INT8 on single image
dev-leip-run leip run -fw tvm --input_names input_1 --input_types=uint8 -ishapes "1, 224, 224, 3" -in leip_compiled_tvm_int8/bin --class_names class_names.txt --preprocessor imagenet_caffe --test_path test_images/dog.jpg
### Evaluate compiled model with INT8
dev-leip-run leip evaluate -fw tvm --input_names input_1 --input_types=uint8 -ishapes "1, 224, 224, 3" -in leip_compiled_tvm_int8/bin --test_path=latentai-zoo-models/datasets/open-images-10-classes/eval/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe

### LEIP Compile with TVM INT8 Pow2
rm -rf leip_compiled_tvm_int8_pow2
mkdir leip_compiled_tvm_int8_pow2
dev-leip-run leip compile -in checkpointCompressedPow2/model_save/ -ishapes "1, 224, 224, 3" -o leip_compiled_tvm_int8_pow2/bin --input_types=uint8  --data_type=int8

### LEIP Compile with TVM FP32
rm -rf leip_compiled_tvm_fp32
mkdir leip_compiled_tvm_fp32
dev-leip-run leip compile -in checkpointCompressed/model_save/ -ishapes "1, 224, 224, 3" -o leip_compiled_tvm_fp32/bin --input_types=float32  --data_type=float32
### Run compiled model with FP32 on single image
dev-leip-run leip run -fw tvm --input_names input_1 --input_types=float32 -ishapes "1, 224, 224, 3" -in leip_compiled_tvm_fp32/bin --class_names class_names.txt --preprocessor imagenet_caffe --test_path test_images/dog.jpg
### Evaluate compiled model with FP32
dev-leip-run leip evaluate -fw tvm --input_names input_1 --input_types=float32 -ishapes "1, 224, 224, 3" -in leip_compiled_tvm_fp32/bin --test_path=latentai-zoo-models/datasets/open-images-10-classes/eval/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe
