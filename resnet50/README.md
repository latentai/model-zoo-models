# Download pretrained model on Open Images 10 Classes
leip zoo download resnetv2-50 keras-open-images-10-classes

# Download pretrained imagenet model
leip zoo download resnetv2-50 keras-imagenet

# Download dataset for Transfer Learning training

./dev_docker_run ./download_dataset.py

# Train a new model with Transfer Learning on top of a base trained on Imagenet

(Set --epochs and --batch_size to 1 for a quick training run.)

./dev_docker_run ./train.py --dataset_path datasets/open_images_10_classes_200/ --epochs 600

# Convert Trained Model to TF Checkpoint format for use in LEIP SDK

./dev_docker_run ./utils/convert_keras_model_to_checkpoint.py --input_model_path trained_model.h5

# Evaluate a trained model

./dev_docker_run ./eval.py --dataset_path datasets/open_images_10_classes_200/ --input_model_path trained_model.h5

# Demo

This runs inference on a single image.
./dev_docker_run ./demo.py --input_model_path trained_model.h5 --image_file test_images/dog.jpg

# Run a converted checkpoint on a single image within LEIP SDK

Assuming your checkpoint is in "checkpoint/" after converting with ./convert_keras_model_to_checkpoint.py :

dev-leip-run leip run -in checkpoint/ --class_names class_names.txt --framework tf --preprocessor imagenet_caffe --test_path test_images/dog.jpg

rm -rf variants baselineFp32Results
mkdir variants
mkdir baselineFp32Results
# TF FP32 Baseline
leip evaluate --output_path baselineFp32Results --framework tf2 --input_path checkpoint/ --test_path datasets/open_images_10_classes_200/eval/index.txt --class_names class_names.txt --task=classifier --dataset=custom --preprocessor imagenet_caffe --input_shapes 1,224,224,3 --input_names input_1 --output_names dense/Softmax
# LEIP Compress
leip compress --input_path checkpoint/ --quantizer ASYMMETRIC --bits 8 --output_path variants/checkpointCompressed/
leip compress --input_path checkpoint/ --quantizer POWER_OF_TWO --bits 8 --output_path variants/checkpointCompressedPow2/
# TF FP32 LEIP
leip evaluate --output_path variants/checkpointCompressed/ --framework tf2 --input_path variants/checkpointCompressed/model_save/ --test_path datasets/open_images_10_classes_200/eval/index.txt --class_names class_names.txt --task=classifier --dataset=custom --preprocessor imagenet_caffe --input_shapes 1,224,224,3 --input_names input_1 --output_names dense/Softmax
# TVM INT8 Baseline
rm -rf variants/compiled_tvm_int8
mkdir variants/compiled_tvm_int8
leip compile --input_path checkpoint/ --input_shapes 1,224,224,3 --output_path variants/compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path variants/compiled_tvm_int8/ --framework tvm --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path variants/compiled_tvm_int8/bin --test_path datasets/open_images_10_classes_200/eval/index.txt --class_names class_names.txt --task=classifier --dataset=custom --preprocessor imagenet_caffe
# TVM FP32 Baseline
rm -rf variants/compiled_tvm_fp32
mkdir variants/compiled_tvm_fp32
leip compile --input_path checkpoint/ --input_shapes 1,224,224,3 --output_path variants/compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path variants/compiled_tvm_fp32/ --framework tvm --input_names input_1 --input_types=float32 --input_shapes 1,224,224,3 --input_path variants/compiled_tvm_fp32/bin --test_path datasets/open_images_10_classes_200/eval/index.txt --class_names class_names.txt --task=classifier --dataset=custom --preprocessor imagenet_caffe
# TVM INT8 LEIP
rm -rf variants/leip_compiled_tvm_int8
mkdir variants/leip_compiled_tvm_int8
leip compile --input_path variants/checkpointCompressed/model_save/ --input_shapes 1,224,224,3 --output_path variants/leip_compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path variants/leip_compiled_tvm_int8 --framework tvm --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path variants/leip_compiled_tvm_int8/bin --test_path datasets/open_images_10_classes_200/eval/index.txt --class_names class_names.txt --task=classifier --dataset=custom --preprocessor imagenet_caffe
# TVM FP32 LEIP
rm -rf variants/leip_compiled_tvm_fp32
mkdir variants/leip_compiled_tvm_fp32
leip compile --input_path variants/checkpointCompressed/model_save/ --input_shapes 1,224,224,3 --output_path variants/leip_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path variants/leip_compiled_tvm_fp32 --framework tvm --input_names input_1 --input_types=float32 --input_shapes 1,224,224,3 --input_path variants/leip_compiled_tvm_fp32/bin --test_path datasets/open_images_10_classes_200/eval/index.txt --class_names class_names.txt --task=classifier --dataset=custom --preprocessor imagenet_caffe
