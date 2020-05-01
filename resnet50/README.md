# Download pretrained model on Open Images 10 Classes
./dev_docker_run leip zoo download --model_id resnetv2-50 --variant_id keras-open-images-10-classes

# Download pretrained imagenet model
./dev_docker_run leip zoo download --model_id resnetv2-50 --variant_id keras-imagenet

# Download dataset for Transfer Learning training

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval

# Train a new model with Transfer Learning on top of a base trained on Imagenet

(Set --epochs and --batch_size to 1 for a quick training run.)

./dev_docker_run ./train.py --dataset_path datasets/open-images-10-classes/train/  --eval_dataset_path datasets/open-images-10-classes/eval/ --epochs 600

# Evaluate a trained model

./dev_docker_run ./eval.py --dataset_path datasets/open-images-10-classes/eval/ --input_model_path trained_model.h5

# Demo

This runs inference on a single image.
./dev_docker_run ./demo.py --input_model_path trained_model.h5 --image_file test_images/dog.jpg

# Run multi-evaluate on open images 10 classes model
dev-leip-run leip-evaluate-variants --model_id resnetv2-50 --model_variant keras-open-images-10-classes --dataset_id open-images-10-classes --dataset_variant eval --input_checkpoint workspace/models/resnetv2-50/keras-open-images-10-classes --dataset_index_file workspace/datasets/open-images-10-classes/eval/index.txt --class_names_file workspace/models/resnetv2-50/keras-open-images-10-classes/class_names.txt       --output_folder resnet50-oi
# Run multi-evaluate on imagenet model
dev-leip-run leip-evaluate-variants --model_id resnetv2-50 --model_variant keras-imagenet --input_checkpoint workspace/models/resnetv2-50/keras-imagenet --dataset_index_file /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names_file workspace/models/resnetv2-50/keras-imagenet/class_names.txt     --start_cmd_number 10 --output_folder resnet50-imagenet



# Run a converted checkpoint on a single image within LEIP SDK

Assuming your checkpoint is in "checkpoint/" after converting with ./convert_keras_model_to_checkpoint.py :

dev-leip-run leip run -in checkpoint/ --class_names class_names.txt --framework tf --preprocessor imagenet_caffe --test_path test_images/dog.jpg

Open Image 10 Classes Commands
# Preparation
leip zoo download --model_id resnetv2-50 --variant_id keras-open-images-10-classes
leip zoo download --dataset_id open-images-10-classes --variant_id eval
rm -rf resnet50-oi
mkdir resnet50-oi
mkdir resnet50-oi/baselineFp32Results
# CMD#1 Baseline FP32 TF
leip evaluate --output_path resnet50-oi/baselineFp32Results --framework tf2 --input_path workspace/models/resnetv2-50/keras-open-images-10-classes --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/resnetv2-50/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# LEIP Compress ASYMMETRIC
leip compress --input_path workspace/models/resnetv2-50/keras-open-images-10-classes --quantizer ASYMMETRIC --bits 8 --output_path resnet50-oi/checkpointCompressed/
# LEIP Compress POWER_OF_TWO (POW2)
leip compress --input_path workspace/models/resnetv2-50/keras-open-images-10-classes --quantizer POWER_OF_TWO --bits 8 --output_path resnet50-oi/checkpointCompressedPow2/
# CMD#2 LEIP FP32 TF
leip evaluate --output_path resnet50-oi/checkpointCompressed/ --framework tf2 --input_path resnet50-oi/checkpointCompressed/model_save/ --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/resnetv2-50/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#3 Baseline INT8 TVM
rm -rf resnet50-oi/compiled_tvm_int8
mkdir resnet50-oi/compiled_tvm_int8
leip compile --input_path workspace/models/resnetv2-50/keras-open-images-10-classes --output_path resnet50-oi/compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path resnet50-oi/compiled_tvm_int8/ --framework tvm --input_types=uint8 --input_path resnet50-oi/compiled_tvm_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/resnetv2-50/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#4 Baseline FP32 TVM
rm -rf resnet50-oi/compiled_tvm_fp32
mkdir resnet50-oi/compiled_tvm_fp32
leip compile --input_path workspace/models/resnetv2-50/keras-open-images-10-classes --output_path resnet50-oi/compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path resnet50-oi/compiled_tvm_fp32/ --framework tvm --input_types=float32 --input_path resnet50-oi/compiled_tvm_fp32/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/resnetv2-50/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#5 LEIP INT8 TVM
rm -rf resnet50-oi/leip_compiled_tvm_int8
mkdir resnet50-oi/leip_compiled_tvm_int8
leip compile --input_path resnet50-oi/checkpointCompressed/model_save/ --output_path resnet50-oi/leip_compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path resnet50-oi/leip_compiled_tvm_int8 --framework tvm --input_types=uint8 --input_path resnet50-oi/leip_compiled_tvm_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/resnetv2-50/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#6 LEIP FP32 TVM
rm -rf resnet50-oi/leip_compiled_tvm_fp32
mkdir resnet50-oi/leip_compiled_tvm_fp32
leip compile --input_path resnet50-oi/checkpointCompressed/model_save/ --output_path resnet50-oi/leip_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path resnet50-oi/leip_compiled_tvm_fp32 --framework tvm --input_types=float32 --input_path resnet50-oi/leip_compiled_tvm_fp32/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/resnetv2-50/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#7 LEIP-POW2 INT8 TVM
rm -rf resnet50-oi/leip_compiled_tvm_int8_pow2
mkdir resnet50-oi/leip_compiled_tvm_int8_pow2
leip compile --input_path resnet50-oi/checkpointCompressedPow2/model_save/ --output_path resnet50-oi/leip_compiled_tvm_int8_pow2/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path resnet50-oi/leip_compiled_tvm_int8_pow2 --framework tvm --input_types=uint8 --input_path resnet50-oi/leip_compiled_tvm_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/resnetv2-50/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#8 TfLite Asymmetric INT8 TF
rm -rf resnet50-oi/tfliteOutput
mkdir resnet50-oi/tfliteOutput
leip convert --input_path workspace/models/resnetv2-50/keras-open-images-10-classes --framework tflite --output_path resnet50-oi/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared-workdir/workspace/datasets/open-images-10-classes/eval/Apple/06e47f3aa0036947.jpg
leip evaluate --output_path resnet50-oi/tfliteOutput --framework tflite --input_types=uint8 --input_path resnet50-oi/tfliteOutput/model_save/inference_model.cast.tflite --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/resnetv2-50/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom --preprocessor ''
# CMD#9 TfLite Asymmetric INT8 TVM
leip compile --input_path resnet50-oi/tfliteOutput/model_save/inference_model.cast.tflite --output_path resnet50-oi/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path resnet50-oi/tfliteOutput/model_save/binuint8 --framework tvm --input_types=uint8 --input_path resnet50-oi/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/resnetv2-50/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom --preprocessor ''

Imagenet Commands
# Preparation
leip zoo download --model_id resnetv2-50 --variant_id keras-imagenet
rm -rf resnet50-imagenet
mkdir resnet50-imagenet
mkdir resnet50-imagenet/baselineFp32Results
# CMD#10 Baseline FP32 TF
leip evaluate --output_path resnet50-imagenet/baselineFp32Results --framework tf2 --input_path workspace/models/resnetv2-50/keras-imagenet --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/resnetv2-50/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# LEIP Compress ASYMMETRIC
leip compress --input_path workspace/models/resnetv2-50/keras-imagenet --quantizer ASYMMETRIC --bits 8 --output_path resnet50-imagenet/checkpointCompressed/
# LEIP Compress POWER_OF_TWO (POW2)
leip compress --input_path workspace/models/resnetv2-50/keras-imagenet --quantizer POWER_OF_TWO --bits 8 --output_path resnet50-imagenet/checkpointCompressedPow2/
# CMD#11 LEIP FP32 TF
leip evaluate --output_path resnet50-imagenet/checkpointCompressed/ --framework tf2 --input_path resnet50-imagenet/checkpointCompressed/model_save/ --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/resnetv2-50/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#12 Baseline INT8 TVM
rm -rf resnet50-imagenet/compiled_tvm_int8
mkdir resnet50-imagenet/compiled_tvm_int8
leip compile --input_path workspace/models/resnetv2-50/keras-imagenet --output_path resnet50-imagenet/compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path resnet50-imagenet/compiled_tvm_int8/ --framework tvm --input_types=uint8 --input_path resnet50-imagenet/compiled_tvm_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/resnetv2-50/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#13 Baseline FP32 TVM
rm -rf resnet50-imagenet/compiled_tvm_fp32
mkdir resnet50-imagenet/compiled_tvm_fp32
leip compile --input_path workspace/models/resnetv2-50/keras-imagenet --output_path resnet50-imagenet/compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path resnet50-imagenet/compiled_tvm_fp32/ --framework tvm --input_types=float32 --input_path resnet50-imagenet/compiled_tvm_fp32/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/resnetv2-50/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#14 LEIP INT8 TVM
rm -rf resnet50-imagenet/leip_compiled_tvm_int8
mkdir resnet50-imagenet/leip_compiled_tvm_int8
leip compile --input_path resnet50-imagenet/checkpointCompressed/model_save/ --output_path resnet50-imagenet/leip_compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path resnet50-imagenet/leip_compiled_tvm_int8 --framework tvm --input_types=uint8 --input_path resnet50-imagenet/leip_compiled_tvm_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/resnetv2-50/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#15 LEIP FP32 TVM
rm -rf resnet50-imagenet/leip_compiled_tvm_fp32
mkdir resnet50-imagenet/leip_compiled_tvm_fp32
leip compile --input_path resnet50-imagenet/checkpointCompressed/model_save/ --output_path resnet50-imagenet/leip_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path resnet50-imagenet/leip_compiled_tvm_fp32 --framework tvm --input_types=float32 --input_path resnet50-imagenet/leip_compiled_tvm_fp32/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/resnetv2-50/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#16 LEIP-POW2 INT8 TVM
rm -rf resnet50-imagenet/leip_compiled_tvm_int8_pow2
mkdir resnet50-imagenet/leip_compiled_tvm_int8_pow2
leip compile --input_path resnet50-imagenet/checkpointCompressedPow2/model_save/ --output_path resnet50-imagenet/leip_compiled_tvm_int8_pow2/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path resnet50-imagenet/leip_compiled_tvm_int8_pow2 --framework tvm --input_types=uint8 --input_path resnet50-imagenet/leip_compiled_tvm_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/resnetv2-50/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#17 TfLite Asymmetric INT8 TF
rm -rf resnet50-imagenet/tfliteOutput
mkdir resnet50-imagenet/tfliteOutput
leip convert --input_path workspace/models/resnetv2-50/keras-imagenet --framework tflite --output_path resnet50-imagenet/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared/data/sample-models/resources/images/imagenet_images/preprocessed/ILSVRC2012_val_00000001.JPEG
leip evaluate --output_path resnet50-imagenet/tfliteOutput --framework tflite --input_types=uint8 --input_path resnet50-imagenet/tfliteOutput/model_save/inference_model.cast.tflite --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/resnetv2-50/keras-imagenet/class_names.txt --task=classifier --dataset=custom --preprocessor ''
# CMD#18 TfLite Asymmetric INT8 TVM
leip compile --input_path resnet50-imagenet/tfliteOutput/model_save/inference_model.cast.tflite --output_path resnet50-imagenet/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path resnet50-imagenet/tfliteOutput/model_save/binuint8 --framework tvm --input_types=uint8 --input_path resnet50-imagenet/tfliteOutput/model_save/binuint8 --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/resnetv2-50/keras-imagenet/class_names.txt --task=classifier --dataset=custom --preprocessor ''

