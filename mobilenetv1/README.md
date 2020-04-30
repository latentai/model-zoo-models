# MobileNetV1

# Download pretrained model on Open Images 10 Classes
./dev_docker_run leip zoo download --model_id mobilenetv1 --variant_id keras-open-images-10-classes

# Download pretrained imagenet model
./dev_docker_run leip zoo download --model_id mobilenetv1 --variant_id keras-imagenet

# Download dataset for Transfer Learning training

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval

# Train

(Set --epochs and --batch_size to 1 for a quick training run.)
48 epochs produced about 80% top1 accuracy. 150 epochs would perform better.

./dev_docker_run ./train.py --dataset_path datasets/open-images-10-classes/train/train/  --eval_dataset_path datasets/open-images-10-classes/eval/eval/ --epochs 48

# Evaluate a trained model

./dev_docker_run ./eval.py --dataset_path datasets/open-images-10-classes/eval/eval/ --input_model_path trained_model.h5

# Demo

This runs inference on a single image.
./dev_docker_run ./demo.py --input_model_path trained_model.h5 --image_file test_images/dog.jpg

# Run a converted checkpoint on a single image within LEIP SDK

Assuming your checkpoint is in "checkpoint/" after converting with ./convert_keras_model_to_checkpoint.py :

dev-leip-run leip run --input_path checkpoint/ --class_names class_names.txt --framework tf --preprocessor imagenet_caffe --test_path test_images/dog.jpg

# Run multi-evaluate on open images 10 classes model
dev-leip-run leip-evaluate-variants --model_id mobilenetv1 --model_variant keras-open-images-10-classes --dataset_id open-images-10-classes --dataset_variant eval --input_checkpoint workspace/models/mobilenetv1/keras-open-images-10-classes --dataset_index_file workspace/datasets/open-images-10-classes/eval/eval/index.txt --class_names_file workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt       --output_folder mobilenetv1-oi
# Run multi-evaluate on imagenet model [crashes with TVM error]
dev-leip-run leip-evaluate-variants --model_id mobilenetv1 --model_variant keras-imagenet --input_checkpoint workspace/models/mobilenetv1/keras-imagenet --dataset_index_file /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names_file workspace/models/mobilenetv1/keras-imagenet/class_names.txt      --start_cmd_number 10 --output_folder mobilenetv1-imagenet


# Evaluate baseline model within LEIP SDK

dev-leip-run leip evaluate -fw tf --input_path checkpoint/ --test_path=datasets/open-images-10-classes/eval/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe


Open Image 10 Classes Commands
# Preparation
leip zoo download --model_id mobilenetv1 --variant_id keras-open-images-10-classes
leip zoo download --dataset_id open-images-10-classes --variant_id eval
rm -rf mobilenetv1-oi
mkdir mobilenetv1-oi
mkdir mobilenetv1-oi/baselineFp32Results
# CMD#1 Baseline FP32 TF
leip evaluate --output_path mobilenetv1-oi/baselineFp32Results --framework tf2 --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --test_path workspace/datasets/open-images-10-classes/eval/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# LEIP Compress ASYMMETRIC
leip compress --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --quantizer ASYMMETRIC --bits 8 --output_path mobilenetv1-oi/checkpointCompressed/
# LEIP Compress POWER_OF_TWO (POW2)
leip compress --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --quantizer POWER_OF_TWO --bits 8 --output_path mobilenetv1-oi/checkpointCompressedPow2/
# CMD#2 LEIP FP32 TF
leip evaluate --output_path mobilenetv1-oi/checkpointCompressed/ --framework tf2 --input_path mobilenetv1-oi/checkpointCompressed/model_save/ --test_path workspace/datasets/open-images-10-classes/eval/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#3 Baseline INT8 TVM
rm -rf mobilenetv1-oi/compiled_tvm_int8
mkdir mobilenetv1-oi/compiled_tvm_int8
leip compile --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --output_path mobilenetv1-oi/compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv1-oi/compiled_tvm_int8/ --framework tvm --input_types=uint8 --input_path mobilenetv1-oi/compiled_tvm_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#4 Baseline FP32 TVM
rm -rf mobilenetv1-oi/compiled_tvm_fp32
mkdir mobilenetv1-oi/compiled_tvm_fp32
leip compile --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --output_path mobilenetv1-oi/compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv1-oi/compiled_tvm_fp32/ --framework tvm --input_types=float32 --input_path mobilenetv1-oi/compiled_tvm_fp32/bin --test_path workspace/datasets/open-images-10-classes/eval/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#5 LEIP INT8 TVM
rm -rf mobilenetv1-oi/leip_compiled_tvm_int8
mkdir mobilenetv1-oi/leip_compiled_tvm_int8
leip compile --input_path mobilenetv1-oi/checkpointCompressed/model_save/ --output_path mobilenetv1-oi/leip_compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv1-oi/leip_compiled_tvm_int8 --framework tvm --input_types=uint8 --input_path mobilenetv1-oi/leip_compiled_tvm_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#6 LEIP FP32 TVM
rm -rf mobilenetv1-oi/leip_compiled_tvm_fp32
mkdir mobilenetv1-oi/leip_compiled_tvm_fp32
leip compile --input_path mobilenetv1-oi/checkpointCompressed/model_save/ --output_path mobilenetv1-oi/leip_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv1-oi/leip_compiled_tvm_fp32 --framework tvm --input_types=float32 --input_path mobilenetv1-oi/leip_compiled_tvm_fp32/bin --test_path workspace/datasets/open-images-10-classes/eval/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#7 LEIP-POW2 INT8 TVM
rm -rf mobilenetv1-oi/leip_compiled_tvm_int8_pow2
mkdir mobilenetv1-oi/leip_compiled_tvm_int8_pow2
leip compile --input_path mobilenetv1-oi/checkpointCompressedPow2/model_save/ --output_path mobilenetv1-oi/leip_compiled_tvm_int8_pow2/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv1-oi/leip_compiled_tvm_int8_pow2 --framework tvm --input_types=uint8 --input_path mobilenetv1-oi/leip_compiled_tvm_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom
# CMD#8 TfLite Asymmetric INT8 TF
rm -rf mobilenetv1-oi/tfliteOutput
mkdir mobilenetv1-oi/tfliteOutput
leip convert --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --framework tflite --output_path mobilenetv1-oi/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared-workdir/workspace/datasets/open-images-10-classes/eval/eval/Apple/06e47f3aa0036947.jpg
leip evaluate --output_path mobilenetv1-oi/tfliteOutput --framework tflite --input_types=uint8 --input_path mobilenetv1-oi/tfliteOutput/model_save/inference_model.cast.tflite --test_path workspace/datasets/open-images-10-classes/eval/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom --preprocessor ''
# CMD#9 TfLite Asymmetric INT8 TVM
leip compile --input_path mobilenetv1-oi/tfliteOutput/model_save/inference_model.cast.tflite --output_path mobilenetv1-oi/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path mobilenetv1-oi/tfliteOutput/model_save/binuint8 --framework tvm --input_types=uint8 --input_path mobilenetv1-oi/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/open-images-10-classes/eval/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt --task=classifier --dataset=custom --preprocessor ''

