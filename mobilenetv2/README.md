# MobileNetV2

# Download pretrained model on Open Images 10 Classes
./dev_docker_run leip zoo download --model_id mobilenetv2 --variant_id keras-open-images-10-classes

# Download pretrained imagenet model
./dev_docker_run leip zoo download --model_id mobilenetv2 --variant_id keras-imagenet

# Download dataset for Transfer Learning training

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval

# Train

(Set --epochs and --batch_size to 1 for a quick training run.)

./dev_docker_run ./train.py --dataset_path datasets/open_images_10_classes_200/ --epochs 150

# Convert Trained Model to TF Checkpoint format for use in LEIP SDK

./dev_docker_run ./utils/convert_keras_model_to_checkpoint.py --input_model_path trained_model.h5

# Evaluate a trained model

./dev_docker_run ./eval.py --dataset_path datasets/open-images-10-classes/eval/eval/ --input_model_path trained_model.h5

# Demo

This runs inference on a single image.
./dev_docker_run ./demo.py --input_model_path trained_model.h5 --image_file test_images/dog.jpg

# Run multi-evaluate on open images 10 classes model
dev-leip-run leip-evaluate-variants   --model_id mobilenetv2 --model_variant keras-open-images-10-classes-tf-checkpoint --dataset_id open-images-10-classes --dataset_variant eval --input_checkpoint models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint --dataset_index_file datasets/open-images-10-classes/eval/eval/index.txt --class_names_file models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint/class_names.txt --preprocessor 'float32' --input_names input_1 --output_names dense_3/Softmax --input_shapes 1,224,224,3 --output_folder mobilenetv2-oi > /home/kevin/model-zoo-models/mobilenetv2-open_images.txt
# Run multi-evaluate on imagenet model
dev-leip-run leip-evaluate-variants   --model_id mobilenetv2 --model_variant keras-imagenet-tf-checkpoint --input_checkpoint models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint --dataset_index_file /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names_file models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint/class_names.txt --preprocessor imagenet --input_names input_1 --output_names Logits/Softmax --input_shapes 1,224,224,3 --start_cmd_number 10 --output_folder mobilenetv2-imagenet  > /home/kevin/model-zoo-models/mobilenetv2-imagenet.txt


# Run a converted checkpoint on a single image within LEIP SDK

Assuming your checkpoint is in "checkpoint/" after converting with ./convert_keras_model_to_checkpoint.py :

dev-leip-run leip run -in checkpoint/ --class_names class_names.txt --framework tf --preprocessor imagenet_caffe --test_path test_images/dog.jpg

# Evaluate baseline model within LEIP SDK

dev-leip-run leip evaluate -fw tf -in checkpoint/ --test_path=datasets/open-images-10-classes/eval/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe

# Open Image 10 Classes Commands
# Preparation
leip zoo download --model_id mobilenetv2 --variant_id keras-open-images-10-classes-tf-checkpoint
leip zoo download --dataset_id open-images-10-classes --variant_id eval
rm -rf mobilenetv2-oi
mkdir mobilenetv2-oi
mkdir mobilenetv2-oi/baselineFp32Results
# CMD#1 Baseline FP32 TF
leip evaluate --output_path mobilenetv2-oi/baselineFp32Results --framework tf2 --input_path models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint --test_path datasets/open-images-10-classes/eval/eval/index.txt --class_names models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor float32 --input_shapes 1,224,224,3 --input_names input_1 --output_names dense_3/Softmax
# LEIP Compress ASYMMETRIC
leip compress --input_path models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint --quantizer ASYMMETRIC --bits 8 --output_path mobilenetv2-oi/checkpointCompressed/
# LEIP Compress POWER_OF_TWO (POW2)
leip compress --input_path models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint --quantizer POWER_OF_TWO --bits 8 --output_path mobilenetv2-oi/checkpointCompressedPow2/
# CMD#2 LEIP FP32 TF
leip evaluate --output_path mobilenetv2-oi/checkpointCompressed/ --framework tf2 --input_path mobilenetv2-oi/checkpointCompressed/model_save/ --test_path datasets/open-images-10-classes/eval/eval/index.txt --class_names models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor float32 --input_shapes 1,224,224,3 --input_names input_1 --output_names dense_3/Softmax
# CMD#3 Baseline INT8 TVM
rm -rf mobilenetv2-oi/compiled_tvm_int8
mkdir mobilenetv2-oi/compiled_tvm_int8
leip compile --input_path models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint --input_shapes 1,224,224,3 --output_path mobilenetv2-oi/compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-oi/compiled_tvm_int8/ --framework tvm --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path mobilenetv2-oi/compiled_tvm_int8/bin --test_path datasets/open-images-10-classes/eval/eval/index.txt --class_names models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor float32
# CMD#4 Baseline FP32 TVM
rm -rf mobilenetv2-oi/compiled_tvm_fp32
mkdir mobilenetv2-oi/compiled_tvm_fp32
leip compile --input_path models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint --input_shapes 1,224,224,3 --output_path mobilenetv2-oi/compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv2-oi/compiled_tvm_fp32/ --framework tvm --input_names input_1 --input_types=float32 --input_shapes 1,224,224,3 --input_path mobilenetv2-oi/compiled_tvm_fp32/bin --test_path datasets/open-images-10-classes/eval/eval/index.txt --class_names models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor float32
# CMD#5 LEIP INT8 TVM
rm -rf mobilenetv2-oi/leip_compiled_tvm_int8
mkdir mobilenetv2-oi/leip_compiled_tvm_int8
leip compile --input_path mobilenetv2-oi/checkpointCompressed/model_save/ --input_shapes 1,224,224,3 --output_path mobilenetv2-oi/leip_compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-oi/leip_compiled_tvm_int8 --framework tvm --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path mobilenetv2-oi/leip_compiled_tvm_int8/bin --test_path datasets/open-images-10-classes/eval/eval/index.txt --class_names models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor float32
# CMD#6 LEIP FP32 TVM
rm -rf mobilenetv2-oi/leip_compiled_tvm_fp32
mkdir mobilenetv2-oi/leip_compiled_tvm_fp32
leip compile --input_path mobilenetv2-oi/checkpointCompressed/model_save/ --input_shapes 1,224,224,3 --output_path mobilenetv2-oi/leip_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv2-oi/leip_compiled_tvm_fp32 --framework tvm --input_names input_1 --input_types=float32 --input_shapes 1,224,224,3 --input_path mobilenetv2-oi/leip_compiled_tvm_fp32/bin --test_path datasets/open-images-10-classes/eval/eval/index.txt --class_names models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor float32
# CMD#7 LEIP-POW2 INT8 TVM
rm -rf mobilenetv2-oi/leip_compiled_tvm_int8_pow2
mkdir mobilenetv2-oi/leip_compiled_tvm_int8_pow2
leip compile --input_path mobilenetv2-oi/checkpointCompressedPow2/model_save/ --input_shapes 1,224,224,3 --output_path mobilenetv2-oi/leip_compiled_tvm_int8_pow2/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-oi/leip_compiled_tvm_int8_pow2 --framework tvm --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path mobilenetv2-oi/leip_compiled_tvm_int8/bin --test_path datasets/open-images-10-classes/eval/eval/index.txt --class_names models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor float32
# CMD#8 TfLite Asymmetric INT8 TF
rm -rf mobilenetv2-oi/tfliteOutput
mkdir mobilenetv2-oi/tfliteOutput
leip convert --input_path models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint --framework tflite --output_path mobilenetv2-oi/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared-workdir/datasets/open-images-10-classes/eval/eval/Apple/06e47f3aa0036947.jpg --preprocessor float32
leip evaluate --output_path mobilenetv2-oi/tfliteOutput --framework tflite --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path mobilenetv2-oi/tfliteOutput/model_save/inference_model.cast.tflite --test_path datasets/open-images-10-classes/eval/eval/index.txt --class_names models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor ''
# CMD#9 TfLite Asymmetric INT8 TVM
leip compile --input_path mobilenetv2-oi/tfliteOutput/model_save/inference_model.cast.tflite --input_shapes 1,224,224,3 --input_names input_1 --output_path mobilenetv2-oi/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path mobilenetv2-oi/tfliteOutput/model_save/binuint8 --framework tvm --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path mobilenetv2-oi/tfliteOutput/model_save/binuint8 --test_path datasets/open-images-10-classes/eval/eval/index.txt --class_names models/mobilenetv2/keras-open-images-10-classes-tf-checkpoint/checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor ''

# Imagenet Commands
# Preparation
leip zoo download --model_id mobilenetv2 --variant_id keras-imagenet-tf-checkpoint
rm -rf mobilenetv2-imagenet
mkdir mobilenetv2-imagenet
mkdir mobilenetv2-imagenet/baselineFp32Results
# CMD#10 Baseline FP32 TF
leip evaluate --output_path mobilenetv2-imagenet/baselineFp32Results --framework tf2 --input_path models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor imagenet --input_shapes 1,224,224,3 --input_names input_1 --output_names Logits/Softmax
# LEIP Compress ASYMMETRIC
leip compress --input_path models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint --quantizer ASYMMETRIC --bits 8 --output_path mobilenetv2-imagenet/checkpointCompressed/
# LEIP Compress POWER_OF_TWO (POW2)
leip compress --input_path models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint --quantizer POWER_OF_TWO --bits 8 --output_path mobilenetv2-imagenet/checkpointCompressedPow2/
# CMD#11 LEIP FP32 TF
leip evaluate --output_path mobilenetv2-imagenet/checkpointCompressed/ --framework tf2 --input_path mobilenetv2-imagenet/checkpointCompressed/model_save/ --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor imagenet --input_shapes 1,224,224,3 --input_names input_1 --output_names Logits/Softmax
# CMD#12 Baseline INT8 TVM
rm -rf mobilenetv2-imagenet/compiled_tvm_int8
mkdir mobilenetv2-imagenet/compiled_tvm_int8
leip compile --input_path models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint --input_shapes 1,224,224,3 --output_path mobilenetv2-imagenet/compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-imagenet/compiled_tvm_int8/ --framework tvm --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path mobilenetv2-imagenet/compiled_tvm_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor imagenet
# CMD#13 Baseline FP32 TVM
rm -rf mobilenetv2-imagenet/compiled_tvm_fp32
mkdir mobilenetv2-imagenet/compiled_tvm_fp32
leip compile --input_path models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint --input_shapes 1,224,224,3 --output_path mobilenetv2-imagenet/compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv2-imagenet/compiled_tvm_fp32/ --framework tvm --input_names input_1 --input_types=float32 --input_shapes 1,224,224,3 --input_path mobilenetv2-imagenet/compiled_tvm_fp32/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor imagenet
# CMD#14 LEIP INT8 TVM
rm -rf mobilenetv2-imagenet/leip_compiled_tvm_int8
mkdir mobilenetv2-imagenet/leip_compiled_tvm_int8
leip compile --input_path mobilenetv2-imagenet/checkpointCompressed/model_save/ --input_shapes 1,224,224,3 --output_path mobilenetv2-imagenet/leip_compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-imagenet/leip_compiled_tvm_int8 --framework tvm --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path mobilenetv2-imagenet/leip_compiled_tvm_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor imagenet
# CMD#15 LEIP FP32 TVM
rm -rf mobilenetv2-imagenet/leip_compiled_tvm_fp32
mkdir mobilenetv2-imagenet/leip_compiled_tvm_fp32
leip compile --input_path mobilenetv2-imagenet/checkpointCompressed/model_save/ --input_shapes 1,224,224,3 --output_path mobilenetv2-imagenet/leip_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv2-imagenet/leip_compiled_tvm_fp32 --framework tvm --input_names input_1 --input_types=float32 --input_shapes 1,224,224,3 --input_path mobilenetv2-imagenet/leip_compiled_tvm_fp32/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor imagenet
# CMD#16 LEIP-POW2 INT8 TVM
rm -rf mobilenetv2-imagenet/leip_compiled_tvm_int8_pow2
mkdir mobilenetv2-imagenet/leip_compiled_tvm_int8_pow2
leip compile --input_path mobilenetv2-imagenet/checkpointCompressedPow2/model_save/ --input_shapes 1,224,224,3 --output_path mobilenetv2-imagenet/leip_compiled_tvm_int8_pow2/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-imagenet/leip_compiled_tvm_int8_pow2 --framework tvm --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path mobilenetv2-imagenet/leip_compiled_tvm_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor imagenet
# CMD#17 TfLite Asymmetric INT8 TF
rm -rf mobilenetv2-imagenet/tfliteOutput
mkdir mobilenetv2-imagenet/tfliteOutput
leip convert --input_path models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint --framework tflite --output_path mobilenetv2-imagenet/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared/data/sample-models/resources/images/imagenet_images/preprocessed/ILSVRC2012_val_00000001.JPEG --preprocessor imagenet
leip evaluate --output_path mobilenetv2-imagenet/tfliteOutput --framework tflite --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path mobilenetv2-imagenet/tfliteOutput/model_save/inference_model.cast.tflite --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor ''
# CMD#18 TfLite Asymmetric INT8 TVM
leip compile --input_path mobilenetv2-imagenet/tfliteOutput/model_save/inference_model.cast.tflite --input_shapes 1,224,224,3 --input_names input_1 --output_path mobilenetv2-imagenet/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path mobilenetv2-imagenet/tfliteOutput/model_save/binuint8 --framework tvm --input_names input_1 --input_types=uint8 --input_shapes 1,224,224,3 --input_path mobilenetv2-imagenet/tfliteOutput/model_save/binuint8 --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names models/mobilenetv2/keras-imagenet-tf-checkpoint/imagenet_checkpoint/class_names.txt --task=classifier --dataset=custom --preprocessor ''

