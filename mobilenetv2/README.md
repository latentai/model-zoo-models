# Getting Started with MobilenetV2

Start by cloning this repo:
* ```git clone https://github.com/latentai/model-zoo-models.git```
* ```cd mobilenetv2```

Once this repo is cloned locally, you can use the following commands to explore LEIP framework:


# Download pretrained model on Open Images 10 Classes
```
./dev_docker_run leip zoo download --model_id mobilenetv2 --variant_id keras-open-images-10-classes
```
# Download pretrained imagenet model
```
./dev_docker_run leip zoo download --model_id mobilenetv2 --variant_id keras-imagenet
```
# Download dataset for Transfer Learning training
```
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval
```
# Train

(Set --epochs and --batch_size to 1 for a quick training run.)
```
./dev_docker_run ./train.py --dataset_path workspace/datasets/open-images-10-classes/train/  --eval_dataset_path workspace/datasets/open-images-10-classes/eval/ --epochs 150
```
# Evaluate a trained model
```
./dev_docker_run ./eval.py --dataset_path workspace/datasets/open-images-10-classes/eval/ --input_model_path trained_model/model.h5
```
# Demo

This runs inference on a single image.
```
./dev_docker_run ./demo.py --input_model_path trained_model/model.h5 --image_file test_images/dog.jpg
```

# LEIP SDK Post-Training-Quantization Commands on Pretrained Models

# Open Images 10-Classes SDK Commands
### Preparation
```
leip zoo download --model_id mobilenetv2 --variant_id keras-open-images-10-classes
leip zoo download --dataset_id open-images-10-classes --variant_id eval
rm -rf mobilenetv2-oi
mkdir mobilenetv2-oi
mkdir mobilenetv2-oi/baselineFp32Results
```
### Baseline
```
leip evaluate --output_path mobilenetv2-oi/baselineFp32Results --input_path workspace/models/mobilenetv2/keras-open-images-10-classes/ --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
```
### LEIP Compress Asymmetric
```
leip compress --input_path workspace/models/mobilenetv2/keras-open-images-10-classes/ --quantizer asymmetric --bits 8 --output_path mobilenetv2-oi/checkpointCompressed
leip evaluate --output_path mobilenetv2-oi/checkpointCompressed --input_path mobilenetv2-oi/checkpointCompressed/model_save --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
```
### Prepare representative dataset file
```
head -n 50 workspace/datasets/open-images-10-classes/eval/index.txt > workspace/datasets/open-images-10-classes/eval/rep_dataset.txt
mkdir mobilenetv2-oi/tfliteOutput
```
### Convert model to integer with Tensor Splitting and Bias Correction
```
leip convert --input_path workspace/models/mobilenetv2/keras-open-images-10-classes/ --output_path mobilenetv2-oi/tfliteOutputTSBC --data_type int8 --rep_dataset workspace/datasets/open-images-10-classes/eval/rep_dataset.txt --optimization tensor_splitting,bias_correction
```
### LRE Int8 (full) with optimizations
```
leip compile --input_path mobilenetv2-oi/tfliteOutputTSBC/model_save/converted_model.tflite --output_path mobilenetv2-oi/tfliteOutputTSBC/model_save/binuint8 '--input_types uint8'
leip evaluate --output_path mobilenetv2-oi/tfliteOutputTSBC/model_save/binuint8 --input_path mobilenetv2-oi/tfliteOutputTSBC/model_save/binuint8 --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt --preprocessor identity
mkdir mobilenetv2-oi/tfliteOutputTSBC_pcq
```
# Imagenet SDK Commands
### Note about Imagenet Dataset
```
To download ImageNet and prepare images for evaluation, see instructions here: https://github.com/latentai/model-zoo-models#user-content-imagenet-download--eval-preparation.
The following instructions assume the imagenet evaluation set has been prepared in this local directory: workspace/datasets/imagenet/eval
```
### Preparation
```
leip zoo download --model_id mobilenetv2 --variant_id keras-imagenet
rm -rf mobilenetv2-imagenet
mkdir mobilenetv2-imagenet
mkdir mobilenetv2-imagenet/baselineFp32Results
```
### Baseline
```
leip evaluate --output_path mobilenetv2-imagenet/baselineFp32Results --input_path workspace/models/mobilenetv2/keras-imagenet/ --test_path workspace/datasets/imagenet/eval/index.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt
```
### LEIP Compress Asymmetric
```
leip compress --input_path workspace/models/mobilenetv2/keras-imagenet/ --quantizer asymmetric --bits 8 --output_path mobilenetv2-imagenet/checkpointCompressed
leip evaluate --output_path mobilenetv2-imagenet/checkpointCompressed --input_path mobilenetv2-imagenet/checkpointCompressed/model_save --test_path workspace/datasets/imagenet/eval/index.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt
```
### Prepare representative dataset file
```
head -n 50 workspace/datasets/imagenet/eval/index.txt > workspace/datasets/imagenet/eval/rep_dataset.txt
mkdir mobilenetv2-imagenet/tfliteOutput
```
### Convert model to integer with Tensor Splitting and Bias Correction
```
leip convert --input_path workspace/models/mobilenetv2/keras-imagenet/ --output_path mobilenetv2-imagenet/tfliteOutputTSBC --data_type int8 --rep_dataset workspace/datasets/imagenet/eval/rep_dataset.txt --optimization tensor_splitting,bias_correction
```
### LRE Int8 (full) with optimizations
```
leip compile --input_path mobilenetv2-imagenet/tfliteOutputTSBC/model_save/converted_model.tflite --output_path mobilenetv2-imagenet/tfliteOutputTSBC/model_save/binuint8 '--input_types uint8'
leip evaluate --output_path mobilenetv2-imagenet/tfliteOutputTSBC/model_save/binuint8 --input_path mobilenetv2-imagenet/tfliteOutputTSBC/model_save/binuint8 --test_path workspace/datasets/imagenet/eval/index.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --preprocessor identity
mkdir mobilenetv2-imagenet/tfliteOutputTSBC_pcq
```
