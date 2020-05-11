# Getting Started with VGG16

Start by cloning this repo:
* ```git clone https://github.com/latentai/model-zoo-models.git```
* ```cd vgg16```

Once this repo is cloned locally, you can use the following commands to explore LEIP framework:


# Download pretrained model on Open Images 10 Classes
```bash
./dev_docker_run leip zoo download --model_id vgg16 --variant_id keras-open-images-10-classes
```
# Download pretrained imagenet model
```bash
./dev_docker_run leip zoo download --model_id vgg16 --variant_id keras-imagenet
```
# Download dataset for Transfer Learning training
```bash
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval
```
# Train a new model with Transfer Learning on top of a base trained on Imagenet

(Set --epochs and --batch_size to 1 for a quick training run.)
```bash
./dev_docker_run ./train.py --dataset_path workspace/datasets/open-images-10-classes/train/  --eval_dataset_path workspace/datasets/open-images-10-classes/eval/ --epochs 100
```
# Evaluate a trained model
```bash
./dev_docker_run ./eval.py --dataset_path workspace/datasets/open-images-10-classes/eval/ --input_model_path trained_model/model.h5
```
# Demo

This runs inference on a single image.
```bash
./dev_docker_run ./demo.py --input_model_path trained_model/model.h5 --image_file test_images/dog.jpg
```
# LEIP SDK Post-Training-Quantization Commands on Pretrained Models

Open Image 10 Classes Commands

|       Mode        |Parameter file size (MB)|Speed (inferences/sec)|Top 1 Accuracy (%)|Top 5 Accuracy (%)|
|-------------------|-----------------------:|---------------------:|-----------------:|-----------------:|
|Original FP32      |                   58.95|                 16.21|              80.0|              99.3|
|LRE FP32 (baseline)|                   58.88|                  6.44|              80.0|              99.3|
|LRE FP32 (storage) |                   14.72|                  6.67|              80.0|              99.3|
|LRE Int8 (full)    |                   14.74|                  2.18|              69.3|              97.3|

### Preparation
```bash
leip zoo download --model_id vgg16 --variant_id keras-open-images-10-classes
leip zoo download --dataset_id open-images-10-classes --variant_id eval
rm -rf vgg16-oi
mkdir vgg16-oi
mkdir vgg16-oi/baselineFp32Results
```
### Original FP32
```bash
leip evaluate --output_path vgg16-oi/baselineFp32Results --framework tf --input_path workspace/models/vgg16/keras-open-images-10-classes --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/vgg16/keras-open-images-10-classes/class_names.txt
```
### LEIP Compress ASYMMETRIC
```bash
leip compress --input_path workspace/models/vgg16/keras-open-images-10-classes --quantizer ASYMMETRIC --bits 8 --output_path vgg16-oi/checkpointCompressed/
```
### LRE FP32 (baseline)
```bash
mkdir vgg16-oi/compiled_lre_fp32
leip compile --input_path workspace/models/vgg16/keras-open-images-10-classes --output_path vgg16-oi/compiled_lre_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path vgg16-oi/compiled_lre_fp32/ --framework lre --input_types=float32 --input_path vgg16-oi/compiled_lre_fp32/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/vgg16/keras-open-images-10-classes/class_names.txt
```
### LRE FP32 (storage)
```bash
mkdir vgg16-oi/compiled_lre_int8
leip compile --input_path workspace/models/vgg16/keras-open-images-10-classes --output_path vgg16-oi/compiled_lre_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path vgg16-oi/compiled_lre_int8/ --framework lre --input_types=uint8 --input_path vgg16-oi/compiled_lre_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/vgg16/keras-open-images-10-classes/class_names.txt
```
### Convert model to integer
```bash
mkdir vgg16-oi/tfliteOutput
leip convert --input_path workspace/models/vgg16/keras-open-images-10-classes --framework tflite --output_path vgg16-oi/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared-workdir/workspace/datasets/open-images-10-classes/eval/Apple/06e47f3aa0036947.jpg
```
### LRE Int8 (full)
```bash
leip compile --input_path vgg16-oi/tfliteOutput/model_save/inference_model.cast.tflite --output_path vgg16-oi/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path vgg16-oi/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path vgg16-oi/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/vgg16/keras-open-images-10-classes/class_names.txt --preprocessor ''
```

Imagenet Commands

|       Mode        |Parameter file size (MB)|Speed (inferences/sec)|Top 1 Accuracy (%)|Top 5 Accuracy (%)|
|-------------------|-----------------------:|---------------------:|-----------------:|-----------------:|
|Original FP32      |                   553.5|                 10.52|              68.8|              90.8|
|LRE FP32 (baseline)|                   553.4|                  7.00|              68.8|              90.8|
|LRE FP32 (storage) |                   138.4|                  6.98|              68.8|              91.2|
|LRE Int8 (full)    |                   138.4|                  2.24|              42.6|              77.5|

### Preparation
```bash
leip zoo download --model_id vgg16 --variant_id keras-imagenet
rm -rf vgg16-imagenet
mkdir vgg16-imagenet
mkdir vgg16-imagenet/baselineFp32Results
```
### Original FP32
```bash
leip evaluate --output_path vgg16-imagenet/baselineFp32Results --framework tf --input_path workspace/models/vgg16/keras-imagenet --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/vgg16/keras-imagenet/class_names.txt
```
### LEIP Compress ASYMMETRIC
```bash
leip compress --input_path workspace/models/vgg16/keras-imagenet --quantizer ASYMMETRIC --bits 8 --output_path vgg16-imagenet/checkpointCompressed/
```
### LRE FP32 (baseline)
```bash
mkdir vgg16-imagenet/compiled_lre_fp32
leip compile --input_path workspace/models/vgg16/keras-imagenet --output_path vgg16-imagenet/compiled_lre_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path vgg16-imagenet/compiled_lre_fp32/ --framework lre --input_types=float32 --input_path vgg16-imagenet/compiled_lre_fp32/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/vgg16/keras-imagenet/class_names.txt
```
### LRE FP32 (storage)
```bash
mkdir vgg16-imagenet/compiled_lre_int8
leip compile --input_path workspace/models/vgg16/keras-imagenet --output_path vgg16-imagenet/compiled_lre_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path vgg16-imagenet/compiled_lre_int8/ --framework lre --input_types=uint8 --input_path vgg16-imagenet/compiled_lre_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/vgg16/keras-imagenet/class_names.txt
```
### Convert model to integer
```bash
mkdir vgg16-imagenet/tfliteOutput
leip convert --input_path workspace/models/vgg16/keras-imagenet --framework tflite --output_path vgg16-imagenet/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared/data/sample-models/resources/images/imagenet_images/preprocessed/ILSVRC2012_val_00000001.JPEG
```
### LRE Int8 (full)
```bash
leip compile --input_path vgg16-imagenet/tfliteOutput/model_save/inference_model.cast.tflite --output_path vgg16-imagenet/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path vgg16-imagenet/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path vgg16-imagenet/tfliteOutput/model_save/binuint8 --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/vgg16/keras-imagenet/class_names.txt --preprocessor ''
```
