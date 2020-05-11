# Getting Started with InceptionV3

Start by cloning this repo:
* ```git clone https://github.com/latentai/model-zoo-models.git```
* ```cd inceptionv3```

Once this repo is cloned locally, you can use the following commands to explore LEIP framework:


# Download pretrained model on Open Images 10 Classes
```bash
./dev_docker_run leip zoo download --model_id inceptionv3 --variant_id keras-open-images-10-classes
```

# Download pretrained imagenet model
```bash
./dev_docker_run leip zoo download --model_id inceptionv3 --variant_id keras-imagenet
```

# Download dataset for Transfer Learning training
```bash
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval
```

# Train a new model with Transfer Learning on top of a base trained on Imagenet

(Set --epochs to 1 for a quick training run.)
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
## Open Image 10 Classes Dataset

|       Mode        |Parameter file size (MB)|Speed (inferences/sec)|Top 1 Accuracy (%)|Top 5 Accuracy (%)|
|-------------------|-----------------------:|---------------------:|-----------------:|-----------------:|
|Original FP32      |                   88.15|                 15.11|              89.3|               100|
|LRE FP32 (baseline)|                   87.24|                 21.32|              89.3|               100|
|LRE FP32 (storage) |                   21.86|                 20.48|              89.3|               100|
|LRE Int8 (full)    |                   21.86|                  9.55|              90.0|               100|


### Preparation
```bash
leip zoo download --model_id inceptionv3 --variant_id keras-open-images-10-classes
leip zoo download --dataset_id open-images-10-classes --variant_id eval
rm -rf inceptionv3-oi
mkdir inceptionv3-oi
mkdir inceptionv3-oi/baselineFp32Results
```
### Original FP32
```bash
leip evaluate --output_path inceptionv3-oi/baselineFp32Results --framework tf --input_path workspace/models/inceptionv3/keras-open-images-10-classes --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/inceptionv3/keras-open-images-10-classes/class_names.txt
```
### LEIP Compress ASYMMETRIC
```bash
leip compress --input_path workspace/models/inceptionv3/keras-open-images-10-classes --quantizer ASYMMETRIC --bits 8 --output_path inceptionv3-oi/checkpointCompressed/
```
### LRE FP32 (baseline)
```bash
mkdir inceptionv3-oi/compiled_lre_fp32
leip compile --input_path workspace/models/inceptionv3/keras-open-images-10-classes --output_path inceptionv3-oi/compiled_lre_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path inceptionv3-oi/compiled_lre_fp32/ --framework lre --input_types=float32 --input_path inceptionv3-oi/compiled_lre_fp32/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/inceptionv3/keras-open-images-10-classes/class_names.txt
```
### LRE FP32 (storage)
```bash
mkdir inceptionv3-oi/compiled_lre_int8
leip compile --input_path workspace/models/inceptionv3/keras-open-images-10-classes --output_path inceptionv3-oi/compiled_lre_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path inceptionv3-oi/compiled_lre_int8/ --framework lre --input_types=uint8 --input_path inceptionv3-oi/compiled_lre_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/inceptionv3/keras-open-images-10-classes/class_names.txt
```
### Convert model to integer
```bash
mkdir inceptionv3-oi/tfliteOutput
leip convert --input_path workspace/models/inceptionv3/keras-open-images-10-classes --framework tflite --output_path inceptionv3-oi/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared-workdir/workspace/datasets/open-images-10-classes/eval/Apple/06e47f3aa0036947.jpg
```
### LRE Int8 (full)
```bash
leip compile --input_path inceptionv3-oi/tfliteOutput/model_save/inference_model.cast.tflite --output_path inceptionv3-oi/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path inceptionv3-oi/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path inceptionv3-oi/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/inceptionv3/keras-open-images-10-classes/class_names.txt --preprocessor ''
```

## Imagenet Dataset
|       Mode        |Parameter file size (MB)|Speed (inferences/sec)|Top 1 Accuracy (%)|Top 5 Accuracy (%)|
|-------------------|-----------------------:|---------------------:|-----------------:|-----------------:|
|Original FP32      |                   96.26|                 22.27|              68.0|              90.3|
|LRE FP32 (baseline)|                   95.36|                 31.42|              68.0|              90.3|
|LRE FP32 (storage) |                   23.88|                 32.18|              68.4|              88.7|
|LRE Int8 (full)    |                   23.89|                 11.32|              64.2|              87.7|

### Preparation
```bash
leip zoo download --model_id inceptionv3 --variant_id keras-imagenet
rm -rf inceptionv3-imagenet
mkdir inceptionv3-imagenet
mkdir inceptionv3-imagenet/baselineFp32Results
```
### Original FP32
```bash
leip evaluate --output_path inceptionv3-imagenet/baselineFp32Results --framework tf --input_path workspace/models/inceptionv3/keras-imagenet --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/inceptionv3/keras-imagenet/class_names.txt
```
### LEIP Compress ASYMMETRIC
```bash
leip compress --input_path workspace/models/inceptionv3/keras-imagenet --quantizer ASYMMETRIC --bits 8 --output_path inceptionv3-imagenet/checkpointCompressed/
```
### LRE FP32 (baseline)
```bash
mkdir inceptionv3-imagenet/compiled_lre_fp32
leip compile --input_path workspace/models/inceptionv3/keras-imagenet --output_path inceptionv3-imagenet/compiled_lre_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path inceptionv3-imagenet/compiled_lre_fp32/ --framework lre --input_types=float32 --input_path inceptionv3-imagenet/compiled_lre_fp32/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/inceptionv3/keras-imagenet/class_names.txt
```
### LRE FP32 (storage)
```bash
mkdir inceptionv3-imagenet/compiled_lre_int8
leip compile --input_path workspace/models/inceptionv3/keras-imagenet --output_path inceptionv3-imagenet/compiled_lre_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path inceptionv3-imagenet/compiled_lre_int8/ --framework lre --input_types=uint8 --input_path inceptionv3-imagenet/compiled_lre_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/inceptionv3/keras-imagenet/class_names.txt
```
### Convert model to integer
```bash
mkdir inceptionv3-imagenet/tfliteOutput
leip convert --input_path workspace/models/inceptionv3/keras-imagenet --framework tflite --output_path inceptionv3-imagenet/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared/data/sample-models/resources/images/imagenet_images/preprocessed/ILSVRC2012_val_00000001.JPEG
```
### LRE Int8 (full)
```bash
leip compile --input_path inceptionv3-imagenet/tfliteOutput/model_save/inference_model.cast.tflite --output_path inceptionv3-imagenet/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path inceptionv3-imagenet/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path inceptionv3-imagenet/tfliteOutput/model_save/binuint8 --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/inceptionv3/keras-imagenet/class_names.txt --preprocessor ''
```