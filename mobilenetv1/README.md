# Getting Started with MobilenetV1

Start by cloning this repo:
* ```git clone https://github.com/latentai/model-zoo-models.git```
* ```cd mobilenetv1```

Once this repo is cloned locally, you can use the following commands to explore LEIP framework:


# Download pretrained model on Open Images 10 Classes
```bash
./dev_docker_run leip zoo download --model_id mobilenetv1 --variant_id keras-open-images-10-classes
```

# Download pretrained imagenet model
```bash
./dev_docker_run leip zoo download --model_id mobilenetv1 --variant_id keras-imagenet
```
# Download dataset for Transfer Learning training
```bash
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval
```
# Train

(Set --epochs and --batch_size to 1 for a quick training run.)
48 epochs produced about 80% top1 accuracy. 150 epochs would perform better.
```bash
./dev_docker_run ./train.py --dataset_path workspace/datasets/open-images-10-classes/train/  --eval_dataset_path workspace/datasets/open-images-10-classes/eval/ --epochs 150
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
|Original FP32      |                   23.69|                 21.06|              88.0|              98.7|
|LRE FP32 (baseline)|                   23.35|                 41.21|              88.0|              98.7|
|LRE FP32 (storage) |                    5.85|                 41.25|              76.7|              98.7|
|LRE Int8 (full)    |                    5.87|                 27.88|              51.3|              91.3|

### Preparation
```bash
leip zoo download --model_id mobilenetv1 --variant_id keras-open-images-10-classes
leip zoo download --dataset_id open-images-10-classes --variant_id eval
rm -rf mobilenetv1-oi
mkdir mobilenetv1-oi
mkdir mobilenetv1-oi/baselineFp32Results
```
### Original FP32
```bash
leip evaluate --output_path mobilenetv1-oi/baselineFp32Results --framework tf --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt
```
### LEIP Compress ASYMMETRIC
```bash
leip compress --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --quantizer ASYMMETRIC --bits 8 --output_path mobilenetv1-oi/checkpointCompressed/
```
### LRE FP32 (baseline)
```bash
mkdir mobilenetv1-oi/compiled_lre_fp32
leip compile --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --output_path mobilenetv1-oi/compiled_lre_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv1-oi/compiled_lre_fp32/ --framework lre --input_types=float32 --input_path mobilenetv1-oi/compiled_lre_fp32/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt
```
### LRE FP32 (storage)
```bash
mkdir mobilenetv1-oi/compiled_lre_int8
leip compile --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --output_path mobilenetv1-oi/compiled_lre_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv1-oi/compiled_lre_int8/ --framework lre --input_types=uint8 --input_path mobilenetv1-oi/compiled_lre_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt
```
### Convert model to integer
```bash
mkdir mobilenetv1-oi/tfliteOutput
leip convert --input_path workspace/models/mobilenetv1/keras-open-images-10-classes --framework tflite --output_path mobilenetv1-oi/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared-workdir/workspace/datasets/open-images-10-classes/eval/Apple/06e47f3aa0036947.jpg
```
### LRE Int8 (full)
```bash
leip compile --input_path mobilenetv1-oi/tfliteOutput/model_save/inference_model.cast.tflite --output_path mobilenetv1-oi/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path mobilenetv1-oi/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path mobilenetv1-oi/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv1/keras-open-images-10-classes/class_names.txt --preprocessor ''
```
