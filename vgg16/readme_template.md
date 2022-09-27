# Getting Started with VGG16

Start by cloning this repo:
* ```git clone https://github.com/latentai/model-zoo-models.git```
* ```cd vgg16```

Once this repo is cloned locally, you can use the following commands to explore LEIP framework:


# Download pretrained model on Open Images 10 Classes
```
./dev_docker_run leip zoo download --model_id vgg16 --variant_id keras-open-images-10-classes
```
# Download pretrained imagenet model
```
./dev_docker_run leip zoo download --model_id vgg16 --variant_id keras-imagenet
```
# Download dataset for Transfer Learning training
```
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval
```
# Train a new model with Transfer Learning on top of a base trained on Imagenet

(Set --epochs and --batch_size to 1 for a quick training run.)
```
./dev_docker_run ./train.py --dataset_path workspace/datasets/open-images-10-classes/train/  --eval_dataset_path workspace/datasets/open-images-10-classes/eval/ --epochs 100
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

LEIP_SDK_COMMANDS_HERE
