# Getting Started with MobilenetV2-pytorch

Start by cloning this repo:
* ```git clone https://github.com/latentai/model-zoo-models.git```
* ```cd model-zoo-models/mobilenetv2_pytorch```

Once this repo is cloned locally, you can use the following commands to explore LEIP framework:


# Download pretrained model on Open Images 10 Classes
```
./dev_docker_run leip zoo download --model_id mobilenetv2 --variant_id pytorch-open-images-10-classes
```
# Download dataset for training and evaluation
```
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval
```
# Train a new model

(Set --epochs and --batch_size to 1 for a quick training run.)
```
./dev_docker_run ./train.py --dataset_path workspace/datasets/open-images-10-classes/
```
# Evaluate a trained model
```
./dev_docker_run ./eval.py --dataset_path workspace/datasets/open-images-10-classes/eval/ --input_model_path workspace/trained_model/mobilenetv2.jit.pt
```
# Demo a trained model

This runs inference on a single image.
```
./dev_docker_run ./demo.py --input_model_path workspace/trained_model/mobilenetv2.jit.pt --image_file test_images/dog.jpg --input_class_names_path workspace/datasets/open-images-10-classes/eval/class_names.txt 
```
