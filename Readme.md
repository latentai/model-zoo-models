# Overview
The Latent AI Efficient Inference Platform (LEIP) is a modular, fully-integrated workflow for AI scientists and embedded software engineers. The LEIP software development kit (SDK) enables developers to train, quantize and deploy efficient deep neural networks.

In our model zoo, you can easily test and evaluate pretrained models that have been quantized with LEIP.

# Pretrained Models

## LeNet (Training Aware)

The classic LeNet convolutional neural network proposed by Yann LeCun, trained using Training Aware quantization.

Get started with pretrained LeNet here: https://github.com/latentai/model-zoo-models/tree/master/lenet_gtc#readme

## Mobilenet V1

Mobilenet V1 is an image classification model that implements depth-wise convolutions within the network in an effort to reduce latency on mobile devices.

Get started with Mobilenet V1 here: https://github.com/latentai/model-zoo-models/tree/master/mobilenetv1#readme

## Mobilenet V2

Mobilenet V2 is an image classification model that implements depth-wise convolutions within the network in an effort to optimize latency on mobile devices. MobilenetV2 is architecturally similar to V1, but has been further optimized to reduce latency on mobile devices.

Get started with pretrained Mobilenet V2 here: https://github.com/latentai/model-zoo-models/tree/master/mobilenetv2#readme

## Resnetv2-50

Resnetv2-50 is a convolutional neural network used for image classification that is 50 layers deep. ResNet is a residual neural network known for it's ability to learn skip functions during training, allowing it to effectively skip layers during the training process resulting in a simplflied neural network that uses fewer layers.

Get started with pretrained Resnetv2-50 here: https://github.com/latentai/model-zoo-models/tree/master/resnet50#readme

## VGG16

VGG16 is a convolution neural network with 16 layers that acheives high performance on image classifcation tasks.

Get started with pretrained VGG16 here: https://github.com/latentai/model-zoo-models/tree/master/vgg16#readme

## Inception V3

Inception V3 is a convolutional neural network developed by Google to perform image classification tasks.

Get started with pretrained Inception V3 here: https://github.com/latentai/model-zoo-models/tree/master/inceptionv3#readme

## Xception

Xception is a convolutional neural network developed by Google to perform image classification tasks.

Get started with pretrained Inception V3 here: https://github.com/latentai/model-zoo-models/tree/master/xception#readme


# Imagenet Download & Eval Preparation

**The following instructions are only needed if you want to evaluate one of our pretrained models on Imagenet

(1) Go to http://image-net.org/download-images

(2) log in or sign up in order to access the full set of original ImageNet images

(3) Download the tar file ILSVRC2012_img_val.tar into utils/imagenet within your local copy of this this repo.

(if you haven't already, git clone https://github.com/latentai/model-zoo-models.git)

(4) To extract the tar file, run the following command in your terminal within the utils/imagenet directory:

tar xf ILSVRC2012_img_val.tar -C raw_images

(5) To preprocess a set of ImageNet images for evaluation with any of the pretrained models within this model zoo, run the following command.

python3 preprocess_script.py --img_dir raw_images --out_dir preprocessed
