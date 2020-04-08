# MobileNetV1 SSD

## Download dataset

`./dev_docker_run leip zoo download --dataset_id pascal-voc2007 --variant_id full-dataset`

## Download pretrained checkpoints

### Tensorflow

`./dev_docker_run leip zoo download --model_id ssd_mobilenetv1 --variant_id tf-checkpoint`

### Keras

`./dev_docker_run leip zoo download --model_id ssd_mobilenetv1 --variant_id keras-checkpoint`

## Evaluate pretrained checkpoint

`./dev_docker_run python eval.py --images_dir datasets/pascal-voc2007/full-dataset/VOC2007/JPEGImages/ --weight_file models/ssd_mobilenetv1/keras-checkpoint/keras_h5/ssd300_epoch-1000.h5 -gtformat xyrb -detformat xyrb -gt datasets/pascal-voc2007/full-dataset/VOC2007/Annotations/ -det detections/`

## Train the model

(Set epochs to 1 for a quick training run.)

`./dev_docker_run python train.py --voc_dir_path datasets/pascal-voc2007/full-dataset --epochs 1000 --batch_size 5`

After last iteration the additional directory ***checkpoint*** will be created. This directory will have a tensorflow checkpoint.

## Evaluate the trained model
(Change ssd300_epoch-1000.h5 to ssd300_epoch-01.h5 if you did 1 epoch...)

`rm -rf detections/`

`./dev_docker_run python eval.py --images_dir datasets/pascal-voc2007/full-dataset/VOC2007/JPEGImages/ --weight_file ssd300_epoch-1000.h5 -gtformat xyrb -detformat xyrb -gt datasets/pascal-voc2007/full-dataset/VOC2007/Annotations/ -det detections/`

## Showcase on single example

(Change ssd300_epoch-1000.h5 to ssd300_epoch-01.h5 if you did 1 epoch...)

Person riding the horse:
`./dev_docker_run python demo.py --weight_file ssd300_epoch-1000.h5 --filename datasets/pascal-voc2007/full-dataset/VOC2007/JPEGImages/000483.jpg`

Cat:
`./dev_docker_run python demo.py --weight_file ssd300_epoch-1000.h5 --filename datasets/pascal-voc2007/full-dataset/VOC2007/JPEGImages/000486.jpg`

## Compress tensorflow checkpoint

`rm -rf checkpoint_compressed_asym && dev-leip-run leip compress --input_path checkpoint/ --quantizer ASYMMETRIC --bits 8 --output_path checkpoint_compressed_asym/`

`rm -rf checkpoint_compressed_pow2/ && dev-leip-run leip compress --input_path checkpoint/ --quantizer POWER_OF_TWO --bits 8 --output_path checkpoint_compressed_pow2/`

## Compile tensorflow checkpoint into int8

`rm -rf compiled_tvm_int8 && mkdir compiled_tvm_int8 && dev-leip-run leip compile --input_path checkpoint/ --input_shapes "1, 300, 300, 3" --output_path compiled_tvm_int8/bin --input_types=uint8 --data_type=int8`

`rm -rf compiled_tvm_int8 && mkdir compiled_tvm_int8 && dev-leip-run leip compile --input_path checkpoint/ --input_shapes "1, 300, 300, 3" --output_path compiled_tvm_int8/bin --input_types=uint8 --data_type=int8 --input_names input_1 --output_names predictions/concat`

## Compile tensorflow checkpoint into fp32

`rm -rf compiled_tvm_fp32 && mkdir compiled_tvm_fp32 && dev-leip-run leip compile --input_path checkpoint/ --input_shapes "1, 300, 300, 3" --output_path compiled_tvm_fp32/bin --input_types=float32 --data_type=float32`

`rm -rf compiled_tvm_fp32 && mkdir compiled_tvm_fp32 && dev-leip-run leip compile --input_path checkpoint/ --input_shapes "1, 300, 300, 3" --output_path compiled_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names input_1 --output_names predictions/concat`

