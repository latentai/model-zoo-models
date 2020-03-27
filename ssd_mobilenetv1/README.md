# MobeilNetV1 SSD

## Download dataset

`./dev_docker_run leip zoo download pascal-voc2007 full-dataset`

## Train the model

`python train.py --voc_dir_path dataset/VOCdevkit --epochs 1000 --batch_size 5`

After last iteration the additional directory ***checkpoint*** will be created. This directory will have a tensorflow checkpoint.

## Evaluate the model

`python eval.py --voc_dir_path dataset/VOCdevkit --weight_file ssd300_epoch-1000.h5`

This script will evaluate the model on training metrics.

## Showcase on single example

Person riding the horse:
`python demo.py --weight_file ssd300_epoch-1000.h5 --filename dataset/VOCdevkit/VOC2007/JPEGImages/000483.jpg`

Cat:
`python demo.py --weight_file ssd300_epoch-1000.h5 --filename dataset/VOCdevkit/VOC2007/JPEGImages/000486.jpg`

## Compress tensorflow checkpoint

`rm -rf checkpoint_compressed_asym && leip compress --input_path checkpoint/ --quantizer ASYMMETRIC --bits 8 --output_path checkpoint_compressed_asym/`

`rm -rf checkpoint_compressed_pow2/ && leip compress --input_path checkpoint/ --quantizer POWER_OF_TWO --bits 8 --output_path checkpoint_compressed_pow2/`

## Compress keras checkpoint

`leip compress --input_path h5/ --quantizer ASYMMETRIC --bits 8 --output_path h5_compressed`

## Compile tensorflow checkpoint into int8

`rm -rf compiled_tvm_int8 && mkdir compiled_tvm_int8 && leip compile --input_path checkpoint/ --input_shapes "1, 300, 300, 3" --output_path compiled_tvm_int8/bin --input_types=uint8 --data_type=int8`

`rm -rf compiled_tvm_int8 && mkdir compiled_tvm_int8 && leip compile --input_path checkpoint/ --input_shapes "1, 300, 300, 3" --output_path compiled_tvm_int8/bin --input_types=uint8 --data_type=int8 -inames input_1 --output_names predictions/concat`

## Compile keras checkpoint into int8

`rm -rf compiled_h5_tvm_int8 && mkdir compiled_h5_tvm_int8 && leip compile --input_path h5/ --input_shapes "1, 300, 300, 3" --output_path compiled_h5_tvm_int8/bin --input_types=uint8 --data_type=int8 --input_names input_1 --output_names predictions/concat`

## Compile tensorflow checkpoint into fp32

`rm -rf compiled_tvm_fp32 && mkdir compiled_tvm_fp32 && leip compile --input_path checkpoint/ --input_shapes "1, 300, 300, 3" --output_path compiled_tvm_fp32/bin --input_types=float32 --data_type=float32`

`rm -rf compiled_tvm_fp32 && mkdir compiled_tvm_fp32 && leip compile --input_path checkpoint/ --input_shapes "1, 300, 300, 3" --output_path compiled_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names input_1 --output_names predictions/concat`

## Compile keras checkpoint into fp32
`rm -rf compiled_h5_tvm_fp32 && mkdir compiled_h5_tvm_fp32 && leip compile --input_path h5/ --input_shapes "1, 300, 300, 3" --output_path compiled_h5_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names input_1 --output_names predictions/concat`

