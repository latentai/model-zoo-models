# Sample commands for You Look Only Once object detection task

# Download pretrained checkpoint

This command will download pretrained YOLO keras checkpoint.

`./dev_docker_run leip zoo download --model_id yolo --variant_id keras-pretrained

# Train

In order to train the model you first need to download pretrained backbone.

`./dev_docker_run leip zoo download --model_id yolo --variant_id keras-pretrained-backbone

To train the model run

`./dev_docker_run python train.py --conf config_voc_train.json`

Once training finished the directory `h5` with `voc-trained.h4` file will be created.

# Evaluate

## Evaluate pretrained keras model

`./dev_docker_run python eval.py -i dataset/VOCdevkit/VOC2007/JPEGImages/ -c config_voc_demo.json -gtforma xyrb -detformat xyrb -gt dataset/VOCdevkit/VOC2007/Annotations/ -det detections/`

## Evaluate trained keras model

`./dev_docker_run python eval.py -i dataset/VOCdevkit/VOC2007/JPEGImages/ -c config_voc_train.json -gtforma xyrb -detformat xyrb -gt dataset/VOCdevkit/VOC2007/Annotations/ -det detections/`

## Evaluate compressed tensorflow checkpoint

Once you compress the model you will have tensorflow checkpoint as a result. You can use `eval.py` to evaluate this checkpoint. For that you will need to specify additional argument `--tf_checkpoint_dir`.

`./dev_docker_run python eval.py -i dataset/VOCdevkit/VOC2007/JPEGImages/ -c config_voc_train.json -gtforma xyrb -detformat xyrb -gt dataset/VOCdevkit/VOC2007/Annotations/ -det detections/ --tf_checkpoint_dir h5/checkpoint`

where `h5/checkpoint` is the path to the directory with tensorflow checkpoint.

# Demo

Once you download (or train) the model you can run demo script. By default this scrip will create `output` directory and put all predictions there.

`python demo.py --conf config_voc_demo.json --input dataset/VOCdevkit/VOC2007/JPEGImages/000346.jpg`

You can use `config_voc_demo.json` to use pretrained model or `config_voc_train.json` if you trained model yourself as described above.

# Convert keras checkpoint to tensorflow checkpoint

`./dev_docker_run ./utils/convert_keras_model_to_checkpoint.py --input_model_path h5/voc.h5`

# LEIP part

## Compress keras checkpoint

***Asymetric***

`rm -rf tf_compressed_asym && leip compress --input_path checkpoint/ --quantizer ASYMMETRIC --bits 8 --output_path tf_compressed_asym/`

***Power of two***

`rm -rf tf_compressed_pow2/ && leip compress --input_path checkpoint/ --quantizer POWER_OF_TWO --bits 8 --output_path tf_compressed_pow2/`

## Compile checkpoints into int8

`rm -rf tf_compiled_tvm_int8 && mkdir tf_compiled_tvm_int8 && leip compile --input_path checkpoint/ --input_shapes "1, 416, 416, 3" --output_path tf_compiled_tvm_int8/bin --input_types=uint8 --data_type=int8 --input_names input_1 --output_names conv_81/BiasAdd,conv_93/BiasAdd,conv_105/BiasAdd`

`rm -rf tf_compiled_tvm_int8_asym && mkdir tf_compiled_tvm_int8_asym && leip compile --input_path tf_compressed_asym/model_save/ --input_shapes "1, 416, 416, 3" --output_path tf_compiled_tvm_int8_asym/bin --input_types=uint8 --data_type=int8 --input_names input_1 --output_names conv_81/BiasAdd,conv_93/BiasAdd,conv_105/BiasAdd`

`rm -rf tf_compiled_tvm_int8_pow2 && mkdir tf_compiled_tvm_int8_pow2 && leip compile --input_path tf_compressed_pow2/model_save/ --input_shapes "1, 416, 416, 3" --output_path tf_compiled_tvm_int8_pow2/bin --input_types=uint8 --data_type=int8 --input_names input_1 --output_names conv_81/BiasAdd,conv_93/BiasAdd,conv_105/BiasAdd`

## Compile tensorflow checkpoint into fp32

`rm -rf tf_compiled_tvm_fp32 && mkdir tf_compiled_tvm_fp32 && leip compile --input_path checkpoint/ --input_shapes "1, 416, 416, 3" --output_path tf_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names input_1 --output_names conv_81/BiasAdd,conv_93/BiasAdd,conv_105/BiasAdd`

`rm -rf tf_compiled_tvm_fp32_asym && mkdir tf_compiled_tvm_fp32_asym && leip compile --input_path tf_compressed_asym/model_save --input_shapes "1, 416, 416, 3" --output_path tf_compiled_tvm_fp32_asym/bin --input_types=float32 --data_type=float32 --input_names input_1 --output_names conv_81/BiasAdd,conv_93/BiasAdd,conv_105/BiasAdd`

`rm -rf tf_compiled_tvm_fp32_pow2 && mkdir tf_compiled_tvm_fp32_pow2 && leip compile --input_path tf_compressed_pow2/model_save --input_shapes "1, 416, 416, 3" --output_path tf_compiled_tvm_fp32_pow2/bin --input_types=float32 --data_type=float32 --input_names input_1 --output_names conv_81/BiasAdd,conv_93/BiasAdd,conv_105/BiasAdd`
