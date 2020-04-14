# Sample commands for You Look Only Once object detection task

# Download pretrained checkpoint

`./dev_docker_run leip zoo download --model_id yolo --variant_id keras_pretrained

# Demo

Once you download (or train) the model you can run demo script. By default this scrip will create `output` directory and put all predictions there.

`python demo.py --conf config_voc.json --input dataset/VOCdevkit/VOC2007/JPEGImages/000346.jpg`

# Convert keras checkpoint to tensorflow checkpoint

`./dev_docker_run ./utils/convert_keras_model_to_checkpoint.py --input_model_path h5/voc.h5`

# LEIP part

## Compress keras checkpoint

***Asymetric***

`rm -rf tf_compressed_asym && dev-leip-run leip compress --input_path checkpoint/ --quantizer ASYMMETRIC --bits 8 --output_path tf_compressed_asym/`

***Power of two***

`rm -rf tf_compressed_pow2/ && dev-leip-run leip compress --input_path checkpoint/ --quantizer POWER_OF_TWO --bits 8 --output_path tf_compressed_pow2/`

## Compile checkpoints into int8

`rm -rf tf_compiled_tvm_int8 && mkdir tf_compiled_tvm_int8 && dev-leip-run leip compile --input_path checkpoint/ --input_shapes "1, 224, 224, 3" --output_path tf_compiled_tvm_int8/bin --input_types=uint8 --data_type=int8 --input_names input_1 --output_names conv_105/BiasAdd`

`rm -rf tf_compiled_tvm_int8_asym && mkdir tf_compiled_tvm_int8_asym && dev-leip-run leip compile --input_path tf_compressed_asym/model_save/ --input_shapes "1, 224, 224, 3" --output_path tf_compiled_tvm_int8_asym/bin --input_types=uint8 --data_type=int8 --input_names input_1 --output_names conv_105/BiasAdd`

`rm -rf tf_compiled_tvm_int8_pow2 && mkdir tf_compiled_tvm_int8_pow2 && dev-leip-run leip compile --input_path tf_compressed_pow2/model_save/ --input_shapes "1, 224, 224, 3" --output_path tf_compiled_tvm_int8_pow2/bin --input_types=uint8 --data_type=int8 --input_names input_1 --output_names conv_105/BiasAdd`

## Compile tensorflow checkpoint into fp32

`rm -rf tf_compiled_tvm_fp32 && mkdir tf_compiled_tvm_fp32 && dev-leip-run leip compile --input_path checkpoint/ --input_shapes "1, 224, 224, 3" --output_path tf_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names input_1 --output_names conv_105/BiasAdd`

`rm -rf tf_compiled_tvm_fp32_asym && mkdir tf_compiled_tvm_fp32_asym && dev-leip-run leip compile --input_path tf_compressed_asym/model_save --input_shapes "1, 224, 224, 3" --output_path tf_compiled_tvm_fp32_asym/bin --input_types=float32 --data_type=float32 --input_names input_1 --output_names conv_105/BiasAdd`

`rm -rf tf_compiled_tvm_fp32_pow2 && mkdir tf_compiled_tvm_fp32_pow2 && dev-leip-run leip compile --input_path tf_compressed_pow2/model_save --input_shapes "1, 224, 224, 3" --output_path tf_compiled_tvm_fp32_pow2/bin --input_types=float32 --data_type=float32 --input_names input_1 --output_names conv_105/BiasAdd`
