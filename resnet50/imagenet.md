# Save pretrained-Imagenet Model (alternative to training)
./dev_docker_run ./save_imagenet_model.py --output_model_path imagenet.h5
./dev_docker_run ./utils/convert_keras_model_to_checkpoint.py --input_model_path imagenet.h5 --output_model_path imagenet_checkpoint

# Imagenet Baseline TF FP32
dev-leip-run leip evaluate -fw tf2 -in imagenet_checkpoint/ --test_path=/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names=/data/sample-models/resources/data/imagenet/imagenet1000.names --task=classifier --dataset=custom  --preprocessor imagenet_caffe --input_shapes 1,224,224,3 --input_names input_1 --output_names probs/Softmax
# Imagenet LEIP Compress

rm -rf checkpointCompressed checkpointCompressedPow2
dev-leip-run leip compress -in imagenet_checkpoint/ -q ASYMMETRIC -b 8 -out checkpointCompressed/
dev-leip-run leip compress -in imagenet_checkpoint/ -q POWER_OF_TWO -b 8 -out checkpointCompressedPow2/

# Imagenet LEIP TF FP32
dev-leip-run leip evaluate -fw tf2 -in checkpointCompressed/model_save/ --test_path=/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names=/data/sample-models/resources/data/imagenet/imagenet1000.names --task=classifier --dataset=custom  --preprocessor imagenet_caffe  --input_shapes 1,224,224,3 --input_names input_1 --output_names probs/Softmax

# Imagenet Baseline TVM INT8
rm -rf imagenet_compiled_tvm_int8
mkdir imagenet_compiled_tvm_int8
dev-leip-run leip compile -in imagenet_checkpoint/ -ishapes "1, 224, 224, 3" -o imagenet_compiled_tvm_int8/bin --input_types=uint8  --data_type=int8
dev-leip-run leip evaluate -fw tvm --input_names input_1 --input_types=uint8 -ishapes "1, 224, 224, 3" -in imagenet_compiled_tvm_int8/bin --test_path=/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names=/data/sample-models/resources/data/imagenet/imagenet1000.names --task=classifier --dataset=custom  --preprocessor imagenet_caffe
# Imagenet Baseline TVM FP32
rm -rf imagenet_compiled_tvm_fp32
mkdir imagenet_compiled_tvm_fp32
dev-leip-run leip compile -in imagenet_checkpoint/ -ishapes "1, 224, 224, 3" -o imagenet_compiled_tvm_fp32/bin --input_types=float32  --data_type=float32
dev-leip-run leip evaluate -fw tvm --input_names input_1 --input_types=float32 -ishapes "1, 224, 224, 3" -in imagenet_compiled_tvm_fp32/bin --test_path=/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names=/data/sample-models/resources/data/imagenet/imagenet1000.names --task=classifier --dataset=custom  --preprocessor imagenet_caffe
# Imagenet LEIP TVM INT8
rm -rf imagenet_leip_compiled_tvm_int8
mkdir imagenet_leip_compiled_tvm_int8
dev-leip-run leip compile -in checkpointCompressed/model_save/ -ishapes "1, 224, 224, 3" -o imagenet_leip_compiled_tvm_int8/bin --input_types=uint8  --data_type=int8
dev-leip-run leip evaluate -fw tvm --input_names input_1 --input_types=uint8 -ishapes "1, 224, 224, 3" -in imagenet_leip_compiled_tvm_int8/bin --test_path=/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names=/data/sample-models/resources/data/imagenet/imagenet1000.names --task=classifier --dataset=custom  --preprocessor imagenet_caffe
# Imagenet LEIP TVM FP32
rm -rf imagenet_leip_compiled_tvm_fp32
mkdir imagenet_leip_compiled_tvm_fp32
dev-leip-run leip compile -in checkpointCompressed/model_save/ -ishapes "1, 224, 224, 3" -o imagenet_leip_compiled_tvm_fp32/bin --input_types=float32  --data_type=float32
dev-leip-run leip evaluate -fw tvm --input_names input_1 --input_types=float32 -ishapes "1, 224, 224, 3" -in imagenet_leip_compiled_tvm_fp32/bin --test_path=/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names=/data/sample-models/resources/data/imagenet/imagenet1000.names --task=classifier --dataset=custom  --preprocessor imagenet_caffe
