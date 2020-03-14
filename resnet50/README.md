# Download dataset

./dev_docker_run ./download_dataset.py

# Train

./dev_docker_run ./train.py --dataset_path datasets/open_images_10_classes_200/

# Convert Trained Model to TF Checkpoint format for use in LEIP SDK

./dev_docker_run ./utils/convert_keras_model_to_checkpoint.py --input_model_path trained_model.h5

# Evaluate a trained model

./dev_docker_run ./eval.py --dataset_path datasets/open_images_10_classes_200/ --input_model_path trained_model.h5

# Demo

This runs inference on a single image.
./dev_docker_run ./demo.py --input_model_path trained_model.h5 --image_file test_images/dog.jpg

# Run a converted checkpoint on a single image within LEIP SDK

Assuming your checkpoint is in "checkpoint/" after converting with ./convert_keras_model_to_checkpoint.py :

dev-leip-run leip run -in checkpoint/ --class_names class_names.txt --framework tf --preprocessor imagenet_caffe --test_path test_images/dog.jpg

# Make eval dataset index.txt file

./dev_docker_run ./utils/make_dataset_index_file.py --input_dataset_path datasets/open_images_10_classes_200/eval --output_dataset_index_path datasets/open_images_10_classes_200/eval/index.txt

# Evaluate baseline model within LEIP SDK

Ensure the index.txt has been created above...

dev-leip-run leip evaluate -fw tf -in checkpoint/ --test_path=datasets/open_images_10_classes_200/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe


# Evaluate with TVM
### Compile with TVM INT8
mkdir compiled_tvm_int8
dev-leip-run leip compile -in checkpoint/ -ishapes "1, 224, 224, 3" -o compiled_tvm_int8/bin --input_types=float32 --dump_relay=true --data_type=int8
### Run compiled model with INT8 on single image
dev-leip-run leip run -fw tvm --input_names input_1 --input_types=float32 -ishapes "1, 224, 224, 3" -in compiled_tvm_int8/bin --class_names class_names.txt --preprocessor imagenet_caffe --test_path test_images/dog.jpg
### Evaluate compiled model with INT8
dev-leip-run leip evaluate -fw tvm --input_names input_1 --input_types=float32 -ishapes "1, 224, 224, 3" -in compiled_tvm_int8/bin --test_path=datasets/open_images_10_classes_200/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe

### Compile with TVM FP32
mkdir compiled_tvm_fp32
dev-leip-run leip compile -in checkpoint/ -ishapes "1, 224, 224, 3" -o compiled_tvm_fp32/bin --input_types=float32 --dump_relay=true --data_type=float32
### Run compiled model with FP32 on single image
dev-leip-run leip run -fw tvm --input_names input_1 --input_types=float32 -ishapes "1, 224, 224, 3" -in compiled_tvm_fp32/bin --class_names class_names.txt --preprocessor imagenet_caffe --test_path test_images/dog.jpg
### Evaluate compiled model with FP32
dev-leip-run leip evaluate -fw tvm --input_names input_1 --input_types=float32 -ishapes "1, 224, 224, 3" -in compiled_tvm_fp32/bin --test_path=datasets/open_images_10_classes_200/eval/index.txt --class_names=class_names.txt --task=classifier --dataset=custom  --preprocessor imagenet_caffe
