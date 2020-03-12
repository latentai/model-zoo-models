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
./dev_docker_run ./demo.py --input_model_path trained_model.h5 --image_file path_to_image.jpg

# Run a converted checkpoint on a single image within LEIP SDK

Assuming your checkpoint is in "checkpoint/" after converting with ./convert_keras_model_to_checkpoint.py :

dev-leip-run leip run -in checkpoint/ --class_names class_names.txt --framework tf --preprocessor imagenet_caffe --test_path path_to_image.jpg

# Make eval dataset index.txt file

./dev_docker_run ./utils/make_dataset_index_file.py --input_dataset_path datasets/open_images_10_classes_200/eval --output_dataset_index_path datasets/open_images_10_classes_200/eval/index.txt

# Evaluate baseline model within LEIP SDK

dev-leip-run leip evaluate -fw tf -in checkpoint/ --test_path=datasets/open_images_10_classes_200/eval --class_names=class_names.txt -bs 128 --task=classifier --dataset=custom --input_types=float32 -inames input --input_shapes=1,224,224,3 --preprocessor imagenet_caffe

