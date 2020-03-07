# Download dataset

./dev_docker_run ./download_dataset.py

# Train

./dev_docker_run ./train.py --dataset_path datasets/open_images_10_classes_200/

# Convert Trained Model to TF Checkpoint format for use in LEIP SDK

./dev_docker_run ./convert_keras_model_to_checkpoint.py --input_model_path trained_model.h5 --output_model_path converted

# Evaluate a trained model

./dev_docker_run ./eval.py --dataset_path datasets/open_images_10_classes_200/ --input_model_path trained_model.h5

# Demo

This runs inference on a single image.
./dev_docker_run ./demo.py --dataset_path datasets/open_images_10_classes_200/ --input_model_path trained_model.h5 --image_file path_to_image.jpg
