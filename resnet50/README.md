# Build Docker Image

docker build -t $(cat docker_image_name.txt) -f Dockerfile.cpu .

OR:

docker build -t $(cat docker_image_name.txt) -f Dockerfile.gpu .

# Download dataset

./dev_docker_run ./download_dataset.py

# Train

./dev_docker_run ./train.py --dataset_path datasets/open_images_10_classes_200/

# Convert Trained Model to TF Checkpoint format for use in LEIP SDK

./dev_docker_run ./convert_keras_model_to_checkpoint.py --input_model_path trained_model.h5 --output_model_path converted
