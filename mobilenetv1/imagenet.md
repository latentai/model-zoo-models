# Save pretrained-Imagenet Model (alternative to training)
./dev_docker_run ./save_imagenet_model.py --output_model_path imagenet.h5
./dev_docker_run ./utils/convert_keras_model_to_checkpoint.py --input_model_path imagenet.h5 --output_model_path imagenet_checkpoint
