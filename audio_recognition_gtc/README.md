# Sample code for Audio Recognition task: Training Aware Quantization

This sample of code shows how to train the model with LEIP framework.

# Download dataset

./dev_docker_run leip zoo download --dataset_id google-speech-commands --variant_id v0.02

# Download pretrained checkpoints

`./dev_docker_run leip zoo download --model_id audio-recognition-gtc --variant_id gtc_baseline_30`

Downloaded checkpoint will have two directories: `int_model_10000` and `training_10000`

`int_model_10000` has tensorflow checkpoint with quantized model during training i.e. low precision model

`training_10000` this is a high precision model.

# Train

Training procedure for LEIP model (training aware quantization) is being done in two stages.

First stage. Train the model such that only high precision model is being trained.
Second stage. Use pretrained weights to fine tune the model to train low precision weights.

`python train.py --data_dir dataset --exp_dir prepare_hp --batch_size 512 --train_steps 1000 --eval_step_interval 200 --lambda_bit_loss 0.0 --lambda_distillation_loss 0.0`

Important moment to take into account is `lambda_bit_loss` and `lambda_distillation_loss`. These two arguments control the quantization process. For the time of preparing high precision model these argument should be set to zero. That is why bit loss and distillation loss values for training progress should be ignored.

After training process is finished we should get satisfactory results. For example accuracy has value around 60-70%.

Then we proced to second stage: train the low precision model.

The difference here is that we use pretrained high precision weights from the previous stage. That we achieve by using argument `tf_checkpoint`.

Also we set `bit_loss` and `distilation_loss` to non-zero values. For this specific task distillation loss has the value in range [0, 10] so we set lambda for it equal to 0.1. The bit loss is in range [100, 200] so we set lambda value to 0.01.

These lambda values allow us to control how much of impact the loss does for weights update.

And our train command will be as follows:

`python train.py --data_dir dataset --exp_dir prepare_lp --batch_size 512 --train_steps 1000 --eval_step_interval 200 --tf_checkpoint prepare_hp/model/training_01000/variables/variables --lambda_bit_loss 0.01 --lambda_distillation_loss 0.1`
Sample command for training

