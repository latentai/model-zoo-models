# Getting Started with Audio Recognition

Start by cloning this repo:
* git clone https://github.com/latentai/model-zoo-models.git
* cd audio_recognition

Once this repo is cloned locally, you can use the following commands to explore LEIP framework:

# Download dataset

./dev_docker_run leip zoo download --dataset_id google-speech-commands --variant_id v0.02

# Train the model

--how_many_training_steps 10000,10000 means 10000 high learning rate steps followed by 10000 low learning rate steps.

Set --how_many_training_steps 1,1 for a fast training run.

`./dev_docker_run python train.py --how_many_training_steps 20000,15000 --eval_step_interval 2000 --data_dir workspace/datasets/google-speech-commands//train --train_dir train_data --wanted_words up,down,left,right,one,two,three,four,five,six,seven,eight,nine,zero,go,stop,cat,dog,bird,bed,wow,sheila,happy,house,marvin,yes,no,off,on,tree`

Once model trained `train_data` directory will be created. It will contain checkpoint of trained model.

# Evaluate the model

In order to evaluate the model on test set run following command:
(Adjust the file name passed to --checkpoint if needed)

`./dev_docker_run python eval.py --checkpoint train_data/conv.ckpt-35000 --data_dir workspace/datasets/google-speech-commands/eval --wanted_words up,down,left,right,one,two,three,four,five,six,seven,eight,nine,zero,go,stop,cat,dog,bird,bed,wow,sheila,happy,house,marvin,yes,no,off,on,tree --is_test True`

# Demo

To make a prediction on wav file run following command:
(Adjust the file name passed to --checkpoint if needed)

`./dev_docker_run python demo.py --checkpoint train_data/conv.ckpt-35000 --data_dir workspace/datasets/google-speech-commands/eval --wanted_words up,down,left,right,one,two,three,four,five,six,seven,eight,nine,zero,go,stop,cat,dog,bird,bed,wow,sheila,happy,house,marvin,yes,no,off,on,tree --wav workspace/datasets/google-speech-commands/eval/cat/0c540988_nohash_0.wav`

This command will output the prediction of word "cat".

# LEIP SDK Post-Training-Quantization Commands on Pretrained Models

|       Mode        |Parameter file size (MB)|Speed (inferences/sec)|Top 1 Accuracy (%)|Top 5 Accuracy (%)|
|-------------------|-----------------------:|---------------------:|-----------------:|-----------------:|
|Original FP32      |                    8.73|                 20.08|              78.0|              97.1|
|LRE FP32 (baseline)|                    8.73|                 72.81|              82.9|              96.9|
|LRE FP32 (storage) |                    2.18|                 71.34|              82.9|              96.8|
|LRE Int8 (full)    |                    2.18|                 59.44|              48.9|              76.2|

### Preparation
```bash
leip zoo download --model_id audio-recognition --variant_id tf-baseline
leip zoo download --dataset_id google-speech-commands --variant_id eval
rm -rf audio-recognition-evaluate-variants
mkdir audio-recognition-evaluate-variants
mkdir audio-recognition-evaluate-variants/baselineFp32Results
```
### Original FP32
```bash
leip evaluate --output_path audio-recognition-evaluate-variants/baselineFp32Results --framework tf --input_path workspace/models/audio-recognition/tf-baseline --test_path workspace/datasets/google-speech-commands/eval/short_index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt
```
### LEIP Compress ASYMMETRIC
```bash
leip compress --input_path workspace/models/audio-recognition/tf-baseline --quantizer ASYMMETRIC --bits 8 --output_path audio-recognition-evaluate-variants/checkpointCompressed/
```
### LRE FP32 (baseline)
```bash
mkdir audio-recognition-evaluate-variants/compiled_lre_fp32
leip compile --input_path workspace/models/audio-recognition/tf-baseline --output_path audio-recognition-evaluate-variants/compiled_lre_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path audio-recognition-evaluate-variants/compiled_lre_fp32/ --framework lre --input_types=float32 --input_path audio-recognition-evaluate-variants/compiled_lre_fp32/bin --test_path workspace/datasets/google-speech-commands/eval/short_index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt
```
### LRE FP32 (storage)
```bash
mkdir audio-recognition-evaluate-variants/compiled_lre_int8
leip compile --input_path workspace/models/audio-recognition/tf-baseline --output_path audio-recognition-evaluate-variants/compiled_lre_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path audio-recognition-evaluate-variants/compiled_lre_int8/ --framework lre --input_types=uint8 --input_path audio-recognition-evaluate-variants/compiled_lre_int8/bin --test_path workspace/datasets/google-speech-commands/eval/short_index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt
```
### Convert model to integer
```bash
mkdir audio-recognition-evaluate-variants/tfliteOutput
leip convert --input_path workspace/models/audio-recognition/tf-baseline --framework tflite --output_path audio-recognition-evaluate-variants/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared-workdir/workspace/datasets/google-speech-commands/eval/yes/adebe223_nohash_0.wav
```
### LRE Int8 (full)
```bash
leip compile --input_path audio-recognition-evaluate-variants/tfliteOutput/model_save/inference_model.cast.tflite --output_path audio-recognition-evaluate-variants/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path audio-recognition-evaluate-variants/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path audio-recognition-evaluate-variants/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/google-speech-commands/eval/short_index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt --preprocessor speechcommand_uint8
```
