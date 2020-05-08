# Getting Started with Audio Recognition

Start by cloning this repo:
* git clone https://github.com/latentai/model-zoo-models.git
* cd audio_recognition

The following commands should "just work":

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

## Preparation

`leip zoo download --model_id audio-recognition --variant_id tf-baseline`  
`leip zoo download --dataset_id google-speech-commands --variant_id eval`  
`rm -rf audio-recognition && mkdir -p audio-recognition/baselineFp32Results`  

# CMD#1 Baseline FP32 TF
`leip evaluate --output_path audio-recognition/baselineFp32Results --framework tf --input_path workspace/models/audio-recognition/tf-baseline --test_path workspace/datasets/google-speech-commands/eval/index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt --task=classifier --dataset=custom`
# LEIP Compress ASYMMETRIC
`leip compress --input_path workspace/models/audio-recognition/tf-baseline --quantizer ASYMMETRIC --bits 8 --output_path audio-recognition/checkpointCompressed/`
# LEIP Compress POWER_OF_TWO (POW2)
`leip compress --input_path workspace/models/audio-recognition/tf-baseline --quantizer POWER_OF_TWO --bits 8 --output_path audio-recognition/checkpointCompressedPow2/`
# CMD#2 LEIP FP32 TF
`leip evaluate --output_path audio-recognition/checkpointCompressed/ --framework tf --input_path audio-recognition/checkpointCompressed/model_save/ --test_path workspace/datasets/google-speech-commands/eval/index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt --task=classifier --dataset=custom`
# CMD#3 Baseline INT8 TVM
`rm -rf audio-recognition/compiled_tvm_int8`   
`mkdir audio-recognition/compiled_tvm_int8`  
`leip compile --input_path workspace/models/audio-recognition/tf-baseline --output_path audio-recognition/compiled_tvm_int8/bin --input_types=float32 --data_type=int8`  
`leip evaluate --output_path audio-recognition/compiled_tvm_int8/ --framework lre --input_types=float32 --input_path audio-recognition/compiled_tvm_int8/bin --test_path workspace/datasets/google-speech-commands/eval/index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt --task=classifier --dataset=custom`  
# CMD#4 Baseline FP32 TVM
`rm -rf audio-recognition/compiled_tvm_fp32`   
`mkdir audio-recognition/compiled_tvm_fp32`  
`leip compile --input_path workspace/models/audio-recognition/tf-baseline --output_path audio-recognition/compiled_tvm_fp32/bin --input_types=float32 --data_type=float32`  
`leip evaluate --output_path audio-recognition/compiled_tvm_fp32/ --framework lre --input_types=float32 --input_path audio-recognition/compiled_tvm_fp32/bin --test_path workspace/datasets/google-speech-commands/eval/index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt --task=classifier --dataset=custom`  
# CMD#5 LEIP INT8 TVM
`rm -rf audio-recognition/leip_compiled_tvm_int8`  
`mkdir audio-recognition/leip_compiled_tvm_int8`  
`leip compile --input_path audio-recognition/checkpointCompressed/model_save/ --output_path audio-recognition/leip_compiled_tvm_int8/bin --input_types=float32 --data_type=int8`  
`leip evaluate --output_path audio-recognition/leip_compiled_tvm_int8 --framework lre --input_types=float32 --input_path audio-recognition/leip_compiled_tvm_int8/bin --test_path workspace/datasets/google-speech-commands/eval/index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt --task=classifier --dataset=custom`  
# CMD#6 LEIP FP32 TVM
`rm -rf audio-recognition/leip_compiled_tvm_fp32`  
`mkdir audio-recognition/leip_compiled_tvm_fp32`  
`leip compile --input_path audio-recognition/checkpointCompressed/model_save/ --output_path audio-recognition/leip_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32`  
`leip evaluate --output_path audio-recognition/leip_compiled_tvm_fp32 --framework lre --input_types=float32 --input_path audio-recognition/leip_compiled_tvm_fp32/bin --test_path workspace/datasets/google-speech-commands/eval/index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt --task=classifier --dataset=custom`  
# CMD#7 LEIP-POW2 INT8 TVM
`rm -rf audio-recognition/leip_compiled_tvm_int8_pow2`  
`mkdir audio-recognition/leip_compiled_tvm_int8_pow2`  
`leip compile --input_path audio-recognition/checkpointCompressedPow2/model_save/ --output_path audio-recognition/leip_compiled_tvm_int8_pow2/bin --input_types=float32 --data_type=int8`  
`leip evaluate --output_path audio-recognition/leip_compiled_tvm_int8_pow2 --framework lre --input_types=float32 --input_path audio-recognition/leip_compiled_tvm_int8/bin --test_path workspace/datasets/google-speech-commands/eval/index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt --task=classifier --dataset=custom`  
# CMD#8 TfLite Asymmetric INT8 TF
`rm -rf audio-recognition/tfliteOutput`   
`mkdir audio-recognition/tfliteOutput`   
`leip convert --input_path workspace/models/audio-recognition/tf-baseline --framework tflite --output_path audio-recognition/tfliteOutput --data_type uint8 --policy TfLite --rep_dataset workspace/datasets/google-speech-commands/eval/cat/0c540988_nohash_0.wav`   
`leip evaluate --output_path audio-recognition/tfliteOutput --framework tflite --input_types=uint8 --input_path audio-recognition/tfliteOutput/model_save/inference_model.cast.tflite --test_path workspace/datasets/google-speech-commands/eval/index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt  --task=classifier --dataset=custom --preprocessor=speechcommand_uint8`
# CMD#9 TfLite Asymmetric INT8 TVM
`leip compile --input_path audio-recognition/tfliteOutput/model_save/inference_model.cast.tflite --output_path audio-recognition/tfliteOutput/model_save/binuint8 --input_types=uint8`   
`leip evaluate --output_path audio-recognition/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path audio-recognition/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/google-speech-commands/eval/index.txt --class_names workspace/datasets/google-speech-commands/eval/class_names.txt  --task=classifier --dataset=custom --preprocessor=speechcommand_uint8`  