# Getting Started with Audio Recognition

Start by cloning this repo:
* ```git clone https://github.com/latentai/model-zoo-models.git```
* ```cd audio_recognition```

Once this repo is cloned locally, you can use the following commands to explore LEIP framework:

# Download dataset
```
./dev_docker_run leip zoo download --dataset_id google-speech-commands --variant_id v0.02
```
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

LEIP_SDK_COMMANDS_HERE