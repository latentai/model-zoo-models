# Train the model

`./dev_docker_run python train.py --how_many_training_steps 10000,10000 --eval_step_interval 2000 --data_dir dataset/ --train_dir train_data --wanted_words up,down,left,right,one,two,three,four,five,six,seven,eight,nine,zero,go,stop,cat,dog,bird,bed,wow,sheila,happy,house,marvin,yes,no,off,on,tree`

Once model trained `train_data` derictory will be created. It will contain tensorflow summaries and checkpoint of trained model.

# Evaluate the model

In order to evaluate the model on test set run following command:

`./dev_docker_run python eval.py --start_checkpoint train_data/conv.ckpt-20000 --data_dir dataset/ --train_dir train_data --wanted_wordsup,down,left,right,one,two,three,four,five,six,seven,eight,nine,zero,go,stop,cat,dog,bird,bed,wow,sheila,happy,house,marvin,yes,no,off,on,tree`

# Demo

To make a prediction on wav file run following command:

`./dev_docker_run python demo.py --start_checkpoint train_data/conv.ckpt-20000 --data_dir dataset/ --train_dir train_data --wanted_wordsup,down,left,right,one,two,three,four,five,six,seven,eight,nine,zero,go,stop,cat,dog,bird,bed,wow,sheila,happy,house,marvin,yes,no,off,on,tree --wav dataset/cat/030ec18b_nohash_1.wav`

This command will output the prediction of word "cat".

## Compress tensorflow checkpoint

***Asymetric***

`rm -rf checkpoint_compressed_asym && leip compress --input_path train_data/ --quantizer ASYMMETRIC --bits 8 --output_path checkpoint_compressed_asym/`

`python eval.py --start_checkpoint checkpoint_compressed_asym/model_save/new_model --data_dir dataset/ --train_dir train_data --wanted_wordsup,down,left,right,one,two,three,four,five,six,seven,eight,nine,zero,go,stop,cat,dog,bird,bed,wow,sheila,happy,house,marvin,yes,no,off,on,tree`

***Power of two***

`rm -rf checkpoint_compressed_pow2/ && leip compress --input_path train_data/ --quantizer POWER_OF_TWO --bits 8 --output_path checkpoint_compressed_pow2/`

`python eval.py --start_checkpoint checkpoint_compressed_pow2/model_save/new_nodel --data_dir dataset/ --train_dir train_data --wanted_wordsup,down,left,right,one,two,three,four,five,six,seven,eight,nine,zero,go,stop,cat,dog,bird,bed,wow,sheila,happy,house,marvin,yes,no,off,on,tree`

## Compile checkpoints into int8

`rm -rf compiled_tf_tvm_int8 && mkdir compiled_tf_tvm_int8 && leip compile --input_path train_data/ --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_tf_tvm_int8/bin --input_types=uint8 --data_type=int8 --input_names fingerprint_input,dropout_prob --output_names add_2`

`rm -rf compiled_asym_tvm_int8 && mkdir compiled_asym_tf_tvm_int8 && leip compile --input_path checkpoint_compressed_asym/model_save --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_asym_tf_tvm_int8/bin --input_types=uint8 --data_type=int8 --input_names fingerprint_input,dropout_prob --output_names add_2`

`rm -rf compiled_pow2_tvm_int8 && mkdir compiled_pow2_tf_tvm_int8 && leip compile --input_path checkpoint_compressed_pow2/model_save --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_pow2_tf_tvm_int8/bin --input_types=uint8 --data_type=int8 --input_names fingerprint_input,dropout_prob --output_names add_2`

## Compile tensorflow checkpoint into fp32

`rm -rf compiled_tf_tvm_fp32 && mkdir compiled_tf_tvm_fp32 && leip compile --input_path train_data/ --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_tf_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names fingerprint_input,dropout_prob --output_names add_2`

`rm -rf compiled_asym_tvm_fp32 && mkdir compiled_asym_tvm_fp32 && leip compile --input_path checkpoint_compressed_asym/model_save --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_asym_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names fingerprint_input,dropout_prob --output_names add_2`

`rm -rf compiled_pow2_tvm_fp32 && mkdir compiled_pow2_tvm_fp32 && leip compile --input_path checkpoint_compressed_pow2/model_save --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_pow2_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names fingerprint_input,dropout_prob --output_names add_2`
