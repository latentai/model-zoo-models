### Configure settings

```
dataset_dir:  Path to VOCdevkit dir [/VOC/VOCdevkit/]
tf_model_path: Path to saved TF model ['saved_models/tf/]
compressed_model_path: Path to compressed TF model [saved_models/tf/]
input_node: Input tensor ['input_1:0']
output_node: Output tensor ['predictions/concat:0']
img_height: [300]
img_width: [300]
batch_size: Batch size [16]
epochs: Epochs count 
learning_rate: 0.0001
```

Download dataset:

`./dev_docker_run leip zoo download --dataset_id pascal-voc2007 --variant_id full-dataset`

### Train the model

```
./dev_docker_run python train.py --path_to_settings settings/local.yaml
```

### Evaluate the model

```
./dev_docker_run python eval.py -gt  model_evaluation/ground_truth -det model_evaluation/model_prediction --noplot --path_to_settings settings/local.yaml
```

### Showcase on single example
```
./dev_docker_run python demo.py --path_to_settings settings/local.yaml  --path_to_model saved_models/compressed/ --path_to_demo_img imgs/000030.jpg
```


### Compress
```    
./dev_docker_run leip compress --input_path saved_models/tf/ --output_path saved_models/compressed_asymmetric/ --quantizer asymmetric
```

### Evaluate compressed model

```
./dev_docker_run python eval.py -gt  model_evaluation/ground_truth -det model_evaluation/model_prediction --noplot --path_to_settings settings/local.yaml --model_checkpoints saved_models/compressed_asymmetric/model_save/
```

### Compile

```
./dev_docker_run leip compile --input_path /model-zoo-models/ssd_mobilenetv2/saved_models/tf/ --input_shapes "1, 224, 224, 3" --output_path compiled_tvm_int8/bin --input_types=float32 --data_type=int8 --output_names predictions/concat
```

### Evaluate compiled model

```
./dev_docker_run python x86_eval.py -gt  model_evaluation/ground_truth -det model_evaluation/model_prediction --noplot --path_to_settings settings/local.yaml
```
