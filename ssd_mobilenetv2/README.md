### Configure settings

```
dataset_dir:  Path to VOCdevkit dir [/VOC/VOCdevkit/]
tf_model_path: Path to saved TF model ['saved_models/tf/]
compressed_model_path: Path to compressed TF model [saved_models/tf/]
input_node: Input tensor ['input_1:0']
output_node: Output tesnor ['predictions/concat:0']
img_height: [300]
img_width: [300]
batch_size: Batch size [16]
epochs: Epochs count 
learning_rate: 0.0001
```

### Train the model

```
../utils/dev_docker_run python train.py --path_to_settings=settings/local.yaml  --pretrained_weights=weights/MobileNetV2SSD300Lite_p14-p77.hdf5
```

### Evaluate the modelтак

```
1. ../utils/dev_docker_run python prepate_eval_data.py --path_to_settings settings/local.yaml --model_checkpoints  saved_models/tf/
2. ../utils/dev_docker_run python eval.py -gt  model_evaluation/ground_truth -det model_evaluation/model_prediction --noplot
 
```

### Showcase on single example
```
../utils/dev_docker_run python  demo.py --path_to_settings settings/local.yaml  --path_to_model saved_models/compressed/ --path_to_demo_img imgs/000030.jpg
```

### Compress
```    
../utils/dev_docker_run leip compress --input_path saved_models/tf/ --output_path saved_models/compressed_tf/ --quantizer asymmetric
```

### Evaluate compressed model

```
1. ../utils/dev_docker_run  python prepate_eval_data.py --path_to_settings settings/local.yaml --model_checkpoints saved_models/compressed_tf/
2. ../utils/dev_docker_run  eval.py -gt  model_evaluation/ground_truth -det model_evaluation/model_prediction --noplot
 