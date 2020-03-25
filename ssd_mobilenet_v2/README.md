### Configure settings

Dataset: s3://latentai-model-zoo/datasets/voc_dataset.zip

```
dataset_dir:  Path to VOCdevkit dir [/Data/VOC/VOCdevkit/]
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
docker run -it -v {PATH_TO_DIR}/ssd_mobilenet_v2/:/ssd_mobilenet_v2 latentaiorg/latentai-sdk python /ssd_mobilenet_v2/train.py --path_to_settings=/ssd_mobilenet_v2/settings/local.yaml  --pretrained_weights=/ssd_mobilenet_v2/weights/MobileNetV2SSD300Lite_p14-p77.hdf5
```

### Evaluate the model

```
1. docker run -it -v {PATH_TO_DIR}/ssd_mobilenet_v2/:/ssd_mobilenet_v2 latentaiorg/latentai-sdk python /ssd_mobilenet_v2/prepate_eval_data.py --path_to_settings /ssd_mobilenet_v2/settings/local.yaml --model_checkpoints  /ssd_mobilenet_v2/saved_models/tf/
2. docker run -it -v {PATH_TO_DIR}/ssd_mobilenet_v2/:/ssd_mobilenet_v2 latentaiorg/latentai-sdk python /ssd_mobilenet_v2/eval.py -gt  /ssd_mobilenet_v2/model_evaluation/ground_truth -det /ssd_mobilenet_v2/model_evaluation/model_prediction --noplot
 
```

### Showcase on single example
```
python /ssd_mobilenet_v2/demo.py --path_to_settings /ssd_mobilenet_v2/settings/local.yaml  --path_to_model /ssd_mobilenet_v2/saved_models/compressed/ --path_to_demo_img /ssd_mobilenet_v2/imgs/000030.jpg
```

### Compress
```    
docker run -it -v /home/vladyslav.hamolia/Projects/model-zoo/leip_zoo/archs/ssd_mobilenet_v2/:/ssd_mobilenet_v2 latentaiorg/latentai-sdk leip compress --input_path /ssd_mobilenet_v2/saved_models/tf/ --output_path /ssd_mobilenet_v2/saved_models/compressed_tf/ --quantizer asymmetric
```

### Evaluate compressed model

```
1. docker run -it -v {PATH_TO_DIR}/ssd_mobilenet_v2/:/ssd_mobilenet_v2 latentaiorg/latentai-sdk python /ssd_mobilenet_v2/prepate_eval_data.py --path_to_settings /ssd_mobilenet_v2/settings/local.yaml --model_checkpoints /ssd_mobilenet_v2/saved_models/compressed_tf/
2. docker run -it -v {PATH_TO_DIR}/ssd_mobilenet_v2/:/ssd_mobilenet_v2 latentaiorg/latentai-sdk python /ssd_mobilenet_v2/eval.py -gt  /ssd_mobilenet_v2/model_evaluation/ground_truth -det /ssd_mobilenet_v2/model_evaluation/model_prediction --noplot
 