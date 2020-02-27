# Inception V3

### Docker Build

```bash
sudo docker build -t $(cat docker_image_name.txt) .
```

### Download Dataset

```bash
sudo ./dev_docker_run ./download_dataset.py
```

### Train

```bash
sudo ./dev_docker_run ./train.py
```

Quick train for testing:
```bash
sudo ./dev_docker_run ./train.py --how_many_training_steps 10
```

### Eval

sudo ./dev_docker_run ./eval.py

### Run Model Demo on command line or web browser

sudo ./dev_docker_run ./demo.py

### Quantize with LEIP SDK
Todo.

### Evaluate with LEIP SDK

Todo.